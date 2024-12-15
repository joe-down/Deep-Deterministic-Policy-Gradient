import multiprocessing
import pathlib
import typing

import numpy
import torch
from actor_critic.actor import Actor
from actor_critic.critic import Critic
from agent.buffer import Buffer
from agent.runner import Runner


class TrainAgent:
    def __init__(self,
                 train_agent_count: int,
                 save_path: pathlib.Path,
                 environment: str,
                 seed: int,
                 actor_nn_width: int,
                 actor_nn_depth: int,
                 critic_nn_width: int,
                 critic_nn_depth: int,
                 discount_factor: float,
                 train_batch_size: int,
                 buffer_size: int,
                 random_action_probability: float,
                 minimum_random_action_probability: float,
                 random_action_probability_decay: float,
                 observation_length: int,
                 action_length: int,
                 target_update_proportion: float,
                 noise_variance: float,
                 action_formatter: typing.Callable[[numpy.ndarray], numpy.ndarray],
                 reward_function: typing.Callable[[numpy.ndarray, float, bool], float],
                 ) -> None:
        self.__action_length = action_length
        self.__discount_factor = discount_factor
        self.__train_batch_size = train_batch_size
        self.__target_update_proportion = target_update_proportion
        self.__noise_variance = noise_variance
        self.__random_action_probability_decay = random_action_probability_decay
        self.__critic = Critic(
            load_path=save_path,
            observation_length=observation_length,
            action_length=action_length,
            nn_width=critic_nn_width,
            nn_depth=critic_nn_depth,
        )
        self.__actor = Actor(
            load_path=save_path,
            observation_length=observation_length,
            action_length=action_length,
            nn_width=actor_nn_width,
            nn_depth=actor_nn_depth,
        )
        self.__runner_observation_queues = [multiprocessing.Queue(maxsize=1) for _ in range(train_agent_count)]
        self.__runner_action_queues = [multiprocessing.Queue(maxsize=1) for _ in range(train_agent_count)]
        self.__runner_dead_reward_queues = [multiprocessing.Queue(maxsize=1) for _ in range(train_agent_count)]
        self.__runner_loops = [multiprocessing.Process(target=self.runner_loop,
                                                       args=(
                                                           environment,
                                                           seed + runner_index,
                                                           action_formatter,
                                                           observation_queue,
                                                           action_queue,
                                                           dead_reward_queue,
                                                           reward_function,
                                                       ))
                               for runner_index, (observation_queue, action_queue, dead_reward_queue)
                               in enumerate(zip(
                self.__runner_observation_queues,
                self.__runner_action_queues,
                self.__runner_dead_reward_queues,
            ))]
        for runner in self.__runner_loops:
            runner.start()
        self.__minimum_random_action_probabilities = (torch.linspace(
            random_action_probability,
            minimum_random_action_probability,
            train_agent_count,
        ) if train_agent_count != 1 else torch.tensor([minimum_random_action_probability])).unsqueeze(dim=-1)
        self.__random_action_probabilities = torch.ones_like(self.__minimum_random_action_probabilities)
        self.__buffer = Buffer(
            train_agent_count=train_agent_count,
            observation_length=observation_length,
            action_length=self.__action_length,
            buffer_size=buffer_size,
        )

    @property
    def state_dicts(self) -> tuple[list[dict[str, typing.Any]], dict[str, typing.Any]]:
        return self.__critic.state_dicts, self.__actor.state_dict

    @property
    def random_action_probabilities(self) -> numpy.ndarray:
        return self.__random_action_probabilities.squeeze().cpu().numpy()

    @property
    def actor(self) -> Actor:
        return self.__actor

    def step(self) -> None:
        observations = torch.stack([torch.tensor(observation_queue.get())
                                    for observation_queue in self.__runner_observation_queues])
        actor_actions = self.actor.forward_network(observations=observations)
        random_action_indexes = torch.rand_like(self.__random_action_probabilities) < self.__random_action_probabilities
        actions = actor_actions * ~random_action_indexes + torch.rand_like(actor_actions) * random_action_indexes
        for action, runner_action_queue in zip(actions, self.__runner_action_queues):
            runner_action_queue.put(action.squeeze().cpu().detach().numpy())
        runner_steps = [dead_reward_queue.get() for dead_reward_queue in self.__runner_dead_reward_queues]
        terminations = torch.tensor([dead for dead, reward in runner_steps])
        rewards = torch.tensor([reward for dead, reward in runner_steps])
        self.__buffer.push(observations=observations, actions=actions, rewards=rewards, terminations=terminations)
        self.__random_action_probabilities = torch.maximum(input=self.__random_action_probabilities
                                                                 * self.__random_action_probability_decay,
                                                           other=self.__minimum_random_action_probabilities)

    def close(self) -> None:
        for runner in self.__runner_loops:
            runner.join()

    def train(self, iteration: int) -> tuple[float, float]:
        if not self.__buffer.ready:
            return 0, 0
        update_actor = True
        update_critic = True
        update_actor_target = True
        update_critic_target = True
        observations, actions, rewards, terminations, next_observations \
            = self.__buffer.random_observations(number=self.__train_batch_size)
        loss_1 = self.__critic.update(
            observation_actions=torch.concatenate((observations.detach(), actions.detach()), dim=-1),
            immediate_rewards=rewards.detach().unsqueeze(dim=-1),
            terminations=terminations.detach().unsqueeze(dim=-1),
            next_observations=next_observations.detach(),
            discount_factor=self.__discount_factor,
            noise_variance=self.__noise_variance,
            actor=self.__actor,
            target_update_proportion=self.__target_update_proportion,
            update_target_networks=update_critic_target,
        ).__float__() if update_critic else None
        loss_2 = self.__actor.update(
            observations=observations.detach(),
            target_update_proportion=self.__target_update_proportion,
            critic=self.__critic,
            update_target_network=update_actor_target,
        ).__float__() if update_actor else None
        return loss_1, loss_2

    @staticmethod
    def runner_loop(
            environment: str,
            seed: int,
            action_formatter: typing.Callable[[numpy.ndarray], numpy.ndarray],
            observation_queue: multiprocessing.Queue,
            action_queue: multiprocessing.Queue,
            dead_reward_queue: multiprocessing.Queue,
            reward_function: typing.Callable[[numpy.ndarray, float, bool], float],
    ) -> None:
        runner = Runner(
            environment=environment,
            seed=seed,
            action_formatter=action_formatter,
            reward_function=reward_function,
        )
        try:
            while True:
                observation_queue.put(runner.observation)
                dead, reward, processed_reward = runner.step(action=action_queue.get())
                dead_reward_queue.put((dead, processed_reward))
        except KeyboardInterrupt:
            runner.close()
