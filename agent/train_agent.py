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
                 discount_factor: float,
                 train_batch_size: int,
                 history_size: int,
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
                 sub_critic_count: int,
                 actor_embedding_dim: int,
                 actor_n_head: int,
                 critic_embedding_dim: int,
                 critic_n_head: int
                 ) -> None:
        self.__action_length = action_length
        self.__discount_factor = discount_factor
        self.__train_batch_size = train_batch_size
        self.__history_size = history_size
        self.__target_update_proportion = target_update_proportion
        self.__noise_variance = noise_variance
        self.__random_action_probability_decay = random_action_probability_decay
        self.__critic = Critic(
            load_path=save_path,
            observation_length=observation_length,
            action_length=action_length,
            history_size=history_size,
            sub_critic_count=sub_critic_count,
            embedding_dim=critic_embedding_dim,
            n_head=critic_n_head,
        )
        self.__actor = Actor(
            load_path=save_path,
            observation_length=observation_length,
            action_length=action_length,
            history_size=history_size,
            embedding_dim=actor_embedding_dim,
            n_head=actor_n_head,
        )
        self.__runner_observation_queues = [multiprocessing.Queue(maxsize=1) for _ in range(train_agent_count)]
        self.__runner_observation_sequence_length_queues = [multiprocessing.Queue(maxsize=1)
                                                            for _ in range(train_agent_count)]
        self.__runner_action_queues = [multiprocessing.Queue(maxsize=1) for _ in range(train_agent_count)]
        self.__runner_dead_reward_queues = [multiprocessing.Queue(maxsize=1) for _ in range(train_agent_count)]
        self.__runner_loops \
            = [multiprocessing.Process(target=self.runner_loop,
                                       args=(
                                           environment,
                                           seed + runner_index,
                                           action_formatter,
                                           observation_queue,
                                           observation_sequence_length_queue,
                                           action_queue,
                                           dead_reward_queue,
                                           reward_function,
                                           observation_length,
                                           action_length,
                                           history_size,
                                       ))
               for runner_index, (observation_queue, observation_sequence_length_queue, action_queue, dead_reward_queue)
               in enumerate(zip(
                self.__runner_observation_queues,
                self.__runner_observation_sequence_length_queues,
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
    def state_dicts(self) -> tuple[tuple[dict[str, typing.Any], ...], dict[str, typing.Any]]:
        return self.__critic.model_state_dicts, self.__actor.model_state_dict

    @property
    def random_action_probabilities(self) -> numpy.ndarray:
        return self.__random_action_probabilities.squeeze().cpu().numpy()

    @property
    def actor(self) -> Actor:
        return self.__actor

    def step(self) -> None:
        observations = torch.stack([torch.tensor(observation_queue.get())
                                    for observation_queue in self.__runner_observation_queues])
        observation_sequence_lengths = torch.stack([torch.tensor(observation_sequence_length_queue.get())
                                                    for observation_sequence_length_queue
                                                    in self.__runner_observation_sequence_length_queues])
        actor_actions = self.__actor.forward_model(
            observations=observations,
            previous_actions=self.__buffer.last_actions(history_size=self.__history_size - 1),
            observations_sequence_length=observation_sequence_lengths,
        )
        random_action_indexes = torch.rand_like(self.__random_action_probabilities) < self.__random_action_probabilities
        actions = actor_actions * ~random_action_indexes + torch.rand_like(actor_actions) * random_action_indexes
        for action, runner_action_queue in zip(actions, self.__runner_action_queues):
            runner_action_queue.put(action.squeeze().cpu().detach().numpy())
        runner_steps = [dead_reward_queue.get() for dead_reward_queue in self.__runner_dead_reward_queues]
        terminations = torch.tensor([dead for dead, reward in runner_steps])
        rewards = torch.tensor([reward for dead, reward in runner_steps])
        self.__buffer.push(
            observations=observations[..., -1, :],
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            sequence_lengths=observation_sequence_lengths
        )
        self.__random_action_probabilities = torch.maximum(input=self.__random_action_probabilities
                                                                 * self.__random_action_probability_decay,
                                                           other=self.__minimum_random_action_probabilities)

    def close(self) -> None:
        for runner in self.__runner_loops:
            runner.join()

    def train(self, iteration: int) -> tuple[float, float]:
        if not self.__buffer.ready:
            return 0, 0
        update_actor = iteration % 2 == 0  # TODO change this
        update_critic = iteration % 1 == 0  # TODO change this
        update_actor_target = iteration % 2 == 0  # TODO change this
        update_critic_target = iteration % 2 == 0  # TODO change this
        observations, actions, rewards, terminations, next_observation, sequence_lengths, next_sequence_lengths \
            = self.__buffer.random_observations(number=self.__train_batch_size, history=self.__history_size)
        loss_1 = self.__critic.update(
            actor=self.__actor,
            observations=observations.detach(),
            actions=actions.detach(),
            observations_sequence_length=sequence_lengths.detach(),
            next_observation=next_observation.detach(),
            next_observation_sequence_length=next_sequence_lengths.detach(),
            immediate_rewards=rewards.detach(),
            terminations=terminations.detach(),
            discount_factor=self.__discount_factor,
            update_target_model=update_critic_target,
            target_update_proportion=self.__target_update_proportion,
        ).__float__() if update_critic else None
        loss_2 = self.__actor.update(
            observations=observations.detach(),
            previous_actions=actions[..., :-1, :].detach(),
            observations_sequence_length=sequence_lengths.detach(),
            target_model_update_proportion=self.__target_update_proportion,
            update_target_network=update_actor_target,
            critic=self.__critic,
        ).__float__() if update_actor else None
        return loss_1, loss_2

    @staticmethod
    def runner_loop(
            environment: str,
            seed: int,
            action_formatter: typing.Callable[[numpy.ndarray], numpy.ndarray],
            observation_queue: multiprocessing.Queue,
            observation_sequence_length_queue: multiprocessing.Queue,
            action_queue: multiprocessing.Queue,
            dead_reward_queue: multiprocessing.Queue,
            reward_function: typing.Callable[[numpy.ndarray, float, bool], float],
            observation_length: int,
            action_length: int,
            history_size: int,
    ) -> None:
        runner = Runner(
            environment=environment,
            seed=seed,
            action_formatter=action_formatter,
            reward_function=reward_function,
            observation_length=observation_length,
            action_length=action_length,
            history_size=history_size,
        )
        try:
            while True:
                observation_queue.put(runner.observation)
                observation_sequence_length_queue.put(runner.observation_sequence_length)
                dead, reward, processed_reward = runner.step(action=action_queue.get())
                dead_reward_queue.put((dead, processed_reward))
        except KeyboardInterrupt:
            runner.close()
