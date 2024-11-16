import pathlib
import typing

import gymnasium
import torch
from agents.actor_critic.actor import Actor
from agents.agent import Agent
from agents.actor_critic.critic import Critic
from agents.runner import Runner


class SuperAgent:
    def __init__(self,
                 train_agent_count: int,
                 save_path: pathlib.Path,
                 environment: str,
                 seed: int,
                 actor_nn_width: int,
                 critic_nn_width: int,
                 discount_factor: float,
                 train_batch_size: int,
                 buffer_size: int,
                 random_action_probability: float,
                 minimum_random_action_probability: float,
                 random_action_probability_decay: float,
                 observation_length: int,
                 action_length: int,
                 target_update_proportion: float,
                 action_formatter: typing.Callable[[torch.Tensor], torch.Tensor],
                 ) -> None:
        self.__action_length = action_length
        self.__discount_factor = discount_factor
        self.__train_batch_size = train_batch_size
        self.__target_update_proportion = target_update_proportion

        self.__critic = Critic(
            load_path=save_path,
            observation_length=observation_length,
            action_length=action_length,
            nn_width=critic_nn_width,
        )

        self.__actor = Actor(
            load_path=save_path,
            observation_length=observation_length,
            action_length=action_length,
            nn_width=actor_nn_width,
        )

        minimum_random_action_probabilities = torch.linspace(
            random_action_probability,
            minimum_random_action_probability,
            train_agent_count,
        )

        self.__agents = [Agent(
            observation_length=observation_length,
            action_length=self.__action_length,
            buffer_size=buffer_size,
            random_action_probability=minimum_random_action_probabilities[max(0, index - 1)].item()
            if len(minimum_random_action_probabilities) > 1
            else random_action_probability,
            minimum_random_action_probability=minimum_random_action_probabilities[index].item()
            if len(minimum_random_action_probabilities) > 1
            else minimum_random_action_probability,
            random_action_probability_decay=random_action_probability_decay,
        ) for index in range(train_agent_count)]

        self.__runners = [Runner(
            env=gymnasium.make(environment, render_mode=None),
            agent=agent,
            seed=seed + agent_index,
            action_formatter=action_formatter,
        ) for agent_index, agent in enumerate(self.__agents)]

    @property
    def state_dicts(self) -> tuple[tuple[dict[str, typing.Any], dict[str, typing.Any]], dict[str, typing.Any]]:
        return self.__critic.state_dicts, self.__actor.state_dict

    @property
    def random_action_probabilities(self) -> list[float]:
        return [agent.random_action_probability for agent in self.__agents]

    @property
    def actor(self) -> Actor:
        return self.__actor

    def step(self) -> None:
        for runner in self.__runners:
            runner.step(actor=self.__actor)

    def close(self) -> None:
        for runner in self.__runners:
            runner.close()

    def train(self) -> tuple[float, float]:
        ready_agents = [agent for agent in self.__agents if agent.buffer_ready]
        if len(ready_agents) < 1:
            return 0, 0
        agent_observation_counts = torch.randint(high=len(ready_agents), size=(self.__train_batch_size,)).bincount()
        (observation_actions,
         next_observation_actions,
         immediate_rewards,
         terminations) = ready_agents[0].random_observations(number=agent_observation_counts[0].item())
        for agent, observation_count in zip(ready_agents[1:], agent_observation_counts[1:]):
            (current_observation_actions,
             current_next_observation_actions,
             current_immediate_rewards,
             current_terminations) = agent.random_observations(number=observation_count)
            observation_actions = torch.concatenate((observation_actions, current_observation_actions))
            next_observation_actions = torch.concatenate((next_observation_actions, current_next_observation_actions))
            immediate_rewards = torch.concatenate((immediate_rewards, current_immediate_rewards))
            terminations = torch.concatenate((terminations, current_terminations))

        loss_1 = self.__critic.update(
            observation_actions=observation_actions,
            immediate_rewards=immediate_rewards,
            terminations=terminations,
            next_observations=next_observation_actions[:, :-self.__action_length],
            discount_factor=self.__discount_factor,
            actor=self.__actor,
        )

        loss_2 = self.__actor.update(
            observations=observation_actions[:, :-self.__action_length],
            target_update_proportion=self.__target_update_proportion,
            critic=self.__critic,
        )

        return loss_1.__float__(), loss_2.__float__()
