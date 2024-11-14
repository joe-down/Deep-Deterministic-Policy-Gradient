import typing

import torch
import numpy

from agents.actor_critic.actor import Actor
from agents.agent import Agent
from agents.base_agent import BaseAgent
from agents.actor_critic.critic import Critic


class SuperAgent(BaseAgent):
    def __init__(self,
                 train_agent_count: int,
                 save_path: str,
                 nn_width: int,
                 discount_factor: float,
                 train_batch_size: int,
                 buffer_size: int,
                 random_action_probability_decay: float,
                 observation_length: int,
                 action_length: int,
                 target_update_proportion: float) -> None:
        self.__action_length = action_length
        self.__agents = [Agent(super_agent=self,
                               observation_length=observation_length,
                               action_length=self.__action_length,
                               buffer_size=buffer_size,
                               random_action_probability_decay=random_action_probability_decay)
                         for _ in range(train_agent_count)]
        self.__discount_factor = discount_factor
        self.__train_batch_size = train_batch_size
        self.__target_update_proportion = target_update_proportion
        self.__critic = Critic(load_path=save_path,
                               observation_length=observation_length,
                               action_length=action_length,
                               nn_width=nn_width)
        self.__actor = Actor(load_path=save_path,
                             observation_length=observation_length,
                             action_length=action_length,
                             nn_width=nn_width)

    @property
    def agents(self) -> list[Agent]:
        return self.__agents

    @property
    def training(self) -> bool:
        return len(self.__agents) > 0

    def state_dicts(self) -> tuple[tuple[dict[str, typing.Any], dict[str, typing.Any]], dict[str, typing.Any]]:
        return self.__critic.state_dicts, self.__actor.state_dict

    def base_action(self, observation: numpy.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        observation = torch.tensor(observation)

        best_action = self.__actor.forward_network(observations=observation).detach()
        observation_action = torch.concatenate((observation, best_action))

        assert best_action.shape == (self.__action_length,)
        assert min(best_action) >= -1
        assert max(best_action) <= 1
        return best_action, observation_action

    def action(self, observation: numpy.ndarray) -> numpy.ndarray:
        best_action, observation_action = self.base_action(observation=observation)
        return best_action.cpu().numpy()

    def train(self) -> tuple[float, float]:
        if not self.training:
            return 0, 0
        ready_agents = torch.tensor([agent_id for agent_id, agent in enumerate(self.__agents) if agent.buffer_ready()])
        if len(ready_agents) < 1:
            return 0, 0
        batch_agents = ready_agents[torch.randint(high=len(ready_agents), size=(self.__train_batch_size,))]
        (observation_actions,
         next_observation_actions,
         immediate_rewards,
         terminations) = self.__agents[batch_agents[0]].random_observations(number=1)
        for agent_id in batch_agents[1:]:
            (current_observation_actions,
             current_next_observation_actions,
             current_immediate_rewards,
             current_terminations) = self.__agents[agent_id].random_observations(number=1)
            observation_actions = torch.concatenate((observation_actions, current_observation_actions))
            next_observation_actions = torch.concatenate((next_observation_actions, current_next_observation_actions))
            immediate_rewards = torch.concatenate((immediate_rewards, current_immediate_rewards))
            terminations = torch.concatenate((terminations, current_terminations))
        # Train step
        loss_1 = self.__critic.update(observation_actions=observation_actions,
                                      immediate_rewards=immediate_rewards,
                                      terminations=terminations,
                                      next_observations=next_observation_actions[:, :-self.__action_length],
                                      discount_factor=self.__discount_factor,
                                      actor=self.__actor)
        loss_2 = self.__actor.update(observations=observation_actions[:, :-self.__action_length],
                                     target_update_proportion=self.__target_update_proportion,
                                     critic=self.__critic)
        return float(loss_1), float(loss_2)
