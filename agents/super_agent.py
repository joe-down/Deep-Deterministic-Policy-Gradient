import typing

import torch
import copy
import numpy

from agents.agent import Agent
from agents.base_agent import BaseAgent


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
        self.__observation_length = observation_length
        self.__action_length = action_length
        self.__agents = [Agent(super_agent=self,
                               observation_length=self.__observation_length,
                               action_length=self.__action_length,
                               buffer_size=buffer_size,
                               random_action_probability_decay=random_action_probability_decay)
                         for _ in range(train_agent_count)]
        self.__save_path = save_path
        self.__nn_width = nn_width
        self.__nn_input = self.__observation_length + self.__action_length
        self.__discount_factor = discount_factor
        self.__train_batch_size = train_batch_size
        self.__target_update_proportion = target_update_proportion

        self.__neural_network: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(self.__nn_input, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, 1),
        )
        try:
            self.__neural_network.load_state_dict(torch.load(self.__save_path + "-q"))
            print("q model loaded")
        except FileNotFoundError:
            self.__neural_network.apply(self.__neural_network_initialisation)
            print("q model initialised")
        self.__target_neural_network = copy.deepcopy(self.__neural_network)
        self.optimiser: torch.optim.Optimizer = torch.optim.Adam(params=self.__neural_network.parameters())
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

        self.__action_neural_network: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(self.__observation_length, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__action_length),
            torch.nn.Sigmoid()
        )
        try:
            self.__action_neural_network.load_state_dict(torch.load(self.__save_path + "-action"))
            print("action model loaded")
        except FileNotFoundError:
            self.__action_neural_network.apply(self.__neural_network_initialisation)
            print("action model initialised")
        self.__target_action_neural_network = copy.deepcopy(self.__action_neural_network)
        self.__action_optimiser: torch.optim.Optimizer = torch.optim.Adam(
            params=self.__action_neural_network.parameters())

    @property
    def agents(self) -> list[Agent]:
        return self.__agents

    @property
    def training(self) -> bool:
        return len(self.__agents) > 0

    @staticmethod
    def __neural_network_initialisation(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def state_dicts(self) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
        return self.__neural_network.state_dict(), self.__action_neural_network.state_dict()

    def base_action(self, observation: numpy.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        assert observation.shape == (self.__observation_length,)
        observation = torch.tensor(observation)

        best_action = self.__action_neural_network(observation).detach()
        observation_action = torch.concatenate((observation, best_action))

        assert observation_action.shape == (self.__nn_input,)
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
        loss_1 = self.__update_q_network(observation_actions=observation_actions,
                                         immediate_rewards=immediate_rewards,
                                         terminations=terminations,
                                         next_observations=next_observation_actions[:, :-self.__action_length])
        loss_2 = self.__update_action_network(observations=observation_actions[:, :-self.__action_length])

        # Update target networks
        self.__update_target_network(self.__neural_network.parameters(), self.__target_neural_network.parameters())
        self.__update_target_network(self.__action_neural_network.parameters(),
                                     self.__target_action_neural_network.parameters())
        return float(loss_1), float(loss_2)

    def __update_q_network(self,
                           observation_actions: torch.Tensor,
                           immediate_rewards: torch.Tensor,
                           terminations: torch.Tensor,
                           next_observations: torch.Tensor) -> float:
        assert next_observations.shape == (self.__train_batch_size, self.__observation_length)
        best_next_actions = self.__target_action_neural_network(next_observations).detach()
        assert best_next_actions.shape == (self.__train_batch_size, self.__action_length)
        best_next_observation_actions = torch.concatenate((next_observations, best_next_actions), dim=1)
        assert best_next_observation_actions.shape == (self.__train_batch_size, self.__nn_input)
        target = (immediate_rewards + self.__discount_factor * (1 - terminations)
                  * self.__target_neural_network(best_next_observation_actions))
        prediction = self.__neural_network(observation_actions)
        self.optimiser.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()
        self.optimiser.step()
        return float(loss)

    def __update_action_network(self, observations: torch.Tensor) -> float:
        assert observations.shape == (self.__train_batch_size, self.__observation_length)
        best_actions = self.__action_neural_network(observations)
        assert best_actions.shape == (self.__train_batch_size, self.__action_length)
        best_observation_actions = torch.concatenate((observations, best_actions), dim=1)
        assert best_observation_actions.shape == (self.__train_batch_size, self.__nn_input)
        self.__action_optimiser.zero_grad()
        loss = (-self.__neural_network(best_observation_actions)).mean()
        loss.backward()
        self.__action_optimiser.step()
        return float(loss)

    def __update_target_network(self,
                                parameters: typing.Iterator[torch.nn.Parameter],
                                target_parameters: typing.Iterator[torch.nn.Parameter]) -> None:
        for parameter, target_parameter in zip(parameters, target_parameters):
            target_parameter.data = ((1 - self.__target_update_proportion) * target_parameter.data
                                     + self.__target_update_proportion * parameter.data)
