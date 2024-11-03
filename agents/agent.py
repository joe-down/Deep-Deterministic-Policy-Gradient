import typing

import numpy
import torch
from agents.buffer import Buffer

if typing.TYPE_CHECKING:
    from agents.super_agent import SuperAgent


class Agent:
    RANDOM_ACTION_PROBABILITY: float = 1
    MINIMUM_RANDOM_ACTION_PROBABILITY: float = 1 / 100
    RANDOM_ACTION_PROBABILITY_DECAY: float = 1 - 1 / 2 ** 14
    assert 0 < RANDOM_ACTION_PROBABILITY_DECAY < 1

    def __init__(self, super_agent: "SuperAgent", observation_length: int, action_length: int) -> None:
        self.__super_agent = super_agent
        self.__observation_length = observation_length
        self.__action_length = action_length
        self.__nn_input_length = self.__observation_length + self.__action_length
        self.__buffer = Buffer(nn_input=self.__nn_input_length)

    def action(self, observation: numpy.ndarray) -> numpy.ndarray:
        assert observation.shape == (self.__observation_length,)

        if torch.rand(1) > self.RANDOM_ACTION_PROBABILITY:
            best_action, observation_action = self.__super_agent.base_action(observation=observation)
        else:
            best_action = self.__super_agent.action_space[torch.randint(0, len(self.__super_agent.action_space), ())]
            observation_action = torch.concatenate((torch.tensor(observation), best_action))

        assert best_action.shape == (self.__action_length,)
        assert observation_action.shape == (self.__nn_input_length,)
        self.__buffer.push_observation(observation=observation_action)
        self.RANDOM_ACTION_PROBABILITY = max(self.RANDOM_ACTION_PROBABILITY * self.RANDOM_ACTION_PROBABILITY_DECAY,
                                             self.MINIMUM_RANDOM_ACTION_PROBABILITY)
        print(self.RANDOM_ACTION_PROBABILITY)
        return best_action.cpu().numpy()

    def reward(self, reward: float, terminated: bool) -> None:
        self.__buffer.push_reward(reward=reward, terminated=terminated)

    def buffer_ready(self) -> bool:
        return self.__buffer.buffer_observations_ready()

    def random_observations(self, number: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.__buffer.random_observations(number=number)
