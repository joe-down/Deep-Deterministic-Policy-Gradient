import numpy
import torch

from agents.actor_critic.actor import Actor
from agents.basic_agent import BasicAgent
from agents.buffer import Buffer


class Agent(BasicAgent):
    def __init__(self,
                 observation_length: int,
                 action_length: int,
                 buffer_size: int,
                 random_action_probability: float,
                 minimum_random_action_probability: float,
                 random_action_probability_decay: float) -> None:
        self.__action_length = action_length
        self.__buffer = Buffer(nn_input=observation_length + self.__action_length, buffer_size=buffer_size)
        self.__random_action_probability = random_action_probability
        self.__random_action_probability_decay = random_action_probability_decay
        self.__minimum_random_action_probability = minimum_random_action_probability

    @property
    def random_action_probability(self) -> float:
        return self.__random_action_probability

    @property
    def buffer_ready(self) -> bool:
        return self.__buffer.buffer_observations_ready

    def action(self, observation: numpy.ndarray, actor: Actor) -> torch.Tensor:
        observation = torch.tensor(observation)
        if torch.rand(1) > self.__random_action_probability:
            best_action = actor.forward_network(observations=observation).detach()
        else:
            best_action = torch.rand((self.__action_length,))
        self.__buffer.push_observation(observation=torch.concatenate((observation, best_action)))
        self.__random_action_probability = max(self.__random_action_probability
                                               * self.__random_action_probability_decay,
                                               self.__minimum_random_action_probability)
        return best_action

    def reward(self, reward: float, terminated: bool) -> None:
        self.__buffer.push_reward(reward=reward, terminated=terminated)

    def random_observations(self, number: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.__buffer.random_observations(number=number)
