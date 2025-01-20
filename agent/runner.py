import typing

import numpy
import gymnasium
import torch

from actor_critic.actor import Actor


class Runner:
    def __init__(self,
                 environment: str,
                 seed: int,
                 action_formatter: typing.Callable[[numpy.ndarray], numpy.ndarray],
                 reward_function: typing.Callable[[numpy.ndarray, float, bool], float],
                 history_size: int,
                 render_mode: str = None,
                 ) -> None:
        self.__env = gymnasium.make(environment, render_mode=render_mode)
        self.__seed = seed
        self.__action_formatter = action_formatter
        observation: numpy.ndarray
        observation, info = self.__env.reset(seed=self.__seed)
        self.__observations = numpy.expand_dims(observation, 0)
        self.__reward_function = reward_function
        self.__history_size = history_size

    @property
    def observations(self) -> numpy.ndarray:
        return self.__observations

    def close(self) -> None:
        self.__env.close()

    def step(self, action: numpy.ndarray) -> tuple[bool, float, float]:
        action = self.__action_formatter(action)
        observation, reward, terminated, truncated, info = self.__env.step(action)
        self.__observations = numpy.concatenate((self.__observations[-self.__history_size + 1:],
                                                 numpy.expand_dims(observation, 0)))
        assert self.__observations.shape[0] <= self.__history_size
        reward = reward.__float__()
        dead = terminated or truncated
        if dead:
            observation, info = self.__env.reset()
            self.__observations = numpy.expand_dims(observation, 0)
        return dead, reward, self.__reward_function(self.__observations[-1], reward, dead)

    def run_full(self, actor: Actor) -> float:
        accumulated_reward = 0
        dead = False
        while not dead:
            dead, reward, processed_reward = self.step(
                action=actor.forward_network(
                    observations=torch.tensor(self.__observations).unsqueeze(0)
                ).squeeze(dim=0).cpu().numpy()
            )
            accumulated_reward += reward
        return accumulated_reward
