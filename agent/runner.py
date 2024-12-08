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
                 ) -> None:
        self.__env = gymnasium.make(environment, render_mode=None)
        self.__seed = seed
        self.__action_formatter = action_formatter
        self.__observation: numpy.ndarray
        self.__observation, info = self.__env.reset(seed=self.__seed)

    @property
    def observation(self) -> numpy.ndarray:
        return self.__observation

    def close(self) -> None:
        self.__env.close()

    def step(self, action: numpy.ndarray) -> tuple[bool, float]:
        action = self.__action_formatter(action)
        self.__observation, reward, terminated, truncated, info = self.__env.step(action)
        reward = reward.__float__()
        dead = terminated or truncated
        if dead:
            self.__observation, info = self.__env.reset()
        return dead, reward

    def run_full(self, actor: Actor) -> float:
        accumulated_reward = 0
        dead = False
        while not dead:
            dead, reward = self.step(
                action=actor.forward_network(observations=torch.tensor(self.__observation)).squeeze().cpu().numpy()
            )
            accumulated_reward += reward
        return accumulated_reward
