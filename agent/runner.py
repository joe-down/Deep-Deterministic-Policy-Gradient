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
                 observation_length: int,
                 action_length: int,
                 history_size: int,
                 render_mode: str = None,
                 ) -> None:
        self.__env = gymnasium.make(environment, render_mode=render_mode)
        self.__action_formatter = action_formatter
        self.__reward_function = reward_function
        self.__observation_length = observation_length
        self.__action_length = action_length
        self.__history_size = history_size

        self.__observation_history = numpy.zeros(shape=(self.__history_size, self.__observation_length))
        self.__observation_count = 0
        self.__action_history = numpy.zeros(shape=(self.__history_size, self.__action_length))
        observation: numpy.ndarray
        observation, info = self.__env.reset(seed=seed)
        self.__update_observation_history(observation=observation)

    @property
    def observation(self) -> tuple[numpy.ndarray, int]:
        return self.__observation_history, self.__observation_length

    def close(self) -> None:
        self.__env.close()

    @staticmethod
    def __next_history(
            expected_item_length: int,
            history_size: int,
            current_history: numpy.ndarray,
            next_item: numpy.ndarray,
    ) -> numpy.ndarray:
        assert expected_item_length > 0
        assert history_size > 0
        assert current_history.shape == (history_size, expected_item_length)
        assert next_item.shape == (expected_item_length,)
        next_history = numpy.concatenate((
            current_history[1:],
            numpy.expand_dims(next_item, 0),
        ))
        assert next_history.shape == (history_size, expected_item_length)
        assert numpy.all(next_history[:-1] == current_history[1:])
        assert numpy.all(next_history[-1] == next_item)
        return next_history

    def __update_observation_history(self, observation: numpy.ndarray) -> None:
        self.__observation_history = self.__next_history(
            expected_item_length=self.__observation_length,
            history_size=self.__history_size,
            current_history=self.__observation_history,
            next_item=observation,
        )
        self.__observation_count += 1

    def __update_action_history(self, action: numpy.ndarray) -> None:
        self.__action_history = self.__next_history(
            expected_item_length=self.__action_length,
            history_size=self.__history_size,
            current_history=self.__action_history,
            next_item=action,
        )

    def step(self, action: numpy.ndarray) -> tuple[bool, float, float]:
        assert action.min() >= 0
        assert action.max() <= 1
        self.__update_action_history(action=action)
        observation, reward, terminated, truncated, info = self.__env.step(self.__action_formatter(action))
        reward = reward.__float__()
        dead = terminated or truncated
        if dead:
            observation, info = self.__env.reset()
            self.__observation_count = 0
        self.__update_observation_history(observation=observation)
        return dead, reward, self.__reward_function(observation, reward, dead)

    def run_full(self, actor: Actor) -> float:
        accumulated_reward = 0
        dead = False
        while not dead:
            dead, reward, processed_reward = self.step(action=actor.forward(
                observation=torch.tensor(self.__observation_history, dtype=torch.float),
            ).cpu().numpy())
            accumulated_reward += reward
        return accumulated_reward
