import numpy


class BaseAgent:
    def action(self, observation: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError

    def reward(self, reward: float, terminated: bool) -> None:
        raise NotImplementedError
