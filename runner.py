import numpy
import gymnasium

from agents.base_agent import BaseAgent


class Runner:
    def __init__(self, env: gymnasium.Env, agent: BaseAgent) -> None:
        self.__env = env
        self.__agent = agent
        self.__observation: numpy.ndarray
        self.__observation, info = self.__env.reset(seed=42)

    def close(self) -> None:
        self.__env.close()

    def step(self) -> bool:
        action = self.__agent.action(self.__observation)[0]
        self.__observation, reward, terminated, truncated, info = self.__env.step(action)
        dead = terminated or truncated
        self.__agent.reward(-100 if dead else float(reward), terminated=dead)
        if dead:
            self.__observation, info = self.__env.reset()
            return False
        else:
            return True

    def run_full(self) -> int:
        self.__observation, info = self.__env.reset(seed=42)
        counter = 0
        while self.step():
            counter += 1
        return counter
