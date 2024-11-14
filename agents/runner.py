import numpy
import gymnasium

from agents.base_agent import BaseAgent


class Runner:
    def __init__(self, env: gymnasium.Env, agent: BaseAgent, seed: int) -> None:
        self.__env = env
        self.__agent = agent
        self.__seed = seed
        self.__observation: numpy.ndarray
        self.__observation, info = self.__env.reset(seed=self.__seed)

    def close(self) -> None:
        self.__env.close()

    def step(self) -> bool:
        action = numpy.rint(self.__agent.action(self.__observation).squeeze()).astype(numpy.integer)
        self.__observation, reward, terminated, truncated, info = self.__env.step(action)
        dead = terminated or truncated
        self.__agent.reward(float(reward), terminated=dead)
        if dead:
            self.__observation, info = self.__env.reset()
            return False
        else:
            return True

    def run_full(self) -> float:
        observation, info = self.__env.reset()
        accumulated_reward = 0
        dead = False
        while not dead:
            action = numpy.rint(self.__agent.action(observation).squeeze()).astype(numpy.integer)
            observation, reward, terminated, truncated, info = self.__env.step(action)
            accumulated_reward += reward
            dead = terminated or truncated
        self.__observation, info = self.__env.reset()
        return float(accumulated_reward)
