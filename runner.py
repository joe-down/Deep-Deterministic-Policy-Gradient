from agents.agent import Agent
import numpy
import gymnasium


class Runner:
    def __init__(self, env: gymnasium.Env, agent: Agent) -> None:
        self.__env = env
        self.__agent = agent
        self.__observation: numpy.ndarray
        self.__observation, info = env.reset(seed=42)

    def close(self) -> None:
        self.__env.close()

    def step(self) -> None:
        action = self.__agent.action(self.__observation)[0]
        self.__observation, reward, terminated, truncated, info = self.__env.step(action)
        dead = terminated or truncated
        self.__agent.reward(float(reward), terminated=dead)
        if dead:
            self.__observation, info = self.__env.reset()
