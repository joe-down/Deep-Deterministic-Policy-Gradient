import gymnasium
import numpy
import torch.cuda

from agents.agent import Agent


def train(env: gymnasium.Env, agent: Agent) -> None:
    observation: numpy.ndarray
    observation, info = env.reset(seed=42)
    while True:
        action = agent.action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        dead = terminated or truncated
        agent.reward(reward=float(reward), terminated=dead)
        agent.train()
        if dead:
            observation, info = env.reset()
            agent.save()
    env.close()


def main() -> None:
    torch.set_default_device('cuda')
    env: gymnasium.Env = gymnasium.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    agent: Agent = Agent()
    train(env, agent)


if __name__ == '__main__':
    main()
