import gymnasium
import numpy
import torch.cuda

from agents.agent import Agent


def run(env: gymnasium.Env, agent: Agent, train: bool) -> None:
    observation: numpy.ndarray
    observation, info = env.reset(seed=42)
    while True:
        action = agent.action(observation)[0]
        observation, reward, terminated, truncated, info = env.step(action)
        dead = terminated or truncated
        if train:
            agent.reward(reward=-100 if dead else float(reward), terminated=dead)
            agent.train()
        if dead:
            observation, info = env.reset()
            print("task failed")
    env.close()


def main(train: bool) -> None:
    torch.set_default_device('cuda')
    env: gymnasium.Env = gymnasium.make("CartPole-v1", render_mode="human")
    agent: Agent = Agent()
    try:
        run(env=env, agent=agent, train=train)
    except KeyboardInterrupt:
        if train:
            agent.save()


if __name__ == '__main__':
    main(train=False)
