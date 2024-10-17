import gymnasium
import numpy
import torch.cuda

from agents.agent import Agent


def run(env: gymnasium.Env, agent: Agent) -> None:
    observation: numpy.ndarray
    observation, info = env.reset(seed=42)
    while True:
        action = agent.action(observation)[0]
        observation, reward, terminated, truncated, info = env.step(action)
        dead = terminated or truncated
        agent.reward(reward=-100 if dead else float(reward), terminated=dead)
        if dead:
            observation, info = env.reset()
            print("task failed")


def main(train: bool) -> None:
    torch.set_default_device('cuda')
    env: gymnasium.Env = gymnasium.make("CartPole-v1", render_mode="human")
    agent: Agent = Agent(training=train)
    try:
        run(env=env, agent=agent)
    except KeyboardInterrupt:
        env.close()
        agent.save()


if __name__ == '__main__':
    main(train=False)
