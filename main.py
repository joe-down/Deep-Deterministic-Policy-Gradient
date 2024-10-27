import gymnasium
import numpy
import torch.cuda
import matplotlib.pyplot

from agents.agent import Agent


def run(env: gymnasium.Env, agent: Agent) -> None:
    matplotlib.pyplot.ion()
    frames = 0
    figure = matplotlib.pyplot.figure()
    model_subplot = figure.add_subplot(2, 2, 1)
    loss_subplot = figure.add_subplot(2, 2, 2)
    random_action_probabilities_subplot = figure.add_subplot(2, 2, 3)
    dead_times: list[int] = []
    losses: list[float] = []
    random_action_probabilities: list[float] = []
    observation: numpy.ndarray
    observation, info = env.reset(seed=42)
    while True:
        action = agent.action(observation)[0]
        observation, reward, terminated, truncated, info = env.step(action)
        dead = terminated or truncated
        loss = agent.reward(float(reward), terminated=dead)
        if dead:
            # Log
            dead_times.append(frames)
            losses.append(loss)
            random_action_probabilities.append(agent.RANDOM_ACTION_PROBABILITY)
            if len(dead_times) % 10 == 0:
                model_subplot.plot(dead_times)
                loss_subplot.plot(losses)
                random_action_probabilities_subplot.plot(random_action_probabilities)
                figure.canvas.draw()
                figure.canvas.flush_events()
            frames = 0
            # Reset
            observation, info = env.reset()
        else:
            frames += 1


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
