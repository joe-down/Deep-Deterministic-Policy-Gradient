import itertools

import gymnasium
import torch.cuda
from agents.super_agent import SuperAgent
from runner import Runner
import matplotlib.pyplot
import tqdm


def main(agent_count: int, plot_interval: int) -> None:
    torch.set_default_device('cuda')
    super_agent = SuperAgent(train_agent_count=agent_count)
    runners = [Runner(env=gymnasium.make("CartPole-v1", render_mode=None), agent=agent)
               for agent in super_agent.agents]

    figure = matplotlib.pyplot.figure()
    loss_subplot = figure.add_subplot(1, 1, 1)
    losses = []
    figure.show()

    try:
        for iteration in tqdm.tqdm(itertools.count()):
            for runner in runners:
                runner.step()
            losses.append(super_agent.train())
            if iteration % plot_interval == 0:
                loss_subplot.plot(losses)
                figure.canvas.draw()
                figure.canvas.flush_events()
    except KeyboardInterrupt:
        for runner in runners:
            runner.close()
        super_agent.save()


if __name__ == '__main__':
    main(agent_count=2 ** 9, plot_interval=10)
