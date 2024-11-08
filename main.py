import itertools

import gymnasium
import numpy
import torch.cuda
from agents.super_agent import SuperAgent
from runner import Runner
import matplotlib.pyplot
import tqdm


def main(agent_count: int, validation_interval: int, validation_repeats: int, save_path: str) -> None:
    torch.set_default_device('cuda')
    super_agent = SuperAgent(train_agent_count=agent_count, save_path=save_path)
    runners = [Runner(env=gymnasium.make("CartPole-v1", render_mode=None), agent=agent, seed=42)
               for agent in super_agent.agents]
    for agent, random_action_minimum in zip(super_agent.agents, numpy.linspace(0, 1, len(super_agent.agents))):
        agent.MINIMUM_RANDOM_ACTION_PROBABILITY = random_action_minimum
    best_state_dict = super_agent.state_dict()

    figure = matplotlib.pyplot.figure()
    loss_subplot = figure.add_subplot(2, 2, 1)
    losses = []
    survival_times_subplot = figure.add_subplot(2, 2, 2)
    survival_times = []
    super_runner = Runner(env=gymnasium.make("CartPole-v1", render_mode=None), agent=super_agent, seed=43)
    random_probability_subplot = figure.add_subplot(2, 2, 3)
    random_probabilities = []
    figure.show()

    try:
        for iteration in tqdm.tqdm(itertools.count()):
            for runner in runners:
                runner.step()
            losses.append(super_agent.train())

            if iteration % validation_interval == 0:
                loss_subplot.plot(losses)
                survival_times.append(numpy.mean([super_runner.run_full() for _ in range(validation_repeats)]))
                survival_times_subplot.plot(survival_times)
                random_probabilities.append([agent.random_action_probability for agent in super_agent.agents])
                random_probability_subplot.plot(random_probabilities)
                figure.canvas.draw()
                figure.canvas.flush_events()
                if len(survival_times) < 2 or survival_times[-1] > max(survival_times[:-1]):
                    best_state_dict = super_agent.state_dict()
    except KeyboardInterrupt:
        for runner in runners:
            runner.close()
        if agent_count > 0:
            torch.save(best_state_dict, save_path)
            print("model saved")


if __name__ == '__main__':
    main(agent_count=2 ** 13, validation_interval=100, validation_repeats=10, save_path="model")
