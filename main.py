import itertools

import gymnasium
import numpy
import torch.cuda
from agents.super_agent import SuperAgent
from agents.runner import Runner
import matplotlib.pyplot
import tqdm


def run(train: bool,
        agent_count: int,
        validation_interval: int,
        validation_repeats: int,
        save_path: str,
        nn_width: int,
        discount_factor: float,
        train_batch_size: int,
        buffer_size: int,
        random_action_probability_decay: float,
        observation_length: int,
        action_length: int,
        environment: str,
        target_update_proportion: float) -> None:
    torch.set_default_device('cuda')
    super_agent = SuperAgent(train_agent_count=agent_count if train else 0,
                             save_path=save_path,
                             nn_width=nn_width,
                             discount_factor=discount_factor,
                             train_batch_size=train_batch_size,
                             buffer_size=buffer_size,
                             random_action_probability_decay=random_action_probability_decay,
                             observation_length=observation_length,
                             action_length=action_length,
                             target_update_proportion=target_update_proportion)
    super_runner = Runner(env=gymnasium.make(environment, render_mode=None if train else "human"),
                          agent=super_agent,
                          seed=43)
    if not train:
        try:
            while True:
                print(super_runner.run_full())
        except KeyboardInterrupt:
            super_runner.close()
            return
    else:
        runners = [Runner(env=gymnasium.make(environment, render_mode=None), agent=agent, seed=42)
                   for agent in super_agent.agents]
        for agent, random_action_minimum in zip(super_agent.agents, numpy.linspace(0, 1, len(super_agent.agents))):
            agent.minimum_random_action_probability = random_action_minimum
        best_state_dicts = super_agent.state_dicts()

        figure = matplotlib.pyplot.figure()
        loss_subplot = figure.add_subplot(2, 2, 1)
        losses = []
        action_loss_subplot = figure.add_subplot(2, 2, 2)
        action_losses = []
        survival_times_subplot = figure.add_subplot(2, 2, 3)
        survival_times = []
        random_probability_subplot = figure.add_subplot(2, 2, 4)
        random_probabilities = []
        figure.show()

        try:
            for iteration in tqdm.tqdm(itertools.count()):
                for runner in runners:
                    runner.step()
                q_loss, action_loss = super_agent.train()
                losses.append(q_loss)
                action_losses.append(action_loss)

                if iteration % validation_interval == 0:
                    loss_subplot.plot(losses)
                    action_loss_subplot.plot(action_losses)
                    survival_times.append(numpy.mean([super_runner.run_full() for _ in range(validation_repeats)]))
                    survival_times_subplot.plot(survival_times)
                    random_probabilities.append([agent.random_action_probability for agent in super_agent.agents])
                    random_probability_subplot.plot(random_probabilities)
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    if len(survival_times) < 2 or survival_times[-1] >= max(survival_times[:-1]):
                        best_state_dicts = super_agent.state_dicts()
        except KeyboardInterrupt:
            for runner in runners:
                runner.close()
            super_runner.close()
            torch.save(best_state_dicts[0], save_path + "-q")
            torch.save(best_state_dicts[1], save_path + "-action")
            print("models saved")


def main(selection: str, train: bool) -> None:
    match selection:
        case 'cartpole':
            agent_count = 2 ** 4
            validation_interval = 10
            validation_repeats = 10
            save_path = "model-cartpole"
            #nn_width = 2 ** 7
            nn_width = 2 ** 6
            discount_factor = 0.9
            train_batch_size = 2 ** 13
            buffer_size = 2 ** 25
            random_action_probability_decay = 1 - 1 / 2 ** 10
            environment = "CartPole-v1"
            observation_length = 4
            action_length = 1
            target_update_proportion = 2 ** -1
        case 'bipedal':
            agent_count = 2 ** 5
            validation_interval = 1000
            validation_repeats = 10
            save_path = "model-bipedal"
            nn_width = 2 ** 9
            discount_factor = 0.9
            train_batch_size = 2 ** 4
            buffer_size = 2 ** 15
            random_action_probability_decay = 1 - 1 / 2 ** 10
            environment = "BipedalWalker-v3"
            observation_length = 24
            action_length = 4
            target_update_proportion = 2 ** -1
        case _:
            raise NotImplementedError
    run(train=train,
        agent_count=agent_count,
        validation_interval=validation_interval,
        validation_repeats=validation_repeats,
        save_path=save_path,
        nn_width=nn_width,
        discount_factor=discount_factor,
        train_batch_size=train_batch_size,
        buffer_size=buffer_size,
        random_action_probability_decay=random_action_probability_decay,
        environment=environment,
        observation_length=observation_length,
        action_length=action_length,
        target_update_proportion=target_update_proportion)


if __name__ == '__main__':
    main(selection='cartpole', train=True)
