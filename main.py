import itertools
import pathlib
import typing

import numpy
import torch.cuda

from actor_critic.actor import Actor
from agent.train_agent import TrainAgent
from agent.runner import Runner
import matplotlib.pyplot
import tqdm


@torch.inference_mode()
def validation_run(
        load_path: pathlib.Path,
        observation_length: int,
        action_length: int,
        environment: str,
        seed: int,
        action_formatter: typing.Callable[[numpy.ndarray], numpy.ndarray],
        reward_function: typing.Callable[[numpy.ndarray, float, bool], float],
        history_size: int,
        actor_embedding_dim: int,
        actor_n_head: int,
) -> None:
    actor = Actor(
        load_path=load_path,
        observation_length=observation_length,
        action_length=action_length,
        history_size=history_size,
        embedding_dim=actor_embedding_dim,
        n_head=actor_n_head,
    )
    runner = Runner(
        environment=environment,
        seed=seed,
        action_formatter=action_formatter,
        render_mode="human",
        reward_function=reward_function,
        observation_length=observation_length,
        action_length=action_length,
        history_size=history_size,
    )
    try:
        while True:
            print(runner.run_full(actor=actor))
    except KeyboardInterrupt:
        return


def train_run(
        agent_count: int,
        validation_interval: int,
        validation_repeats: int,
        save_path: pathlib.Path,
        discount_factor: float,
        train_batch_size: int,
        history_size: int,
        buffer_size: int,
        random_action_probability: float,
        minimum_random_action_probability: float,
        random_action_probability_decay: float,
        observation_length: int,
        action_length: int,
        environment: str,
        seed: int,
        target_update_proportion: float,
        noise_variance: float,
        action_formatter: typing.Callable[[numpy.ndarray], numpy.ndarray],
        reward_function: typing.Callable[[numpy.ndarray, float, bool], float],
        sub_critic_count: int,
        actor_embedding_dim: int,
        actor_n_head: int,
        critic_embedding_dim: int,
        critic_n_head: int,
) -> None:
    train_agent = TrainAgent(train_agent_count=agent_count,
                             save_path=save_path,
                             environment=environment,
                             seed=seed + 1,
                             discount_factor=discount_factor,
                             train_batch_size=train_batch_size,
                             history_size=history_size,
                             buffer_size=buffer_size,
                             random_action_probability=random_action_probability,
                             minimum_random_action_probability=minimum_random_action_probability,
                             random_action_probability_decay=random_action_probability_decay,
                             observation_length=observation_length,
                             action_length=action_length,
                             target_update_proportion=target_update_proportion,
                             noise_variance=noise_variance,
                             action_formatter=action_formatter,
                             reward_function=reward_function,
                             sub_critic_count=sub_critic_count,
                             actor_embedding_dim=actor_embedding_dim,
                             actor_n_head=actor_n_head,
                             critic_embedding_dim=critic_embedding_dim,
                             critic_n_head=critic_n_head,
                             )
    validation_runner = Runner(
        environment=environment,
        seed=seed,
        action_formatter=action_formatter,
        reward_function=reward_function,
        observation_length=observation_length,
        action_length=action_length,
        history_size=history_size,
    )
    best_state_dicts = train_agent.state_dicts
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
            if iteration % validation_interval == 0:
                with torch.inference_mode():
                    loss_subplot.plot(losses)
                    action_loss_subplot.plot(action_losses)
                    survival_times.append(numpy.mean([validation_runner.run_full(train_agent.actor)
                                                      for _ in range(validation_repeats)]))
                    survival_times_subplot.plot(survival_times)
                    random_probabilities.append(train_agent.random_action_probabilities)
                    random_probability_subplot.plot(random_probabilities)
                    figure.canvas.draw()
                    figure.canvas.flush_events()
                    if len(survival_times) < 2 or survival_times[-1] >= max(survival_times[:-1]):
                        best_state_dicts = train_agent.state_dicts
            train_agent.step()
            q_loss, action_loss = train_agent.train(iteration=iteration)
            if q_loss is not None:
                losses.append(q_loss)
            if action_loss is not None:
                action_losses.append(action_loss)
    except KeyboardInterrupt:
        train_agent.close()
        for state_dict_index, state_dict in enumerate(best_state_dicts[0]):
            torch.save(state_dict, save_path / f"q{state_dict_index}")
        torch.save(best_state_dicts[1], save_path / "action")
        print("models saved")


def run(
        train: bool,
        agent_count: int,
        validation_interval: int,
        validation_repeats: int,
        save_path: pathlib.Path,
        discount_factor: float,
        train_batch_size: int,
        history_size: int,
        buffer_size: int,
        random_action_probability: float,
        minimum_random_action_probability: float,
        random_action_probability_decay: float,
        observation_length: int,
        action_length: int,
        environment: str,
        seed: int,
        target_update_proportion: float,
        noise_variance: float,
        action_formatter: typing.Callable[[numpy.ndarray], numpy.ndarray],
        reward_function: typing.Callable[[numpy.ndarray, float, bool], float],
        actor_embedding_dim: int,
        actor_n_head: int,
        critic_embedding_dim: int,
        critic_n_head: int,
        sub_critic_count: int,
) -> None:
    torch.set_default_device('cuda')
    if train:
        train_run(
            agent_count=agent_count,
            validation_interval=validation_interval,
            validation_repeats=validation_repeats,
            save_path=save_path,
            discount_factor=discount_factor,
            train_batch_size=train_batch_size,
            history_size=history_size,
            buffer_size=buffer_size,
            random_action_probability=random_action_probability,
            minimum_random_action_probability=minimum_random_action_probability,
            random_action_probability_decay=random_action_probability_decay,
            observation_length=observation_length,
            action_length=action_length,
            environment=environment,
            seed=seed + 1,
            target_update_proportion=target_update_proportion,
            noise_variance=noise_variance,
            action_formatter=action_formatter,
            reward_function=reward_function,
            sub_critic_count=sub_critic_count,
            actor_embedding_dim=actor_embedding_dim,
            actor_n_head=actor_n_head,
            critic_n_head=critic_n_head,
            critic_embedding_dim=critic_embedding_dim,
        )
    else:
        validation_run(
            load_path=save_path,
            observation_length=observation_length,
            action_length=action_length,
            environment=environment,
            seed=seed,
            action_formatter=action_formatter,
            reward_function=reward_function,
            history_size=history_size,
            actor_embedding_dim=actor_embedding_dim,
            actor_n_head=actor_n_head,
        )


def main(environment: str, train: bool) -> None:
    model_root = pathlib.Path("models")
    random_action_probability = 1
    minimum_random_action_probability = 0
    seed = 42
    sub_critic_count = 2

    def reward_function(observation: numpy.ndarray, reward: float, dead: bool) -> float:
        return reward

    match environment:
        case 'CartPole-v1':
            # Environment properties
            def action_formatter(action: numpy.ndarray) -> numpy.ndarray:
                return numpy.round(action.squeeze()).astype(numpy.int32)

            observation_length = 4
            action_length = 1
            # Model parameters
            actor_embedding_dim = 2 ** 8
            actor_n_head = actor_embedding_dim
            critic_embedding_dim = 2 ** 8
            critic_n_head = critic_embedding_dim
            # Train parameters
            agent_count = 2 ** 6
            train_batch_size = 2 ** 10
            history_size = 2 ** 4
            buffer_size = 2 ** 10
            validation_interval = 100
            validation_repeats = 100
            discount_factor = 0.99
            random_action_probability_decay = 0
            target_update_proportion = 2 ** -5
            noise_variance = 2 ** -3
        case 'Acrobot-v1':
            # Environment properties
            def action_formatter(action: numpy.ndarray) -> numpy.ndarray:
                return numpy.round(action * 3 - 0.5).astype(numpy.int32)

            def reward_function(observation: numpy.ndarray, reward: float, dead: bool) -> float:
                return -10 * observation[0] + observation[2] if not dead else 100

            observation_length = 6
            action_length = 1
            # Model parameters
            actor_embedding_dim = 512
            actor_n_head = actor_embedding_dim
            critic_embedding_dim = 512
            critic_n_head = critic_embedding_dim
            # Train parameters
            train_batch_size = 2 ** 6
            history_size = 2
            agent_count = 2 ** 6
            buffer_size = 2 ** 22
            validation_interval = 100
            validation_repeats = 10
            discount_factor = 0.99
            random_action_probability_decay = 0
            target_update_proportion = 1
            noise_variance = 0
        case 'BipedalWalker-v3':
            # Environment properties
            def action_formatter(action: numpy.ndarray) -> numpy.ndarray:
                return action * 2 - 1

            observation_length = 24
            action_length = 4
            # Model parameters
            actor_embedding_dim = 512
            actor_n_head = actor_embedding_dim
            critic_embedding_dim = 512
            critic_n_head = critic_embedding_dim
            # Train parameters
            train_batch_size = 2 ** 6
            history_size = 5
            agent_count = 2 ** 7
            buffer_size = 2 ** 8
            validation_interval = 100
            validation_repeats = 100
            discount_factor = 0.9
            random_action_probability_decay = 1 - 1 / 2 ** 10
            target_update_proportion = 2 ** 0
            noise_variance = 0
        case _:
            raise NotImplementedError
    if not model_root.exists():
        model_root.mkdir()
    full_model_path = model_root / environment
    if not full_model_path.exists():
        full_model_path.mkdir()
    run(train=train,
        agent_count=agent_count,
        validation_interval=validation_interval,
        validation_repeats=validation_repeats,
        save_path=full_model_path,
        discount_factor=discount_factor,
        train_batch_size=train_batch_size,
        history_size=history_size,
        buffer_size=buffer_size,
        random_action_probability=random_action_probability,
        minimum_random_action_probability=minimum_random_action_probability,
        random_action_probability_decay=random_action_probability_decay,
        environment=environment,
        seed=seed,
        observation_length=observation_length,
        action_length=action_length,
        target_update_proportion=target_update_proportion,
        noise_variance=noise_variance,
        action_formatter=action_formatter,
        reward_function=reward_function,
        actor_embedding_dim=actor_embedding_dim,
        actor_n_head=actor_n_head,
        critic_embedding_dim=critic_embedding_dim,
        critic_n_head=critic_n_head,
        sub_critic_count=sub_critic_count,
        )


if __name__ == '__main__':
    main(environment='CartPole-v1', train=True)
