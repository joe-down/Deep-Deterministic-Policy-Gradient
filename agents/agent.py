import typing

import numpy
import torch


class Buffer:
    BUFFER_SIZE: int = 2 ** 22
    assert BUFFER_SIZE > 0

    def __init__(self, nn_input: int) -> None:
        self.observations = torch.zeros((self.BUFFER_SIZE, nn_input))
        self.observation_index = 0
        self.rewards = torch.zeros(self.BUFFER_SIZE)
        self.terminations = torch.zeros(self.BUFFER_SIZE)
        self.terminations[0] = 1
        self.reward_index = 0

    def push_observation(self, observation: torch.tensor) -> None:
        self.observation_index = (self.observation_index + 1) % self.BUFFER_SIZE
        self.observations[self.observation_index] = observation

    def push_reward(self, reward: float, terminated: bool) -> None:
        self.reward_index = (self.reward_index + 1) % self.BUFFER_SIZE
        self.rewards[self.reward_index] = reward
        self.terminations[self.reward_index] = terminated

    def random_episode(self) -> typing.Optional[tuple[torch.tensor, torch.tensor, torch.tensor]]:
        episode_boundaries = self.terminations.nonzero() + 1
        episode_count = len(episode_boundaries) - 1
        if episode_count < 1:
            return None
        episode_number = torch.randint(0, episode_count, (1,))
        episode_boundary = episode_boundaries[episode_number:episode_number + 2]
        assert episode_boundary.shape == (2, 1)
        episode_slice = slice(episode_boundary[0], episode_boundary[1])
        return self.observations[episode_slice], self.rewards[episode_slice], self.terminations[episode_slice]

    def random_observation(self) -> typing.Optional[tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]]:
        assert self.observation_index == self.reward_index
        random_episode = self.random_episode()
        if random_episode is None:
            return None
        observations, rewards, terminations = random_episode
        assert len(observations) == len(rewards)
        observation_index = torch.randint(0, len(observations), (1,))
        return observations[observation_index], observations[(observation_index + 1) % len(observations)], rewards[
            observation_index], terminations[observation_index]

    def random_observations(self, number: int) \
            -> typing.Optional[tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]]:
        observations = torch.zeros((number, 28))
        next_observations = torch.zeros((number, 28))
        rewards = torch.zeros((number, 1))
        terminations = torch.zeros((number, 1))
        for i in range(number):
            random_observation = self.random_observation()
            if random_observation is None:
                return None
            observations[i], next_observations[i], rewards[i], terminations[i] = random_observation
        return observations, next_observations, rewards, terminations


class Agent:
    RANDOM_ACTION_PROBABILITY: float = 1
    RANDOM_ACTION_PROBABILITY_DECAY: float = 1 - 1 / 2 ** 20
    assert 0 < RANDOM_ACTION_PROBABILITY_DECAY < 1
    NN_WIDTH: int = 2 ** 8
    TRAIN_BATCH_SIZE: int = 2 ** 3
    ACTION_COUNT: int = 2 ** 5
    DISCOUNT_FACTOR: float = 0.9
    assert 0 < DISCOUNT_FACTOR < 1

    OBSERVATION_LENGTH: int = 24
    ACTION_LENGTH: int = 4
    NN_INPUT: int = OBSERVATION_LENGTH + ACTION_LENGTH
    SAVE_PATH: str = "model"

    def __init__(self) -> None:
        self.buffer: Buffer = Buffer(nn_input=self.NN_INPUT)
        self.action_space: torch.tensor = torch.combinations(torch.linspace(-1, 1, self.ACTION_COUNT),
                                                             self.ACTION_LENGTH,
                                                             with_replacement=True)
        self.train_action_space: torch.tensor = self.action_space.unsqueeze(1).repeat(1, self.TRAIN_BATCH_SIZE, 1)
        self.neural_network: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(self.NN_INPUT, self.NN_WIDTH),
            torch.nn.BatchNorm1d(self.NN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(self.NN_WIDTH, self.NN_WIDTH),
            torch.nn.BatchNorm1d(self.NN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(self.NN_WIDTH, self.NN_WIDTH),
            torch.nn.BatchNorm1d(self.NN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(self.NN_WIDTH, 1),
        )
        try:
            self.neural_network.load_state_dict(torch.load(self.SAVE_PATH))
            print("model loaded")
        except FileNotFoundError:
            print("model initialised")
        self.optimiser: torch.optim.Optimizer = torch.optim.Adam(params=self.neural_network.parameters())
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

    def action(self, observation: numpy.ndarray) -> numpy.ndarray:
        assert observation.shape == (self.OBSERVATION_LENGTH,)
        observation = torch.tensor(observation)

        if torch.rand(1) > self.RANDOM_ACTION_PROBABILITY:
            observation_actions = torch.concatenate(
                (observation.repeat(self.action_space.shape[0], 1), self.action_space), 1)
            best_expected_reward_action_index = self.neural_network.forward(observation_actions).argmax()
            best_action = self.action_space[best_expected_reward_action_index]
            observation_action = observation_actions[best_expected_reward_action_index]
        else:
            best_action = torch.rand(self.ACTION_LENGTH) * 2 - 1
            observation_action = torch.concatenate((observation, best_action))

        assert observation_action.shape == (self.NN_INPUT,)
        assert best_action.shape == (self.ACTION_LENGTH,)
        assert min(best_action) >= -1
        assert max(best_action) <= 1
        self.buffer.push_observation(observation=observation_action)
        return best_action.cpu().numpy()

    def reward(self, reward: float, terminated: bool) -> None:
        self.buffer.push_reward(reward=reward, terminated=terminated)

    def train(self) -> None:
        random_observations = self.buffer.random_observations(number=self.TRAIN_BATCH_SIZE)
        if random_observations is None:
            return
        observation_actions, next_observation_actions, immediate_rewards, terminations = random_observations
        next_observations = next_observation_actions[:, :-self.ACTION_LENGTH]
        a = next_observations.repeat(self.train_action_space.shape[0], 1, 1)
        b = torch.concatenate((a, self.train_action_space), 2)
        b_flat = b.flatten(0, 1)
        c = self.neural_network(b_flat).unflatten(0, b.shape[:2]).squeeze(2)
        best_next_action_indexes = c.argmax(0)
        best_next_actions = self.action_space[best_next_action_indexes]
        best_next_observation_actions = torch.concatenate((next_observations, best_next_actions), dim=1)
        # Learn
        self.optimiser.zero_grad()
        target = immediate_rewards + self.DISCOUNT_FACTOR * (1 - terminations) * self.neural_network(
            best_next_observation_actions)
        prediction = self.neural_network(observation_actions)
        loss = self.loss_function(target, prediction)
        loss.backward()
        self.optimiser.step()
        self.RANDOM_ACTION_PROBABILITY *= self.RANDOM_ACTION_PROBABILITY_DECAY

    def save(self) -> None:
        torch.save(self.neural_network.state_dict(), self.SAVE_PATH)
        print("model saved", self.RANDOM_ACTION_PROBABILITY)
