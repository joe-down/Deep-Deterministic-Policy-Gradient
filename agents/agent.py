import itertools
import numpy
import torch
from agents.buffer import Buffer


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
    POSSIBLE_ACTIONS = torch.linspace(-1, 1, ACTION_COUNT)
    NN_INPUT: int = OBSERVATION_LENGTH + ACTION_LENGTH
    SAVE_PATH: str = "model"

    def __init__(self) -> None:
        self.buffer: Buffer = Buffer(nn_input=self.NN_INPUT)
        combinations = itertools.combinations_with_replacement(self.POSSIBLE_ACTIONS, self.ACTION_LENGTH)
        permutations = (torch.tensor(tuple(itertools.permutations(combination))).unique(dim=0)
                        for combination in combinations)
        self.action_space = torch.concatenate(tuple(permutations), dim=0)
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
            best_action = self.action_space[torch.randint(0, len(self.action_space), ())]
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
        if not self.buffer.buffer_observations_ready():
            return
        observation_actions, next_observation_actions, immediate_rewards, terminations \
            = self.buffer.random_observations(number=self.TRAIN_BATCH_SIZE)
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
        print(loss)

    def save(self) -> None:
        torch.save(self.neural_network.state_dict(), self.SAVE_PATH)
        print("model saved", self.RANDOM_ACTION_PROBABILITY)
