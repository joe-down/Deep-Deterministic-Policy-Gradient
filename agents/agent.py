import copy
import itertools

import numpy
import torch
from agents.buffer import Buffer


class Agent:
    RANDOM_ACTION_PROBABILITY: float = 1
    MINIMUM_RANDOM_ACTION_PROBABILITY: float = 1 / 100
    RANDOM_ACTION_PROBABILITY_DECAY: float = 1 - 1 / 2 ** 14
    assert 0 < RANDOM_ACTION_PROBABILITY_DECAY < 1
    NN_WIDTH: int = 2 ** 12
    TRAIN_BATCH_SIZE: int = 2 ** 7
    DISCOUNT_FACTOR: float = 0.9
    assert 0 < DISCOUNT_FACTOR < 1
    TARGET_NETWORK_UPDATE_TIME: int = 1000

    OBSERVATION_LENGTH: int = 4
    ACTION_LENGTH: int = 1
    POSSIBLE_ACTIONS = torch.tensor([0, 1])
    NN_INPUT: int = OBSERVATION_LENGTH + ACTION_LENGTH
    SAVE_PATH: str = "model"

    def __init__(self, training: bool = False) -> None:
        # Buffer
        if training:
            self.buffer = Buffer(nn_input=self.NN_INPUT)
            print("buffer initialised")
        else:
            self.buffer = None
            print("buffer ignored (no train)")

        # Action spaces
        combinations = itertools.combinations_with_replacement(self.POSSIBLE_ACTIONS, self.ACTION_LENGTH)
        permutations = (torch.tensor(tuple(itertools.permutations(combination))).unique(dim=0)
                        for combination in combinations)
        self.action_space = torch.concatenate(tuple(permutations), dim=0)
        assert self.action_space.dim() == 2 and self.action_space.shape[1] == self.ACTION_LENGTH
        self.train_action_space: torch.tensor = self.action_space.repeat(self.TRAIN_BATCH_SIZE, 1, 1)
        assert self.train_action_space.shape == (self.TRAIN_BATCH_SIZE, self.action_space.shape[0], self.ACTION_LENGTH)

        # Neural networks
        self.neural_network: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(self.NN_INPUT, self.NN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(self.NN_WIDTH, self.NN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(self.NN_WIDTH, self.NN_WIDTH),
            torch.nn.ReLU(),
            torch.nn.Linear(self.NN_WIDTH, 1),
        )
        try:
            self.neural_network.load_state_dict(torch.load(self.SAVE_PATH))
            print("model loaded")
        except FileNotFoundError:
            self.neural_network.apply(self.neural_network_initialisation)
            print("model initialised")
        self.target_neural_network = copy.deepcopy(self.neural_network)
        self.target_neural_network_update_counter: int = 0
        self.optimiser: torch.optim.Optimizer = torch.optim.Adam(params=self.neural_network.parameters())
        self.loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

    @staticmethod
    def neural_network_initialisation(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    @property
    def training(self) -> bool:
        return self.buffer is not None

    def action(self, observation: numpy.ndarray) -> numpy.ndarray:
        assert observation.shape == (self.OBSERVATION_LENGTH,)
        observation = torch.tensor(observation)

        if not self.training or torch.rand(1) > self.RANDOM_ACTION_PROBABILITY:
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
        if self.training:
            self.buffer.push_observation(observation=observation_action)
        return best_action.cpu().numpy()

    def reward(self, reward: float, terminated: bool) -> float:
        if not self.training:
            return 0
        self.buffer.push_reward(reward=reward, terminated=terminated)
        return self.train()

    def train(self) -> float:
        if not self.training:
            return 0
        if not self.buffer.buffer_observations_ready():
            return 0
        observation_actions, next_observation_actions, immediate_rewards, terminations \
            = self.buffer.random_observations(number=self.TRAIN_BATCH_SIZE)
        next_observations = next_observation_actions[:, :-self.ACTION_LENGTH]
        assert next_observations.shape == (self.TRAIN_BATCH_SIZE, self.OBSERVATION_LENGTH)
        a = next_observations.unsqueeze(1).repeat(1, self.train_action_space.shape[1], 1)
        assert a.shape == (self.TRAIN_BATCH_SIZE, self.train_action_space.shape[1], self.OBSERVATION_LENGTH)
        b = torch.concatenate((a, self.train_action_space), 2)
        assert b.shape == (self.TRAIN_BATCH_SIZE, self.train_action_space.shape[1], self.NN_INPUT)
        b_flat = b.flatten(0, 1)
        assert b_flat.shape == (self.TRAIN_BATCH_SIZE * self.train_action_space.shape[1], self.NN_INPUT)
        b_2 = self.neural_network(b_flat)
        assert b_2.shape == (self.TRAIN_BATCH_SIZE * self.train_action_space.shape[1], 1)
        c = b_2.unflatten(0, b.shape[:2])
        assert c.shape == (self.TRAIN_BATCH_SIZE, self.train_action_space.shape[1], 1)
        best_next_action_indexes = c.argmax(1).squeeze(1)
        assert best_next_action_indexes.shape == (self.TRAIN_BATCH_SIZE,)
        best_next_actions = self.action_space[best_next_action_indexes]
        assert best_next_actions.shape == (self.TRAIN_BATCH_SIZE, self.ACTION_LENGTH)
        best_next_observation_actions = torch.concatenate((next_observations, best_next_actions), dim=1)
        assert best_next_observation_actions.shape == (self.TRAIN_BATCH_SIZE, self.NN_INPUT)
        # Learn
        target = (immediate_rewards + self.DISCOUNT_FACTOR * (1 - terminations)
                  * self.target_neural_network(best_next_observation_actions))
        prediction = self.neural_network(observation_actions)
        self.optimiser.zero_grad()
        loss = self.loss_function(target, prediction)
        loss.backward()
        self.optimiser.step()
        self.RANDOM_ACTION_PROBABILITY = max(self.RANDOM_ACTION_PROBABILITY * self.RANDOM_ACTION_PROBABILITY_DECAY,
                                             self.MINIMUM_RANDOM_ACTION_PROBABILITY)
        # Update target network
        self.target_neural_network_update_counter = ((self.target_neural_network_update_counter + 1)
                                                     % self.TARGET_NETWORK_UPDATE_TIME)
        if self.target_neural_network_update_counter == 0:
            self.target_neural_network = copy.deepcopy(self.neural_network)
        return float(loss)

    def save(self) -> None:
        if not self.training:
            return
        torch.save(self.neural_network.state_dict(), self.SAVE_PATH)
        print("model saved")
