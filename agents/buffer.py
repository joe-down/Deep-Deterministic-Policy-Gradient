import torch
import typing


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
        self.buffer_fill: int = 0

    def push_observation(self, observation: torch.tensor) -> None:
        self.observation_index = (self.observation_index + 1) % self.BUFFER_SIZE
        self.observations[self.observation_index] = observation
        self.buffer_fill = max(self.buffer_fill, min(self.observation_index, self.reward_index))
        self.update_fill()

    def push_reward(self, reward: float, terminated: bool) -> None:
        self.reward_index = (self.reward_index + 1) % self.BUFFER_SIZE
        self.rewards[self.reward_index] = reward
        self.terminations[self.reward_index] = terminated
        self.update_fill()

    def update_fill(self) -> None:
        self.buffer_fill = max(self.buffer_fill, min(self.observation_index, self.reward_index))

    def random_observations(self, number: int) \
            -> typing.Optional[tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]]:
        assert self.buffer_fill > 0
        indexes = torch.randint(0, self.buffer_fill, (number,))
        return (self.observations[indexes],
                self.observations[indexes + 1],
                self.rewards[indexes].unsqueeze(1),
                self.terminations[indexes].unsqueeze(1))
