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
        self.reward_termination_index = 0

        self.buffer_start: int = 1
        self.buffer_end: int = 0

    def push_observation(self, observation: torch.tensor) -> None:
        self.observation_index = (self.observation_index + 1) % self.BUFFER_SIZE
        self.observations[self.observation_index] = observation
        self.update_start_end()

    def push_reward(self, reward: float, terminated: bool) -> None:
        self.reward_termination_index = (self.reward_termination_index + 1) % self.BUFFER_SIZE
        self.rewards[self.reward_termination_index] = reward
        self.terminations[self.reward_termination_index] = terminated
        self.update_start_end()

    def update_start_end(self) -> None:
        self.buffer_end = max(self.buffer_end, min(self.observation_index, self.reward_termination_index))
        if self.observation_index == self.reward_termination_index == 0 and self.buffer_end == self.BUFFER_SIZE - 1:
            self.buffer_start = 0

    def buffer_observations_ready(self) -> bool:
        return self.buffer_end >= self.buffer_start + 1

    def random_observations(self, number: int) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        assert self.buffer_observations_ready()
        indexes = torch.randint(self.buffer_start, self.buffer_end + 1, (number,))
        return (self.observations[indexes],
                self.observations[(indexes + 1) % self.BUFFER_SIZE],
                self.rewards[indexes].unsqueeze(1),
                self.terminations[indexes].unsqueeze(1))
