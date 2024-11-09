import torch


class Buffer:
    def __init__(self, nn_input: int, buffer_size: int) -> None:
        assert nn_input >= 1
        assert buffer_size >= 1
        self.nn_input_length: int = nn_input
        self.__buffer_size = buffer_size

        self.observations: torch.Tensor = torch.zeros((self.__buffer_size, self.nn_input_length))
        self.next_observation_index: int = 0

        self.rewards: torch.Tensor = torch.zeros(self.__buffer_size, 1)
        self.terminations: torch.Tensor = torch.zeros(self.__buffer_size, 1)
        self.next_reward_termination_index: int = 0

        self.entry_count: int = 0

    def push_observation(self, observation: torch.tensor) -> None:
        assert observation.shape == (self.nn_input_length,)
        assert self.next_observation_index == self.next_reward_termination_index
        self.observations[self.next_observation_index] = observation
        self.next_observation_index = (self.next_observation_index + 1) % self.__buffer_size

    def push_reward(self, reward: float, terminated: bool) -> None:
        assert self.next_reward_termination_index == (self.next_observation_index - 1) % self.__buffer_size
        self.rewards[self.next_reward_termination_index] = reward
        self.terminations[self.next_reward_termination_index] = terminated
        self.next_reward_termination_index = (self.next_reward_termination_index + 1) % self.__buffer_size
        self.update_entry_count()

    def update_entry_count(self) -> None:
        assert self.next_observation_index == self.next_reward_termination_index
        self.entry_count = self.next_observation_index if self.entry_count < self.next_observation_index \
            else self.__buffer_size

    def buffer_observations_ready(self) -> bool:
        return self.entry_count >= 2 and self.next_observation_index == self.next_reward_termination_index

    def filled(self) -> bool:
        return self.entry_count == self.__buffer_size

    def random_observations(self, number: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.buffer_observations_ready()
        valid_indexes = torch.tensor([i for i in range(self.entry_count) if i != self.next_observation_index])
        indexes = valid_indexes[torch.randint(0, self.entry_count - 1, (number,))]
        states, next_states, rewards, terminations = (self.observations[indexes],
                                                      self.observations[(indexes + 1) % self.__buffer_size],
                                                      self.rewards[indexes],
                                                      self.terminations[indexes])
        assert states.shape == next_states.shape == (number, self.nn_input_length)
        assert rewards.shape == terminations.shape == (number, 1)
        return states, next_states, rewards, terminations
