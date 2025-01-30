import torch


class Buffer:
    @torch.no_grad()
    def __init__(self, train_agent_count: int, observation_length: int, action_length: int, buffer_size: int) -> None:
        assert train_agent_count >= 1
        assert observation_length >= 1
        assert action_length >= 1
        assert buffer_size >= 1

        self.__train_agent_count = train_agent_count
        self.__observation_length = observation_length
        self.__action_length = action_length
        self.__buffer_size = buffer_size

        self.__observations = torch.zeros((self.__buffer_size, self.__train_agent_count, self.__observation_length))
        self.__actions = torch.zeros((self.__buffer_size, self.__train_agent_count, self.__action_length))
        self.__rewards = torch.zeros(self.__buffer_size, self.__train_agent_count)
        self.__terminations = torch.zeros(self.__buffer_size, self.__train_agent_count)
        self.__sequence_lengths = torch.zeros(self.__buffer_size, self.__train_agent_count, dtype=torch.long)
        self.__next_index = 0
        self.__entry_count = 0

    def ready(self, history_size: int) -> bool:
        return self.__entry_count >= history_size + 2

    @property
    def __full(self) -> bool:
        return self.__entry_count == self.__buffer_size

    @torch.no_grad()
    def push(
            self,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminations: torch.Tensor,
            sequence_lengths: torch.Tensor,
    ) -> None:
        assert observations.shape == (self.__train_agent_count, self.__observation_length)
        assert actions.shape == (self.__train_agent_count, self.__action_length)
        assert rewards.shape == (self.__train_agent_count,)
        assert terminations.shape == (self.__train_agent_count,)
        assert sequence_lengths.shape == (self.__train_agent_count,)
        assert sequence_lengths.dtype == torch.long
        assert torch.all(sequence_lengths >= 0)

        self.__observations[self.__next_index] = observations
        self.__actions[self.__next_index] = actions
        self.__rewards[self.__next_index] = rewards
        self.__terminations[self.__next_index] = terminations
        self.__sequence_lengths[self.__next_index] = sequence_lengths

        self.__next_index = (self.__next_index + 1) % self.__buffer_size
        self.__entry_count = self.__next_index if self.__entry_count < self.__next_index else self.__buffer_size

    @staticmethod
    @torch.no_grad()
    def __history_index(
            tensor: torch.Tensor,
            entry_indexes: torch.Tensor,
            agent_indexes: torch.Tensor,
            history_size: int,
    ) -> torch.Tensor:
        return torch.stack([tensor[entry_indexes - i, agent_indexes] for i in range(history_size - 1, -1, -1)], dim=1)

    @torch.no_grad()
    def random_observations(self, number: int, history_size: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert number > 0
        assert self.ready(history_size=history_size)
        required_history = history_size + 2

        agent_probabilities = torch.ones(size=(self.__train_agent_count,))
        agent_indexes = torch.multinomial(input=agent_probabilities, num_samples=number, replacement=True)
        entry_probabilities = torch.ones(size=(self.__entry_count,))
        invalid_entry_indexes = (torch.arange(
            start=self.__next_index,
            end=self.__next_index + required_history - 1,
        ) % self.__entry_count).unique()
        entry_probabilities[invalid_entry_indexes] = 0
        entry_indexes = torch.multinomial(input=entry_probabilities, num_samples=number, replacement=True)

        full_range_observations = self.__history_index(tensor=self.__observations,
                                                       entry_indexes=entry_indexes,
                                                       agent_indexes=agent_indexes,
                                                       history_size=required_history)
        assert full_range_observations.shape == (number, required_history, self.__observation_length)
        full_range_actions = self.__history_index(tensor=self.__actions,
                                                  entry_indexes=entry_indexes,
                                                  agent_indexes=agent_indexes,
                                                  history_size=required_history)
        assert full_range_actions.shape == (number, required_history, self.__action_length)

        observations = full_range_observations[..., 1:-1, :]
        assert observations.shape == (number, history_size, self.__observation_length)
        actions = full_range_actions[..., 1:-1, :]
        assert actions.shape == (number, history_size, self.__action_length)
        rewards = self.__rewards[entry_indexes - 1, agent_indexes]
        assert rewards.shape == (number,)
        terminations = self.__terminations[entry_indexes - 1, agent_indexes]
        assert terminations.shape == (number,)
        sequence_lengths = self.__sequence_lengths[entry_indexes - 1, agent_indexes]
        assert sequence_lengths.shape == (number,)
        assert sequence_lengths.dtype == torch.long
        previous_observations = full_range_observations[..., :-2, :]
        assert previous_observations.shape == (number, history_size, self.__observation_length)
        assert torch.all(previous_observations[..., 1:, :] == observations[..., :-1, :])
        previous_actions = full_range_actions[..., :-2, :]
        assert previous_actions.shape == (number, history_size, self.__action_length)
        assert torch.all(previous_actions[..., 1:, :] == actions[..., :-1, :])
        previous_sequence_lengths = self.__sequence_lengths[entry_indexes - 2, agent_indexes,]
        assert previous_sequence_lengths.shape == (number,)
        assert previous_sequence_lengths.dtype == torch.long
        next_observations = full_range_observations[..., 2:, :]
        assert next_observations.shape == (number, history_size, self.__observation_length)
        assert torch.all(next_observations[..., :-1, :] == observations[..., 1:, :])
        next_sequence_lengths = self.__sequence_lengths[entry_indexes, agent_indexes]
        assert next_sequence_lengths.shape == (number,)
        assert next_sequence_lengths.dtype == torch.long
        return (
            observations,
            actions,
            rewards,
            terminations,
            next_observations,
            sequence_lengths,
            next_sequence_lengths,
            previous_observations,
            previous_actions,
            previous_sequence_lengths,
        )

    @torch.no_grad()
    def last_actions(self, history_size: int) -> torch.Tensor:
        assert history_size > 0
        start_index = self.__next_index - history_size
        transposed_last_actions = self.__actions[start_index:self.__next_index] if start_index >= 0 \
            else torch.concatenate((self.__actions[start_index:], self.__actions[:self.__next_index]))
        assert transposed_last_actions.shape == (history_size, self.__train_agent_count, self.__action_length)
        last_actions = transposed_last_actions.transpose(dim0=0, dim1=1)
        assert last_actions.shape == (self.__train_agent_count, history_size, self.__action_length)
        return last_actions
