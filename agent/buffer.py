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

    @property
    def ready(self) -> bool:
        return self.__entry_count >= 2

    @property
    def __incomplete_index(self) -> int:
        return (self.__next_index - 1) % self.__buffer_size

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
            history: int,
    ) -> torch.Tensor:
        return torch.stack([tensor[entry_indexes - i, agent_indexes] for i in range(history - 1, -1, -1)], dim=1)

    @torch.no_grad()
    def random_observations(self, number: int, history: int) -> tuple[
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
        assert self.ready
        assert number >= 1

        agent_index_starts = (torch.arange(0, self.__train_agent_count) * self.__entry_count).unsqueeze(-1)
        assert agent_index_starts.shape == (self.__train_agent_count, 1)
        assert agent_index_starts[-1] == self.__entry_count * (self.__train_agent_count - 1)
        agent_valid_buffer_indexes = torch.concatenate((
            torch.arange(0, self.__incomplete_index),
            torch.arange(self.__incomplete_index + 1, self.__entry_count),
        )).unsqueeze(0)
        assert agent_valid_buffer_indexes.shape == (1, self.__entry_count - 1)
        assert self.__incomplete_index not in agent_valid_buffer_indexes
        valid_buffer_indexes = (agent_index_starts + agent_valid_buffer_indexes).flatten()
        assert valid_buffer_indexes.shape == (self.__train_agent_count * (self.__entry_count - 1),)
        for i in range(self.__train_agent_count):
            assert i * self.__entry_count + self.__incomplete_index not in valid_buffer_indexes

        random_valid_buffer_indexes = valid_buffer_indexes[torch.randperm(valid_buffer_indexes.size(0))[:number]]
        repeated_random_valid_buffer_indexes = torch.concatenate(
            (random_valid_buffer_indexes.repeat(number // random_valid_buffer_indexes.size(0)),
             random_valid_buffer_indexes[:number % random_valid_buffer_indexes.size(0)]),
        )
        entry_indexes = repeated_random_valid_buffer_indexes // self.__train_agent_count
        agent_indexes = repeated_random_valid_buffer_indexes // self.__entry_count

        full_range_observations = self.__history_index(tensor=self.__observations,
                                                       entry_indexes=entry_indexes,
                                                       agent_indexes=agent_indexes,
                                                       history=history + 2)
        assert full_range_observations.shape == (number, history + 2, self.__observation_length)
        full_range_actions = self.__history_index(tensor=self.__actions,
                                                  entry_indexes=entry_indexes,
                                                  agent_indexes=agent_indexes,
                                                  history=history + 2)
        assert full_range_actions.shape == (number, history + 2, self.__action_length)

        observations = full_range_observations[..., 1:-1, :]
        assert observations.shape == (number, history, self.__observation_length)
        actions = full_range_actions[..., 1:-1, :]
        assert actions.shape == (number, history, self.__action_length)
        rewards = self.__rewards[entry_indexes - 1, agent_indexes]
        assert rewards.shape == (number,)
        terminations = self.__terminations[entry_indexes - 1, agent_indexes]
        assert terminations.shape == (number,)
        sequence_lengths = self.__sequence_lengths[entry_indexes - 1, agent_indexes]
        assert sequence_lengths.shape == (number,)
        assert sequence_lengths.dtype == torch.long
        previous_observations = full_range_observations[..., :-2, :]
        assert previous_observations.shape == (number, history, self.__observation_length)
        assert torch.all(previous_observations[..., 1:, :] == observations[..., :-1, :])
        previous_actions = full_range_actions[..., :-2, :]
        assert previous_actions.shape == (number, history, self.__action_length)
        assert torch.all(previous_actions[..., 1:, :] == actions[..., :-1, :])
        previous_sequence_lengths = self.__sequence_lengths[entry_indexes - 2, agent_indexes,]
        assert previous_sequence_lengths.shape == (number,)
        assert previous_sequence_lengths.dtype == torch.long
        next_observations = full_range_observations[..., 2:, :]
        assert next_observations.shape == (number, history, self.__observation_length)
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
