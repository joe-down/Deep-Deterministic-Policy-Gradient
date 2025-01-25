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
        self.__sequence_lengths = torch.zeros(self.__buffer_size, self.__train_agent_count, dtype=torch.int)
        self.__next_index = 0
        self.__entry_count = 0

    @property
    def ready(self) -> bool:
        return self.__entry_count >= 2

    @property
    def __incomplete_index(self) -> int:
        return (self.__next_index - 1) % self.__buffer_size

    @torch.no_grad()
    def push(self,
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
        assert sequence_lengths.dtype == torch.int
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
    def __history_index(tensor: torch.Tensor,
                        entry_indexes: torch.Tensor,
                        agent_indexes: torch.Tensor,
                        history: int,
                        ) -> torch.Tensor:
        return torch.stack([tensor[entry_indexes - i, agent_indexes] for i in range(history)], dim=1)

    @torch.no_grad()
    def random_observations(
            self,
            number: int,
            history: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        observations = self.__history_index(tensor=self.__observations,
                                            entry_indexes=entry_indexes,
                                            agent_indexes=agent_indexes,
                                            history=history)
        actions = self.__history_index(tensor=self.__actions,
                                       entry_indexes=entry_indexes,
                                       agent_indexes=agent_indexes,
                                       history=history)
        rewards = self.__rewards[entry_indexes, agent_indexes]
        terminations = self.__terminations[entry_indexes, agent_indexes]
        sequence_lengths = self.__sequence_lengths[entry_indexes, agent_indexes]
        next_sequence_lengths = self.__sequence_lengths[(entry_indexes + 1) % self.__buffer_size, agent_indexes]
        next_observation = self.__history_index(tensor=self.__observations,
                                                entry_indexes=(entry_indexes + 1) % self.__buffer_size,
                                                agent_indexes=agent_indexes,
                                                history=1)

        assert observations.shape == (number, history, self.__observation_length)
        assert actions.shape == (number, history, self.__action_length)
        assert rewards.shape == (number,)
        assert terminations.shape == (number,)
        assert sequence_lengths.shape == (number,)
        assert sequence_lengths.dtype == torch.int
        assert next_sequence_lengths.shape == (number,)
        assert next_sequence_lengths.dtype == torch.int
        assert next_observation.shape == (number, 1, self.__observation_length)
        return observations, actions, rewards, terminations, next_observation, sequence_lengths, next_sequence_lengths

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
