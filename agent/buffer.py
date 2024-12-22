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
        self.__rewards: torch.Tensor = torch.zeros(self.__buffer_size, self.__train_agent_count)
        self.__terminations: torch.Tensor = torch.zeros(self.__buffer_size, self.__train_agent_count)
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
             ) -> None:
        assert observations.shape == (self.__train_agent_count, self.__observation_length)
        assert actions.shape == (self.__train_agent_count, self.__action_length)
        assert rewards.shape == (self.__train_agent_count,)
        assert terminations.shape == (self.__train_agent_count,)

        self.__observations[self.__next_index] = observations
        self.__actions[self.__next_index] = actions
        self.__rewards[self.__next_index] = rewards
        self.__terminations[self.__next_index] = terminations

        self.__next_index = (self.__next_index + 1) % self.__buffer_size
        self.__entry_count = self.__next_index if self.__entry_count < self.__next_index else self.__buffer_size

    @torch.no_grad()
    def random_observations(self,
                            number: int,
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        entry_indexes = random_valid_buffer_indexes // self.__train_agent_count
        agent_indexes = random_valid_buffer_indexes // self.__entry_count

        observations = self.__observations[entry_indexes, agent_indexes]
        actions = self.__actions[entry_indexes, agent_indexes]
        rewards = self.__rewards[entry_indexes, agent_indexes]
        terminations = self.__terminations[entry_indexes, agent_indexes]
        next_observations = self.__observations[(entry_indexes + 1) % self.__buffer_size, agent_indexes]

        assert observations.shape == (number, self.__observation_length)
        assert actions.shape == (number, self.__action_length)
        assert rewards.shape == (number,)
        assert terminations.shape == (number,)
        assert next_observations.shape == (number, self.__observation_length)
        return observations, actions, rewards, terminations, next_observations
