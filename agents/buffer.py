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

    @torch.no_grad()
    def push(self,
             observations: torch.Tensor,
             actions: torch.Tensor,
             rewards: torch.Tensor,
             terminations=torch.Tensor,
             ) -> None:
        assert observations.shape == (self.__train_agent_count, self.__observation_length)
        assert actions.shape == (self.__train_agent_count, self.__action_length)
        assert rewards.shape == (self.__train_agent_count,)
        assert terminations.shape == (self.__train_agent_count,)

        self.__observations[self.__next_index] = observations.detach()
        self.__actions[self.__next_index] = actions.detach()
        self.__rewards[self.__next_index] = rewards.detach()
        self.__terminations[self.__next_index] = terminations.detach()

        self.__next_index = (self.__next_index + 1) % self.__buffer_size
        self.__entry_count = self.__next_index if self.__entry_count < self.__next_index else self.__buffer_size

    @torch.no_grad()
    def random_observations(self,
                            number: int,
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.ready
        assert number >= 1

        valid_buffer_indexes = torch.tensor([i for i in range(self.__entry_count)
                                                  if i != (self.__next_index - 1 % self.__buffer_size)])
        buffer_indexes = valid_buffer_indexes[torch.randint(0, len(valid_buffer_indexes), (number,))]
        agent_indexes = torch.randint(0, self.__train_agent_count, (number,))

        observations = self.__observations[buffer_indexes, agent_indexes]
        actions = self.__actions[buffer_indexes, agent_indexes]
        rewards = self.__rewards[buffer_indexes, agent_indexes]
        terminations = self.__terminations[buffer_indexes, agent_indexes]
        next_observations = self.__observations[(buffer_indexes + 1) % self.__buffer_size, agent_indexes]

        assert observations.shape == (number, self.__observation_length)
        assert actions.shape == (number, self.__action_length)
        assert rewards.shape == (number,)
        assert terminations.shape == (number,)
        assert next_observations.shape == (number, self.__observation_length)
        return observations, actions, rewards, terminations, next_observations
