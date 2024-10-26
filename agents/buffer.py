import torch


class Buffer:
    def __init__(self, nn_input: int) -> None:
        self.BUFFER_SIZE: int = 2 ** 3
        assert self.BUFFER_SIZE > 0

        assert nn_input >= 1
        self.nn_input_length: int = nn_input

        self.observations: torch.Tensor = torch.zeros((self.BUFFER_SIZE, self.nn_input_length))
        self.observation_index: int = 0

        self.rewards: torch.Tensor = torch.zeros(self.BUFFER_SIZE)
        self.terminations: torch.Tensor = torch.zeros(self.BUFFER_SIZE)
        self.reward_termination_index: int = 0

        self.buffer_end: int = 0

    def push_observation(self, observation: torch.tensor) -> None:
        assert observation.shape == (self.nn_input_length,)
        assert self.observation_index == self.reward_termination_index
        self.observations[self.observation_index] = observation
        self.observation_index = (self.observation_index + 1) % self.BUFFER_SIZE

    def push_reward(self, reward: float, terminated: bool) -> None:
        assert self.reward_termination_index == (self.observation_index - 1) % self.BUFFER_SIZE
        self.rewards[self.reward_termination_index] = reward
        self.terminations[self.reward_termination_index] = terminated
        self.reward_termination_index = (self.reward_termination_index + 1) % self.BUFFER_SIZE
        self.update_end()

    def update_end(self) -> None:
        assert self.observation_index == self.reward_termination_index
        self.buffer_end = max(self.buffer_end, self.observation_index)

    def buffer_observations_ready(self) -> bool:
        return self.buffer_end >= 2

    def filled(self) -> bool:
        return self.buffer_end == self.BUFFER_SIZE - 1

    def random_observations(self, number: int) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        assert self.buffer_observations_ready()
        indexes = torch.randint(0, self.buffer_end, (number,))
        states, next_states, rewards, terminations = (self.observations[indexes],
                                                      self.observations[(indexes + 1) % self.BUFFER_SIZE],
                                                      self.rewards[indexes].unsqueeze(1),
                                                      self.terminations[indexes].unsqueeze(1))
        assert states.shape == next_states.shape == (number, self.nn_input_length)
        assert rewards.shape == terminations.shape == (number,)
        return states, next_states, rewards, terminations
