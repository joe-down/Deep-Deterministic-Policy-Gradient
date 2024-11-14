import torch
from agents.actor_critic.actor_critic_base import ActorCriticBase


class SubCritic(ActorCriticBase):
    def __init__(self, load_path: str, observation_length: int, action_length: int, nn_width: int):
        self.__nn_width = nn_width
        super().__init__(load_path=load_path, neural_network=torch.nn.Sequential(
            torch.nn.Linear(observation_length + action_length, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, 1),
        ))
        self.__optimiser = torch.optim.Adam(params=self._neural_network_parameters)

    @property
    def optimiser(self) -> torch.optim.Optimizer:
        return self.__optimiser
