import torch
from agents.actor_critic.actor_critic_base import ActorCriticBase


class SubCritic(ActorCriticBase):
    def __init__(self, load_path: str, observation_length: int, action_length: int, nn_width: int):
        super().__init__(load_path=load_path,
                         neural_network=torch.nn.Sequential(
                             torch.nn.Linear(observation_length + action_length, nn_width),
                             torch.nn.ReLU(),
                             torch.nn.Linear(nn_width, nn_width),
                             torch.nn.ReLU(),
                             torch.nn.Linear(nn_width, nn_width),
                             torch.nn.ReLU(),
                             torch.nn.Linear(nn_width, 1),
                         ))
        self.__optimiser = torch.optim.Adam(params=self._parameters)

    @property
    def optimiser(self) -> torch.optim.Optimizer:
        return self.__optimiser
