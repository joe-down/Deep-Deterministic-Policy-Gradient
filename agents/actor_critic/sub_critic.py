import pathlib

import torch

from agents.actor_critic.actor import Actor
from agents.actor_critic.actor_critic_base import ActorCriticBase


class SubCritic(ActorCriticBase):
    def __init__(self, load_path: pathlib.Path, observation_length: int, action_length: int, nn_width: int):
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
        self.__optimiser = torch.optim.AdamW(params=self._parameters)

    def update(self,
               observation_actions: torch.Tensor,
               immediate_rewards: torch.Tensor,
               terminations: torch.Tensor,
               next_observations: torch.Tensor,
               discount_factor: float,
               loss_function: torch.nn.MSELoss,
               other_critic: "SubCritic",
               actor: "Actor") -> float:
        best_next_actions = actor.forward_target_network(observations=next_observations).detach()
        best_next_observation_actions = torch.concatenate((next_observations, best_next_actions), dim=1)
        target = (immediate_rewards + discount_factor * (1 - terminations)
                  * other_critic.forward_network(best_next_observation_actions))
        prediction = self.forward_network(observation_actions)
        self.__optimiser.zero_grad()
        loss = loss_function(target, prediction)
        loss.backward()
        self.__optimiser.step()
        return float(loss)
