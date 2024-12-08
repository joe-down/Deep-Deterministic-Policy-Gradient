import pathlib

import torch
import typing
import copy

from actor_critic.actor_critic_base import ActorCriticBase

if typing.TYPE_CHECKING:
    from actor_critic.critic import Critic


class Actor(ActorCriticBase):
    def __init__(self,
                 load_path: pathlib.Path,
                 observation_length: int,
                 action_length: int,
                 nn_width: int,
                 nn_depth: int,
                 ) -> None:
        neural_network = torch.nn.Sequential(
            torch.nn.Linear(observation_length, nn_width),
            torch.nn.ReLU(),
        )
        for _ in range(nn_depth):
            neural_network.append(torch.nn.Linear(nn_width, nn_width))
            neural_network.append(torch.nn.ReLU())
        neural_network.append(torch.nn.Linear(nn_width, action_length))
        neural_network.append(torch.nn.Sigmoid())
        super().__init__(load_path=load_path / "action", neural_network=neural_network, action_length=action_length)
        self.__optimiser = torch.optim.AdamW(params=self._parameters)
        self.__target_neural_network = copy.deepcopy(neural_network)
        self.__update_target_network(target_update_proportion=1)

    def forward_target_network(self, observations: torch.Tensor) -> torch.Tensor:
        return self.__target_neural_network(observations)

    def update(self, observations: torch.Tensor, target_update_proportion: float, critic: "Critic") -> float:
        best_actions = self.forward_network(observations)
        best_observation_actions = torch.concatenate((observations, best_actions), dim=1)
        self.__optimiser.zero_grad()
        loss = (-critic.forward_network(best_observation_actions)).mean()
        loss.backward()
        self.__optimiser.step()
        self.__update_target_network(target_update_proportion=target_update_proportion)
        return -loss

    def __update_target_network(self, target_update_proportion: float) -> None:
        assert 0 <= target_update_proportion <= 1
        for parameter, target_parameter in zip(self._parameters, self.__target_neural_network.parameters()):
            target_parameter.data = ((1 - target_update_proportion) * target_parameter.data
                                     + target_update_proportion * parameter.data)
