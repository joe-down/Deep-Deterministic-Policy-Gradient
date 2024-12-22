import pathlib
import torch
from actor_critic.actor_critic_base import ActorCriticBase


class SubCritic(ActorCriticBase):
    def __init__(self,
                 load_path: pathlib.Path,
                 observation_length: int,
                 action_length: int,
                 nn_width: int,
                 nn_depth: int,
                 ) -> None:
        assert observation_length >= 1
        assert action_length >= 1
        assert nn_width >= 1
        assert nn_depth >= 1
        self.__observation_length = observation_length
        self.__action_length = action_length
        neural_network = torch.nn.Sequential(
            torch.nn.Linear(observation_length + action_length, nn_width),
            torch.nn.ReLU(),
        )
        for _ in range(nn_depth):
            neural_network.append(torch.nn.Linear(nn_width, nn_width))
            neural_network.append(torch.nn.ReLU())
        neural_network.append(torch.nn.Linear(nn_width, 1))
        super().__init__(load_path=load_path, neural_network=neural_network)
        self.__optimiser = torch.optim.AdamW(params=self._parameters)

    @property
    def _nn_output_length(self) -> int:
        return 1

    def update(self,
               observation_actions: torch.Tensor,
               targets: torch.Tensor,
               loss_function: torch.nn.MSELoss,
               target_update_proportion: float,
               update_target_networks: bool,
               ) -> float:
        assert observation_actions.shape[1:] == (self.__observation_length + self.__action_length,)
        assert targets.shape[1:] == (1,)
        assert (observation_actions.shape[0] == targets.shape[0])
        assert 0 < target_update_proportion <= 1
        prediction = self.forward_network(observation_actions)
        self.__optimiser.zero_grad()
        loss = loss_function(targets, prediction)
        loss.backward()
        self.__optimiser.step()
        if update_target_networks:
            self._update_target_network(target_update_proportion=target_update_proportion)
        assert loss.shape == ()
        return loss
