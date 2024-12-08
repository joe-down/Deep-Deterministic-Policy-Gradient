import pathlib

import torch

from actor_critic.actor import Actor
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
        super().__init__(load_path=load_path, neural_network=neural_network, action_length=self.__action_length)
        self.__optimiser = torch.optim.AdamW(params=self._parameters)

    def update(self,
               observation_actions: torch.Tensor,
               immediate_rewards: torch.Tensor,
               terminations: torch.Tensor,
               next_observations: torch.Tensor,
               discount_factor: float,
               loss_function: torch.nn.MSELoss,
               noise_variance: float,
               other_critic: "SubCritic",
               actor: "Actor",
               ) -> float:
        assert observation_actions.shape[1:] == (self.__observation_length + self.__action_length,)
        assert immediate_rewards.shape[1:] == (1,)
        assert terminations.shape[1:] == (1,)
        assert next_observations.shape[1:] == (self.__observation_length,)
        assert (observation_actions.shape[0]
                == immediate_rewards.shape[0]
                == terminations.shape[0]
                == next_observations.shape[0])
        assert 0 <= discount_factor <= 1
        assert noise_variance >= 0
        noiseless_best_next_actions = actor.forward_target_network(observations=next_observations)
        noise = torch.randn(size=noiseless_best_next_actions.shape) * noise_variance ** 0.5
        noisy_best_next_actions = (noiseless_best_next_actions + noise)
        best_next_observation_actions = torch.concatenate((next_observations, noisy_best_next_actions), dim=1)
        target = (immediate_rewards + discount_factor * (1 - terminations)
                  * other_critic.forward_network(best_next_observation_actions))
        prediction = self.forward_network(observation_actions)
        self.__optimiser.zero_grad()
        loss = loss_function(target, prediction)
        loss.backward()
        self.__optimiser.step()
        assert loss.shape == ()
        return loss
