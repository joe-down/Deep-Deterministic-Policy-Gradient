import pathlib

import torch
import typing
from actor_critic.sub_critic import SubCritic

if typing.TYPE_CHECKING:
    from actor_critic.actor import Actor


class Critic:
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
        self.__sub_critics = [SubCritic(load_path=load_path / f"q{i}",
                                        observation_length=observation_length,
                                        action_length=action_length,
                                        nn_width=nn_width,
                                        nn_depth=nn_depth,
                                        )
                              for i in range(2)]
        self.__loss_function = torch.nn.MSELoss()
        self.__observation_actions_length = observation_length + action_length

    @property
    def state_dicts(self) -> list[dict[str, typing.Any]]:
        return [sub_critic.state_dict for sub_critic in self.__sub_critics]

    def __forward_network_base(self, observation_actions: torch.Tensor, q_rewards: torch.Tensor) -> torch.Tensor:
        assert observation_actions.ndim == 2
        assert observation_actions.shape[-1] == self.__observation_actions_length
        least_reward_values, least_reward_indexes = q_rewards.min(dim=-1)
        least_reward_values = least_reward_values.unsqueeze(-1)
        assert least_reward_values.shape == (observation_actions.shape[0], 1)
        return least_reward_values

    def forward_network(self, observation_actions: torch.Tensor) -> torch.Tensor:
        return self.__forward_network_base(
            observation_actions=observation_actions,
            q_rewards=torch.concatenate([sub_critic.forward_network(observations=observation_actions)
                                         for sub_critic in self.__sub_critics],
                                        dim=-1),
        )

    def forward_target_network(self, observation_actions: torch.Tensor) -> torch.Tensor:
        return self.__forward_network_base(
            observation_actions=observation_actions,
            q_rewards=torch.concatenate([sub_critic.forward_target_network(observations=observation_actions)
                                         for sub_critic in self.__sub_critics],
                                        dim=-1),
        )

    def update(self,
               observation_actions: torch.Tensor,
               immediate_rewards: torch.Tensor,
               terminations: torch.Tensor,
               next_observations: torch.Tensor,
               discount_factor: float,
               noise_variance: float,
               actor: "Actor",
               target_update_proportion: float,
               update_target_networks: bool,
               ) -> float:
        noiseless_best_next_actions = actor.forward_target_network(observations=next_observations).detach()
        noise = torch.randn(size=noiseless_best_next_actions.shape) * noise_variance ** 0.5
        noisy_best_next_actions = torch.clamp(input=noiseless_best_next_actions + noise, min=0, max=1)
        noisy_best_next_observation_actions = torch.concatenate((next_observations, noisy_best_next_actions),
                                                                dim=1)
        worst_next_observation_action_qs = self.forward_target_network(noisy_best_next_observation_actions)
        loss = sum(sub_critic.update(
            observation_actions=observation_actions.detach(),
            immediate_rewards=immediate_rewards.detach(),
            terminations=terminations.detach(),
            discount_factor=discount_factor,
            loss_function=self.__loss_function,
            target_update_proportion=target_update_proportion,
            update_target_networks=update_target_networks,
            worst_next_observation_action_qs=worst_next_observation_action_qs.detach(),
        ) for sub_critic in self.__sub_critics)
        return loss

    @staticmethod
    def __noisy_best_next_observation_actions(
            next_observations: torch.Tensor,
            noiseless_best_next_actions: torch.Tensor,
            noise_variance: float,
    ) -> torch.Tensor:
        noise = torch.randn(size=noiseless_best_next_actions.shape) * noise_variance ** 0.5
        noisy_best_next_actions = torch.clamp(input=noiseless_best_next_actions + noise, min=0, max=1)
        assert noisy_best_next_actions.min() >= 0 and noisy_best_next_actions.max() <= 1
        return torch.concatenate((next_observations, noisy_best_next_actions), dim=1)
