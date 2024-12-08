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
        self.__sub_critics = (SubCritic(load_path=load_path / "q1",
                                    observation_length=observation_length,
                                    action_length=action_length,
                                    nn_width=nn_width,
                                    nn_depth=nn_depth,
                                    ),
                          SubCritic(load_path=load_path / "q2",
                                    observation_length=observation_length,
                                    action_length=action_length,
                                    nn_width=nn_width,
                                    nn_depth=nn_depth,
                                    ))
        self.__loss_function = torch.nn.MSELoss()
        self.__q1_main = True
        self.__observation_actions_length = observation_length + action_length

    @property
    def state_dicts(self) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
        return self.__sub_critics[0].state_dict, self.__sub_critics[1].state_dict

    def forward_network(self, observation_actions: torch.Tensor) -> torch.Tensor:
        assert observation_actions.ndim == 2
        assert observation_actions.shape[-1] == self.__observation_actions_length
        q_rewards = torch.concatenate(tuple(sub_critic.forward_network(observations=observation_actions)
                                            for sub_critic in self.__sub_critics),
                                      dim=-1)
        assert q_rewards.shape == (observation_actions.shape[0], 2)
        least_reward_values, least_reward_indexes = q_rewards.min(dim=-1)
        assert least_reward_values.shape == (observation_actions.shape[0],)
        least_reward_values = least_reward_values.unsqueeze(-1)
        assert least_reward_values.shape == (observation_actions.shape[0], 1)
        return least_reward_values

    def update(self,
               observation_actions: torch.Tensor,
               immediate_rewards: torch.Tensor,
               terminations: torch.Tensor,
               next_observations: torch.Tensor,
               discount_factor: float,
               noise_variance: float,
               actor: "Actor",
               ) -> float:
        sub_critic = self.__sub_critics[self.__q1_main]
        loss = sub_critic.update(observation_actions=observation_actions,
                                 immediate_rewards=immediate_rewards,
                                 terminations=terminations,
                                 next_observations=next_observations,
                                 discount_factor=discount_factor,
                                 loss_function=self.__loss_function,
                                 noise_variance=noise_variance,
                                 other_critic=self.__sub_critics[not self.__q1_main],
                                 actor=actor)
        self.__q1_main = not self.__q1_main
        return loss
