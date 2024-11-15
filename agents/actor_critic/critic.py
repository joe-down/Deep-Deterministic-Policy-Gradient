import torch
import typing
from agents.actor_critic.sub_critic import SubCritic

if typing.TYPE_CHECKING:
    from agents.actor_critic.actor import Actor


class Critic:
    def __init__(self, load_path: str, observation_length: int, action_length: int, nn_width: int) -> None:
        self.__critics: tuple[SubCritic, SubCritic] = (SubCritic(load_path=load_path + "-q1",
                                                                 observation_length=observation_length,
                                                                 action_length=action_length,
                                                                 nn_width=nn_width),
                                                       SubCritic(load_path=load_path + "-q2",
                                                                 observation_length=observation_length,
                                                                 action_length=action_length,
                                                                 nn_width=nn_width))
        self.__loss_function: torch.nn.MSELoss = torch.nn.MSELoss()
        self.__q1_main = True

    @property
    def state_dicts(self) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
        return self.__critics[0].state_dict, self.__critics[1].state_dict

    def forward_network(self, observation_actions: torch.Tensor) -> torch.Tensor:
        return self.__critics[0].forward_network(observations=observation_actions)

    def update(self,
               observation_actions: torch.Tensor,
               immediate_rewards: torch.Tensor,
               terminations: torch.Tensor,
               next_observations: torch.Tensor,
               discount_factor: float,
               actor: "Actor") -> float:
        loss = self.__critics[self.__q1_main].update(observation_actions=observation_actions,
                                                     immediate_rewards=immediate_rewards,
                                                     terminations=terminations,
                                                     next_observations=next_observations,
                                                     discount_factor=discount_factor,
                                                     loss_function=self.__loss_function,
                                                     other_critic=self.__critics[not self.__q1_main],
                                                     actor=actor)
        self.__q1_main = not self.__q1_main
        return float(loss)
