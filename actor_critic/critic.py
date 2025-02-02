import pathlib
import torch
import typing
from actor_critic.sub_critic import SubCritic

if typing.TYPE_CHECKING:
    from actor_critic.actor import Actor


class Critic:
    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
            sub_critic_count: int,
    ) -> None:
        self.__sub_critic_count = sub_critic_count
        self.__sub_critics = [SubCritic(
            load_path=load_path / f"q{i}",
            observation_length=observation_length,
            action_length=action_length,
            history_size=history_size,
        ) for i in range(sub_critic_count)]
        self.__loss_function = torch.nn.HuberLoss()

    @property
    def model_state_dicts(self) -> tuple[dict[str, typing.Any], ...]:
        return tuple(sub_critic.model_a_state_dict for sub_critic in self.__sub_critics)

    def forward_model(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_rewards = torch.stack([sub_critic.forward(observation=observation, action=action).mean(dim=-1)  # TODO
                                 for sub_critic in self.__sub_critics])
        assert q_rewards.shape == (self.__sub_critic_count,) + observation.shape[:-2]
        least_reward_values, _ = q_rewards.min(dim=0)
        assert least_reward_values.shape == observation.shape[:-2]
        return least_reward_values

    def update(
            self,
            actor: "Actor",
            observations: torch.Tensor,
            actions: torch.Tensor,
            next_observations: torch.Tensor,
            immediate_rewards: torch.Tensor,
            terminations: torch.Tensor,
            discount_factor: float,
    ) -> float:
        loss = torch.tensor(data=[sub_critic.update(
            observation=observations.detach(),
            action=actions.detach(),
            next_observation=next_observations.detach(),
            immediate_reward=immediate_rewards.unsqueeze(-1).detach(),
            termination=terminations.unsqueeze(-1).detach(),
            discount_factor=discount_factor,
            loss_function=self.__loss_function,
            actor=actor,
        ) for sub_critic in self.__sub_critics]).mean()
        assert loss.shape == ()
        return loss.item()
