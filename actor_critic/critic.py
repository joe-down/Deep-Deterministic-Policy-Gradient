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
            embedding_dim: int,
            n_head: int,
    ) -> None:
        self.__history_size = history_size
        self.__sub_critic_count = sub_critic_count
        self.__sub_critics = [SubCritic(
            load_path=load_path / f"q{i}",
            observation_length=observation_length,
            action_length=action_length,
            history_size=history_size,
            embedding_dim=embedding_dim,
            n_head=n_head,
        ) for i in range(sub_critic_count)]
        self.__loss_function = torch.nn.HuberLoss()

    @property
    def model_state_dicts(self) -> tuple[dict[str, typing.Any], ...]:
        return tuple(sub_critic.model_state_dict for sub_critic in self.__sub_critics)

    def forward_model(
            self,
            observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        q_rewards = torch.concatenate([sub_critic.forward_model(
            observation_actions=observation_actions,
            observation_actions_sequence_length=observation_actions_sequence_length,
        ) for sub_critic in self.__sub_critics], dim=-1)
        assert q_rewards.shape == observation_actions.shape[:-2] + (self.__sub_critic_count,)
        least_reward_values, _ = q_rewards.min(dim=-1)
        assert least_reward_values.shape == q_rewards.shape[:-1]
        return least_reward_values

    def update(
            self,
            actor: "Actor",
            observations: torch.Tensor,
            actions: torch.Tensor,
            observations_sequence_length: torch.Tensor,
            next_observations: torch.Tensor,
            next_observations_sequence_length: torch.Tensor,
            immediate_rewards: torch.Tensor,
            terminations: torch.Tensor,
            discount_factor: float,
    ) -> float:
        loss = torch.tensor(data=[sub_critic.update(
            observation=observations.detach(),
            action=actions.detach(),
            next_observation=next_observations.detach(),
            observation_sequence_length=observations_sequence_length.detach(),
            next_observation_sequence_length=next_observations_sequence_length.detach(),
            immediate_reward=immediate_rewards.unsqueeze(-1).detach(),
            termination=terminations.unsqueeze(-1).detach(),
            discount_factor=discount_factor,
            loss_function=self.__loss_function,
            actor=actor,
        ) for sub_critic in self.__sub_critics]).mean()
        assert loss.shape == ()
        return loss.item()
