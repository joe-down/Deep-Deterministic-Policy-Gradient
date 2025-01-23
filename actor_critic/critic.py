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
            n_head: int,
            sub_critic_count: int,
    ) -> None:
        self.__observation_length = observation_length
        self.__action_length = action_length
        self.__history_size = history_size
        self.__sub_critics = [SubCritic(
            load_path=load_path / f"q{i}",
            observation_length=observation_length,
            action_length=action_length,
            history_size=history_size,
            n_head=n_head,
        ) for i in range(sub_critic_count)]
        self.__loss_function = torch.nn.MSELoss()

    @property
    def model_state_dicts(self) -> tuple[dict[str, typing.Any], ...]:
        return tuple(sub_critic.model_state_dict for sub_critic in self.__sub_critics)

    @staticmethod
    def __forward_model_base(q_rewards: torch.Tensor) -> torch.Tensor:
        least_reward_values, _ = q_rewards.min(dim=0)
        assert least_reward_values.shape == q_rewards.shape[1:]
        return least_reward_values

    def forward_model(
            self,
            observation_actions: torch.Tensor,
            next_observation_actions: torch.Tensor,
            observation_sequence_length: torch.IntTensor,
            next_observation_sequence_length: torch.IntTensor
    ) -> torch.Tensor:
        return self.__forward_model_base(q_rewards=torch.stack([sub_critic.forward_model(
            src=observation_actions,
            tgt=next_observation_actions,
            src_sequence_length=observation_sequence_length,
            tgt_sequence_length=next_observation_sequence_length,
        ) for sub_critic in self.__sub_critics]))

    def forward_target_model(
            self,
            observation_actions: torch.Tensor,
            next_observation_actions: torch.Tensor,
            observation_sequence_length: torch.IntTensor,
            next_observation_sequence_length: torch.IntTensor
    ) -> torch.Tensor:
        return self.__forward_model_base(q_rewards=torch.stack([sub_critic.forward_target_model(
            src=observation_actions,
            tgt=next_observation_actions,
            src_sequence_length=observation_sequence_length,
            tgt_sequence_length=next_observation_sequence_length,
        ) for sub_critic in self.__sub_critics]))

    def update(
            self,
            observation_actions: torch.Tensor,
            next_observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.IntTensor,
            next_observation_actions_sequence_length: torch.IntTensor,
            immediate_rewards: torch.Tensor,
            terminations: torch.Tensor,
            discount_factor: float,
            actor: "Actor",
            update_target_model: bool,
            target_update_proportion: float,
    ) -> float:
        assert observation_actions.ndim >= 2
        assert immediate_rewards.shape == observation_actions.shape[:-2]
        assert terminations.shape == observation_actions.shape[:-2]
        next_observation = next_observation_actions[..., -1, :self.__observation_length]
        assert next_observation.shape == observation_actions.shape[:-2] + (self.__observation_length,)
        best_next_action = actor.forward_target_model(
            src=next_observation,
            tgt=HELP,
            src_sequence_length=next_observation_actions_sequence_length,
            tgt_sequence_length=HELP,
        )
        assert best_next_action.shape == next_observation.shape[:-1] + (self.__action_length,)
        best_next_observation_action = torch.concatenate(tensors=(next_observation, best_next_action), dim=-1)
        assert (best_next_action.shape
                == next_observation.shape[:-1] + (self.__observation_length + self.__action_length,))
        history_dimension_best_next_observation_action = best_next_observation_action.unsqueeze(dim=-2)
        assert (history_dimension_best_next_observation_action.shape
                == observation_actions.shape[:-2] + (1,) + observation_actions.shape[:-1])
        best_next_observation_actions = torch.concatenate(
            tensors=(observation_actions[..., 1:, :], history_dimension_best_next_observation_action),
            dim=-2,
        )
        assert best_next_observation_actions.shape == observation_actions.shape
        worst_next_observation_action_qs = self.forward_target_model(
            observation_actions=best_next_observation_actions,
            next_observation_actions=HELP,
            observation_sequence_length=observation_actions_sequence_length + 1,
            next_observation_sequence_length=HELP,
        )
        q_targets = (immediate_rewards + discount_factor * (1 - terminations) * worst_next_observation_action_qs)
        loss = sum(sub_critic.update(
            observation_actions=observation_actions.detach(),
            next_observation_actions=next_observation_actions.detach(),
            observation_actions_sequence_length=observation_actions_sequence_length.detach(),
            next_observation_actions_sequence_length=next_observation_actions_sequence_length.detach(),
            q_targets=q_targets.detach(),
            loss_function=self.__loss_function,
            update_target_model=update_target_model,
            target_update_proportion=target_update_proportion,
        ) for sub_critic in self.__sub_critics)
        return loss
