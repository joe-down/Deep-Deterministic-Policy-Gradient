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
        self.__observation_length = observation_length
        self.__action_length = action_length
        self.__history_size = history_size
        self.__sub_critics = [SubCritic(
            load_path=load_path / f"q{i}",
            observation_length=observation_length,
            action_length=action_length,
            history_size=history_size,
            embedding_dim=embedding_dim,
            n_head=n_head,
        ) for i in range(sub_critic_count)]
        self.__loss_function = torch.nn.MSELoss()

    @property
    def model_state_dicts(self) -> tuple[dict[str, typing.Any], ...]:
        return tuple(sub_critic.model_state_dict for sub_critic in self.__sub_critics)

    @staticmethod
    def __forward_model_base(q_rewards: torch.Tensor) -> torch.Tensor:
        print(q_rewards.shape)
        least_reward_values, _ = q_rewards.min(dim=0)
        assert least_reward_values.shape == q_rewards.shape[1:]
        return least_reward_values

    def forward_model(
            self,
            observation_actions: torch.Tensor,
            previous_qs: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(q_rewards=torch.stack([sub_critic.forward_model(
            observation_actions=observation_actions,
            previous_qs=previous_qs,
            observation_actions_sequence_length=observation_actions_sequence_length,
        ) for sub_critic in self.__sub_critics]))

    def forward_target_model(
            self,
            observation_actions: torch.Tensor,
            previous_qs: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(q_rewards=torch.stack([sub_critic.forward_target_model(
            observation_actions=observation_actions,
            previous_qs=previous_qs,
            observation_actions_sequence_length=observation_actions_sequence_length,
        ) for sub_critic in self.__sub_critics]))

    def update(
            self,
            actor: "Actor",
            observations: torch.Tensor,
            actions: torch.Tensor,
            qs: torch.Tensor,
            observations_sequence_length: torch.Tensor,
            next_observation: torch.Tensor,
            next_observation_sequence_length: torch.Tensor,
            immediate_rewards: torch.Tensor,
            terminations: torch.Tensor,
            discount_factor: float,
            update_target_model: bool,
            target_update_proportion: float,
    ) -> float:
        assert observations.ndim >= 2
        assert observations.shape[-2:] == (self.__history_size, self.__observation_length)
        assert actions.shape == observations.shape[-2:] + (self.__history_size, self.__action_length)
        assert qs.shape == observations.shape[-2:] + (self.__history_size,)
        assert observations_sequence_length.shape == observations.shape[:-2]
        assert next_observation.shape == observations.shape[:-2] + (1, self.__observation_length,)
        assert next_observation_sequence_length.shape == observations_sequence_length.shape
        assert immediate_rewards.shape == observations.shape[:-2]
        assert terminations.shape == observations.shape[:-2]
        observation_actions = torch.concatenate((observations, actions), dim=-1)
        assert (observation_actions.shape
                == observations[:-2] + (self.__history_size, self.__observation_length + self.__action_length))
        assert torch.all(observation_actions[..., :self.__observation_length] == observations)
        assert torch.all(observation_actions[..., self.__observation_length:] == actions)
        next_observations = torch.concatenate((observations[..., 1:, :], next_observation), dim=-2)
        assert next_observations.shape == observations.shape
        assert next_observations[..., :-1, :] == observations[..., 1:, :]
        assert next_observations[..., -1:, :] == next_observation
        best_next_action = actor.forward_target_model(
            observations=next_observations,
            previous_actions=actions[..., 1:, :],
            observations_sequence_length=next_observation_sequence_length,
        )
        assert best_next_action.shape == next_observation.shape[:-2] + (1, self.__action_length,)
        best_next_observation_action = torch.concatenate(tensors=(next_observation, best_next_action), dim=-1)
        assert (best_next_observation_action.shape
                == next_observation.shape[:-2] + (1, self.__observation_length + self.__action_length,))
        assert torch.all(best_next_observation_action[..., :self.__observation_length] == next_observation)
        assert torch.all(best_next_observation_action[..., self.__observation_length:] == best_next_action)
        best_next_observation_actions = torch.concatenate(
            tensors=(observation_actions[..., 1:, :], best_next_observation_action),
            dim=-2,
        )
        assert (best_next_observation_actions.shape
                == observations.shape[:-1] + (self.__history_size, self.__observation_length + self.__action_length))
        assert best_next_observation_actions[..., :-1, :self.__observation_length] == observations
        assert best_next_observation_actions[..., :-1, self.__observation_length:] == actions
        assert best_next_observation_actions[..., -1:, :self.__observation_length] == next_observation
        assert best_next_observation_actions[..., -1:, self.__observation_length:] == best_next_action
        worst_best_next_observation_actions_q = self.forward_target_model(
            observation_actions=best_next_observation_actions,
            previous_qs=qs[..., 1:, :],
            observation_actions_sequence_length=next_observation_sequence_length,
        )
        assert worst_best_next_observation_actions_q.shape== observations.shape[:-2]
        q_targets = (immediate_rewards + discount_factor * (1 - terminations) * worst_best_next_observation_actions_q)
        loss = sum(sub_critic.update(
            observation_actions=observation_actions.detach(),
            previous_qs=qs[:-1],
            observation_actions_sequence_length=observations_sequence_length,
            q_targets=q_targets,
            loss_function=self.__loss_function,
            update_target_model=update_target_model,
            target_update_proportion=target_update_proportion,
        ) for sub_critic in self.__sub_critics)
        return loss
