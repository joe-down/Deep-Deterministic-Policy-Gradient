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

    @staticmethod
    def __forward_model_base(q_rewards: torch.Tensor) -> torch.Tensor:
        least_reward_values, _ = q_rewards.min(dim=0)
        assert least_reward_values.shape == q_rewards.shape[1:]
        return least_reward_values

    def forward_model(
            self,
            observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(q_rewards=torch.stack([sub_critic.forward_model(
            observation_actions=observation_actions,
            observation_actions_sequence_length=observation_actions_sequence_length,
        ) for sub_critic in self.__sub_critics]))

    def forward_target_model(
            self,
            observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(q_rewards=torch.stack([sub_critic.forward_target_model(
            observation_actions=observation_actions,
            observation_actions_sequence_length=observation_actions_sequence_length,
        ) for sub_critic in self.__sub_critics]))

    def observation_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert observations.ndim >= 2
        assert observations.shape[-2:] == (self.__history_size, self.__observation_length)
        assert actions.shape == observations.shape[:-2] + (self.__history_size, self.__action_length)
        observation_actions = torch.concatenate((observations, actions), dim=-1)
        assert (observation_actions.shape
                == observations.shape[:-2] + (self.__history_size, self.__observation_length + self.__action_length))
        assert torch.all(observation_actions[..., :self.__observation_length] == observations)
        assert torch.all(observation_actions[..., self.__observation_length:] == actions)
        return observation_actions

    def update(
            self,
            actor: "Actor",
            observations: torch.Tensor,
            actions: torch.Tensor,
            previous_observations: torch.Tensor,
            previous_actions: torch.Tensor,
            observations_sequence_length: torch.Tensor,
            previous_observations_sequence_length: torch.Tensor,
            next_observations: torch.Tensor,
            next_observations_sequence_length: torch.Tensor,
            immediate_rewards: torch.Tensor,
            terminations: torch.Tensor,
            discount_factor: float,
            update_target_model: bool,
            target_update_proportion: float,
    ) -> float:
        assert observations.ndim >= 2
        assert observations.shape[-2:] == (self.__history_size, self.__observation_length)
        assert actions.shape == observations.shape[:-2] + (self.__history_size, self.__action_length)
        assert previous_observations.shape == observations.shape
        assert torch.all(previous_observations[..., 1:, :] == observations[..., :-1, :])
        assert previous_actions.shape == actions.shape
        assert torch.all(previous_actions[..., 1:, :] == actions[..., :-1, :])
        assert observations_sequence_length.shape == observations.shape[:-2]
        assert previous_observations_sequence_length.shape == previous_observations.shape[:-2]
        assert next_observations.shape == observations.shape
        assert torch.all(next_observations[..., :-1, :] == observations[..., 1:, :])
        assert next_observations_sequence_length.shape == observations_sequence_length.shape
        assert immediate_rewards.shape == observations.shape[:-2]
        assert terminations.shape == observations.shape[:-2]
        assert 0 <= discount_factor <= 1
        assert 0 <= target_update_proportion <= 1
        observation_actions = self.observation_actions(observations=observations, actions=actions)
        previous_observation_actions = self.observation_actions(
            observations=previous_observations,
            actions=previous_actions,
        )
        assert torch.all(previous_observation_actions[..., 1:, :] == observation_actions[..., :-1, :])

        squeezed_best_next_action = actor.forward_target_model(
            observations=next_observations,
            previous_actions=actions[..., 1:, :],
            observations_sequence_length=next_observations_sequence_length,
        )
        assert squeezed_best_next_action.shape == next_observations.shape[:-2] + (self.__action_length,)
        best_next_action = squeezed_best_next_action.unsqueeze(dim=-2)
        assert best_next_action.shape == next_observations.shape[:-2] + (1, self.__action_length,)
        best_next_actions = torch.concatenate(tensors=(actions[..., 1:, :], best_next_action), dim=-2)
        assert best_next_actions.shape == actions.shape
        best_next_observation_actions = self.observation_actions(
            observations=next_observations,
            actions=best_next_actions,
        )
        assert best_next_observation_actions.shape == observation_actions.shape
        assert torch.all(best_next_observation_actions[..., :-1, :self.__observation_length]
                         == observations[..., 1:, :])
        assert torch.all(best_next_observation_actions[..., :-1, self.__observation_length:] == actions[..., 1:, :])
        assert torch.all(best_next_observation_actions[..., -1:, :self.__observation_length]
                         == next_observations[..., -1:, :])
        assert torch.all(best_next_observation_actions[..., -1:, self.__observation_length:] == best_next_action)

        worst_best_next_observation_actions_q = self.forward_target_model(
            observation_actions=best_next_observation_actions,
            observation_actions_sequence_length=next_observations_sequence_length,
        )
        assert worst_best_next_observation_actions_q.shape == observations.shape[:-2]
        q_targets = (immediate_rewards
                     + discount_factor * (1 - terminations) * worst_best_next_observation_actions_q).unsqueeze(dim=-1)
        assert q_targets.shape == observations.shape[:-2] + (1,)
        sub_critic_observation_count = observation_actions.size(dim=-3) // self.__sub_critic_count
        assert sub_critic_observation_count > 0
        assert sub_critic_observation_count * self.__sub_critic_count <= observation_actions.size(dim=-3)
        loss = torch.tensor(data=[sub_critic.update(
            observation_actions
            =observation_actions[
             ...,
             sub_critic_number * sub_critic_observation_count:(sub_critic_number + 1) * sub_critic_observation_count,
             :,
             :,
             ].detach(),
            previous_observation_actions
            =previous_observation_actions[
             ...,
             sub_critic_number * sub_critic_observation_count:(sub_critic_number + 1) * sub_critic_observation_count,
             :,
             :,
             ].detach(),
            observation_actions_sequence_length
            =observations_sequence_length[
             ...,
             sub_critic_number * sub_critic_observation_count:(sub_critic_number + 1) * sub_critic_observation_count,
             ].detach(),
            previous_observation_actions_sequence_length
            =previous_observations_sequence_length[
             ...,
             sub_critic_number * sub_critic_observation_count:(sub_critic_number + 1) * sub_critic_observation_count,
             ].detach(),
            q_targets
            =q_targets[
             ...,
             sub_critic_number * sub_critic_observation_count:(sub_critic_number + 1) * sub_critic_observation_count,
             :,
             ].detach(),
            loss_function=self.__loss_function,
            update_target_model=update_target_model,
            target_update_proportion=target_update_proportion,
        ) for sub_critic_number, sub_critic in enumerate(self.__sub_critics)]).mean()
        assert loss.shape == ()
        return loss.item()
