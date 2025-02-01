import pathlib
import torch
import typing
from actor_critic.actor_critic_base import ActorCriticBase
from actor_critic.actor_model import ActorModel

if typing.TYPE_CHECKING:
    from actor_critic.critic import Critic


class Actor(ActorCriticBase):
    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
            embedding_dim: int,
            n_head: int,
    ) -> None:
        super().__init__(
            load_path=load_path / "action",
            model=ActorModel(
                src_features=observation_length,
                tgt_features=action_length,
                history_size=history_size,
                embedding_dim=embedding_dim,
                n_head=n_head,
            ),
            input_features=observation_length,
            output_features=action_length,
            history_size=history_size,
        )
        self.__optimiser = torch.optim.AdamW(params=self._model_parameters)
        self.__history_loss_function = torch.nn.HuberLoss()

    def __forward_model_postprocess(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        assert actions.shape == observations.shape[:-2] + (self._history_size, self._output_features)
        action = actions[..., -1, :]
        assert action.shape == observations.shape[:-2] + (self._output_features,)
        return action

    def forward_model(
            self,
            observations: torch.Tensor,
            previous_actions: torch.Tensor,
            observations_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_postprocess(
            observations=observations,
            actions=self._forward_model(
                src=observations,
                tgt=previous_actions,
                src_sequence_length=observations_sequence_length,
            ),
        )

    def forward_target_model(
            self,
            observations: torch.Tensor,
            previous_actions: torch.Tensor,
            observations_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_postprocess(
            observations=observations,
            actions=self._forward_target_model(
                src=observations,
                tgt=previous_actions,
                src_sequence_length=observations_sequence_length,
            ),
        )

    def update(
            self,
            observations: torch.Tensor,
            previous_actions: torch.Tensor,
            observations_sequence_length: torch.Tensor,
            target_model_update_proportion: float,
            update_target_network: bool,
            critic: "Critic",
    ) -> float:
        unprocessed_best_actions = self._forward_model(
            src=observations.detach(),
            tgt=previous_actions.detach(),
            src_sequence_length=observations_sequence_length.detach()
        )
        assert unprocessed_best_actions.shape == observations.shape[:-2] + (self._history_size, self._output_features,)
        best_actions = torch.concatenate(
            tensors=(previous_actions.detach(), unprocessed_best_actions[..., -1:, :]),
            dim=-2,
        )
        assert best_actions.shape == unprocessed_best_actions.shape
        assert torch.all(best_actions[..., :-1, :] == previous_actions)
        assert torch.all(best_actions[..., -1:, :] == unprocessed_best_actions[..., -1:, :])
        best_observation_actions = torch.concatenate((observations.detach(), best_actions), dim=-1)
        assert (best_observation_actions.shape
                == observations.shape[:-2] + (self._history_size, self._input_features + self._output_features,))
        self.__optimiser.zero_grad()
        q_loss = (-critic.forward_model(
            observation_actions=best_observation_actions,
            observation_actions_sequence_length=observations_sequence_length.detach(),
        )).mean()
        assert q_loss.shape == ()
        history_loss = self.__history_loss_function.forward(
            input=unprocessed_best_actions[..., :-1, :],
            target=previous_actions.detach(),
        )
        assert history_loss.shape == ()
        loss = q_loss + history_loss
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()
        if update_target_network:
            self._update_target_model(target_update_proportion=target_model_update_proportion)
        return -loss
