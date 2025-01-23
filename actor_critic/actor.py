import pathlib
import torch
import typing
import typing_extensions
from actor_critic.actor_critic_base import ActorCriticBase
from actor_critic.model import Model

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
            model=Model(
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
        self.__transformer_loss_function = torch.nn.MSELoss()

    @typing_extensions.override
    def forward_model(
            self,
            observations: torch.Tensor,
            previous_actions: torch.Tensor,
            observations_sequence_length: torch.IntTensor,
    ) -> torch.Tensor:
        return super().forward_model(
            src=observations,
            tgt=previous_actions,
            src_sequence_length=observations_sequence_length,
        )

    @typing_extensions.override
    def forward_target_model(
            self,
            observations: torch.Tensor,
            previous_actions: torch.Tensor,
            observations_sequence_length: torch.IntTensor,
    ) -> torch.Tensor:
        return super().forward_target_model(
            src=observations,
            tgt=previous_actions,
            src_sequence_length=observations_sequence_length,
        )

    def update(
            self,
            observations: torch.Tensor,
            previous_actions: torch.Tensor,
            previous_qs: torch.Tensor,
            observations_sequence_length: torch.IntTensor,
            target_model_update_proportion: float,
            update_target_network: bool,
            critic: "Critic",
    ) -> float:
        best_actions = self.forward_model(
            observations=observations,
            previous_actions=previous_actions,
            observations_sequence_length=observations_sequence_length
        )
        assert best_actions.shape == observations.shape[:-2] + (self._history_size, self._output_features,)
        best_observation_actions = torch.concatenate((observations, best_actions), dim=-1)
        assert (best_observation_actions.shape
                == observations.shape[:-2] + (self._history_size, self._input_features + self._output_features,))
        self.__optimiser.zero_grad()
        loss = (-critic.forward_model(
            observation_actions=best_observation_actions,
            previous_qs=previous_qs,
            observation_actions_sequence_length=observations_sequence_length,
        )).mean() + self.__transformer_loss_function.forward(input=best_actions[:-1], target=previous_actions)
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()
        if update_target_network:
            self._update_target_model(target_update_proportion=target_model_update_proportion)
        return -loss.item()
