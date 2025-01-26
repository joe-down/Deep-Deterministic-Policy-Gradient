import pathlib
import torch
from actor_critic.actor_critic_base import ActorCriticBase
from actor_critic.model import Model


class SubCritic(ActorCriticBase):
    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
            embedding_dim: int,
            n_head: int,
    ) -> None:
        assert observation_length > 0
        assert action_length > 0
        super().__init__(
            load_path=load_path,
            model=Model(
                src_features=observation_length + action_length,
                tgt_features=1,
                history_size=history_size,
                embedding_dim=embedding_dim,
                n_head=n_head,
            ),
            input_features=observation_length + action_length,
            output_features=1,
            history_size=history_size,
        )
        self.__optimiser = torch.optim.AdamW(params=self._model_parameters)

    def __forward_model_postprocess(self, observation_actions: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        assert qs.shape == observation_actions.shape[:-2] + (self._history_size, self._output_features)
        assert self._output_features == 1
        q = qs[..., -1, :].squeeze(dim=-1)
        assert q.shape == observation_actions.shape[:-2]
        return q

    def forward_model(
            self,
            observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_postprocess(
            observation_actions=observation_actions,
            qs=self._forward_model_no_tgt(
                src=observation_actions,
                src_sequence_length=observation_actions_sequence_length,
            ),
        )

    def forward_target_model(
            self,
            observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_postprocess(
            observation_actions=observation_actions,
            qs=self._forward_target_model_no_tgt(
                src=observation_actions,
                src_sequence_length=observation_actions_sequence_length,
            ),
        )

    def update(
            self,
            observation_actions: torch.Tensor,
            previous_observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
            previous_observation_actions_sequence_length: torch.Tensor,
            q_targets: torch.Tensor,
            loss_function: torch.nn.MSELoss,
            update_target_model: bool,
            target_update_proportion: float,
    ) -> float:
        assert observation_actions.ndim >= 2
        assert q_targets.shape == observation_actions.shape[:-2] + (self._output_features,)
        previous_qs = self._forward_target_model_no_tgt(
            src=previous_observation_actions,
            src_sequence_length=previous_observation_actions_sequence_length,
        )
        assert previous_qs.shape == observation_actions.shape[:-2] + (self._history_size, self._output_features)
        prediction = self._forward_model(
            src=observation_actions,
            tgt=previous_qs[..., 1:, :],
            src_sequence_length=observation_actions_sequence_length,
        )
        assert prediction.shape == observation_actions.shape[:-2] + (self._history_size, self._output_features)
        self.__optimiser.zero_grad()
        loss = (loss_function.forward(input=prediction[..., -1, :], target=q_targets)
                + loss_function.forward(input=prediction[..., :-1, :], target=previous_qs[..., 1:, :]))
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()
        if update_target_model:
            self._update_target_model(target_update_proportion=target_update_proportion)
        return loss.item()
