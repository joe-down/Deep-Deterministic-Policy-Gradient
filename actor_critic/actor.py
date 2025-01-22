import pathlib
import torch
import typing
from actor_critic.actor_critic_base import ActorCriticBase
from actor_critic.transformer import Transformer

if typing.TYPE_CHECKING:
    from actor_critic.critic import Critic


class Actor(ActorCriticBase):
    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
            n_head: int,
    ) -> None:
        self.__observation_length = observation_length
        self.__action_length = action_length
        self.__history_size = history_size
        model = Transformer(
            in_features=observation_length,
            out_features=action_length,
            history_size=history_size,
            n_head=n_head,
        )
        super().__init__(
            load_path=load_path / "action",
            model=model,
            model_input_shape=(history_size, observation_length),
            model_output_shape=(action_length,),
        )
        self.__optimiser = torch.optim.AdamW(params=self._model_parameters)

    def update(
            self,
            observations: torch.Tensor,
            tgt: torch.Tensor,
            observations_key_padding_mask: torch.Tensor,
            tgt_key_padding_mask: torch.Tensor,
            target_model_update_proportion: float,
            update_target_network: bool,
            critic: "Critic",
    ) -> float:
        best_actions = self.forward_model(
            src=observations,
            tgt=tgt,
            src_key_padding_mask=observations_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        assert best_actions.shape == observations.shape[:-2] + (self.__action_length,)
        history_dimension_best_actions = best_actions.unsqueeze(dim=-2)
        assert history_dimension_best_actions.shape == observations.shape[:-2] + (1, self.__action_length)
        history_repeated_best_actions = history_dimension_best_actions.repeat_interleave(
            repeats=self.__history_size,
            dim=-2,
        )
        assert history_repeated_best_actions.shape == observations.shape[:-1] + (self.__action_length,)
        best_observation_actions = torch.concatenate((observations, history_repeated_best_actions), dim=-1)
        assert best_observation_actions.shape == observations.shape[:-1] + (
        self.__observation_length + self.__action_length,)
        self.__optimiser.zero_grad()
        loss = (-critic.forward_network()).mean()  # TODO critic forward
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()
        if update_target_network:
            self._update_target_model(target_update_proportion=target_model_update_proportion)
        return -loss
