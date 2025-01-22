import pathlib
import torch
from actor_critic.actor_critic_base import ActorCriticBase
from actor_critic.transformer import Transformer


class SubCritic(ActorCriticBase):
    __q_features = 1

    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
            n_head: int,
    ) -> None:
        assert observation_length > 0
        assert action_length > 0
        self.__history_size = history_size
        model = Transformer(
            in_features=observation_length + action_length,
            out_features=self.__q_features,
            history_size=history_size,
            n_head=n_head,
        )
        super().__init__(
            load_path=load_path,
            model=model,
            model_input_shape=(history_size, observation_length + action_length),
            model_output_shape=(1,),
        )
        self.__optimiser = torch.optim.AdamW(params=self._model_parameters)

    def update(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_key_padding_mask: torch.Tensor,
            tgt_key_padding_mask: torch.Tensor,
            q_targets: torch.Tensor,
            loss_function: torch.nn.MSELoss,
            target_update_proportion: float,
            update_target_networks: bool,
    ) -> float:
        assert q_targets.shape == src.shape[:-2] + (self.__q_features,)
        assert 0 < target_update_proportion <= 1
        prediction = self.forward_model(
            src=src,
            tgt=tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        self.__optimiser.zero_grad()
        loss = loss_function(q_targets, prediction)
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()
        if update_target_networks:
            self._update_target_model(target_update_proportion=target_update_proportion)
        return loss
