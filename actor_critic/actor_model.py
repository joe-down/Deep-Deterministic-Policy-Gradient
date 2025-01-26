import typing_extensions
import torch.nn

from actor_critic.model import Model


class ActorModel(Model):
    def __init__(
            self,
            src_features: int,
            tgt_features: int,
            history_size: int,
            embedding_dim: int,
            n_head: int,
    ) -> None:
        super().__init__(
            src_features=src_features,
            tgt_features=tgt_features,
            history_size=history_size,
            embedding_dim=embedding_dim,
            n_head=n_head
        )
        self.__sigmoid = torch.nn.Sigmoid()

    @typing_extensions.override
    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            src_key_padding_mask: torch.Tensor,
            tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.__sigmoid.forward(super().forward(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        ))
