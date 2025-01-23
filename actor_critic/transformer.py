import torch.nn


class Transformer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, history_size: int, n_head: int) -> None:
        assert in_features > 0
        assert out_features > 0
        assert history_size > 0
        assert n_head > 0
        assert in_features % n_head == 0
        self.__in_features = in_features
        self.__out_features = out_features
        self.__history_size = history_size
        super().__init__()
        self.__transformer = torch.nn.Transformer(d_model=in_features, nhead=n_head, batch_first=True, )
        self.__post_transformer = torch.nn.Sequential(
            torch.nn.Linear(in_features=history_size * in_features, out_features=out_features),
            torch.nn.Sigmoid(),
        )

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                tgt_mask: torch.Tensor,
                src_key_padding_mask: torch.BoolTensor,
                tgt_key_padding_mask: torch.BoolTensor,
                ) -> torch.Tensor:
        assert src.ndim >= 2
        assert src.shape[-2:] == (self.__history_size, self.__in_features)
        assert tgt.shape == src.shape
        assert tgt_mask.shape == (tgt.shape[-2], tgt.shape[-2])
        assert src_key_padding_mask.shape == src.shape[:-1]
        assert tgt_key_padding_mask.shape == tgt.shape[:-1]
        transformer_out = self.__transformer.forward(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        assert transformer_out.shape == src.shape[:-2] + (self.__history_size, self.__in_features)
        flat_transformer_out = transformer_out.flatten(start_dim=1, end_dim=-1)
        assert flat_transformer_out.shape == src.shape[:-2] + (self.__history_size * self.__in_features,)
        post_transformer_out = self.__post_transformer(flat_transformer_out)
        assert post_transformer_out.shape == src.shape[:-2] + (self.__out_features,)
        return post_transformer_out
