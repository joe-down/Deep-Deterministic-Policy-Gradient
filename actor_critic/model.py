import torch.nn


class Model(torch.nn.Module):
    def __init__(
            self,
            src_features: int,
            tgt_features: int,
            history_size: int,
            embedding_dim: int,
            n_head: int,
    ) -> None:
        assert src_features > 0
        assert tgt_features > 0
        assert history_size > 0
        assert embedding_dim > 0
        assert n_head > 0
        assert embedding_dim % n_head == 0
        self.__src_features = src_features
        self.__tgt_features = tgt_features
        self.__history_size = history_size
        self.__embedding_dim = embedding_dim
        super().__init__()
        self.__src_feature_expansion = torch.nn.Linear(in_features=src_features, out_features=embedding_dim)
        self.__tgt_feature_expansion =  torch.nn.Linear(in_features=tgt_features, out_features=embedding_dim)
        self.__transformer = torch.nn.Transformer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        self.__post_transformer = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim, out_features=tgt_features),
        )

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            src_key_padding_mask: torch.Tensor,
            tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert src.ndim >= 2
        assert src.shape[-2:] == (self.__history_size, self.__src_features)
        assert tgt.shape == src.shape[:-2] + (self.__history_size, self.__tgt_features,)
        assert tgt_mask.shape == (self.__history_size, self.__history_size)
        assert src_key_padding_mask.dtype == torch.bool
        assert src_key_padding_mask.shape == src.shape[:-1]
        assert tgt_key_padding_mask.dtype == torch.bool
        assert tgt_key_padding_mask.shape == tgt.shape[:-1]
        expanded_src = self.__src_feature_expansion.forward(input=src.float())
        assert expanded_src.shape == src.shape[:-1] +(self.__embedding_dim,)
        expanded_tgt = self.__tgt_feature_expansion.forward(input=tgt.float())
        assert expanded_tgt.shape == src.shape[:-1] + (self.__embedding_dim,)
        transformer_out = self.__transformer.forward(
            src=expanded_src,
            tgt=expanded_tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        assert transformer_out.shape == src.shape[:-2] + (self.__history_size, self.__embedding_dim)
        post_transformer_out = self.__post_transformer(transformer_out)
        assert post_transformer_out.shape == tgt.shape
        return post_transformer_out
