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
        self.__src_embedding = torch.nn.Embedding(num_embeddings=src_features, embedding_dim=embedding_dim)
        self.__tgt_embedding = torch.nn.Embedding(num_embeddings=tgt_features, embedding_dim=embedding_dim)
        self.__transformer = torch.nn.Transformer(d_model=embedding_dim, nhead=n_head, batch_first=True)
        self.__post_transformer = torch.nn.Sequential(
            torch.nn.Linear(in_features=embedding_dim, out_features=tgt_features),
            torch.nn.Sigmoid(),
        )

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            src_key_padding_mask: torch.BoolTensor,
            tgt_key_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        assert src.ndim >= 2
        assert src.shape[-2:] == (self.__history_size, self.__src_features)
        assert tgt.shape == src.shape[:-2] + (self.__history_size, self.__tgt_features,)
        assert tgt_mask.shape == (self.__history_size, self.__history_size)
        assert src_key_padding_mask.shape == src.shape[:-1]
        assert tgt_key_padding_mask.shape == tgt.shape[:-1]
        transformer_out = self.__transformer.forward(
            src=self.__src_embedding.forward(input=src),
            tgt=self.__tgt_embedding.forward(input=tgt),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        assert transformer_out.shape == src.shape[:-2] + (self.__history_size, self.__embedding_dim)
        post_transformer_out = self.__post_transformer(transformer_out)
        assert post_transformer_out.shape == tgt.shape
        return post_transformer_out
