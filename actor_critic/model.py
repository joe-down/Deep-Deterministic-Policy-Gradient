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
        self.__sequential = torch.nn.Sequential(
            torch.nn.Linear(in_features=src_features, out_features=embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            torch.nn.ReLU(),
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
        sequential_out = self.__sequential.forward(src.float())
        assert sequential_out.shape == tgt.shape
        return sequential_out
