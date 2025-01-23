import abc
import copy
import pathlib
import torch
import typing

from actor_critic.model import Model


class ActorCriticBase(abc.ABC):
    def __init__(
            self,
            load_path: pathlib.Path,
            model: Model,
            input_features: int,
            output_features: int,
            history_size: int,
    ) -> None:
        assert input_features > 0
        assert output_features > 0
        self.__model = model
        self.__input_features = input_features
        self.__output_features = output_features
        self.__history_size = history_size
        try:
            self.__model.load_state_dict(torch.load(load_path))
            print("model loaded")
        except FileNotFoundError:
            self.__model.apply(self.__initialise_model)
            print("model initialised")
        self.__target_model = copy.deepcopy(self.__model)
        self.__tgt_mask = torch.triu(-torch.inf * torch.ones(size=(history_size, history_size)), diagonal=1)
        assert self.__tgt_mask.shape == (self.__history_size, self.__history_size)

    @staticmethod
    def __initialise_model(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    @property
    def _model_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__model.parameters()

    @property
    def model_state_dict(self) -> dict[str, typing.Any]:
        return self.__model.state_dict()

    @property
    def _input_features(self) -> int:
        return self.__input_features

    @property
    def _output_features(self) -> int:
        return self.__output_features

    @property
    def _history_size(self) -> int:
        return self.__history_size

    def __forward_model_base(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_sequence_length: torch.IntTensor,
            model: Model,
    ) -> torch.Tensor:
        assert src.shape[-2:] == (self.__history_size, self.__input_features)
        assert tgt.shape == src.shape[:-2] + (self.__history_size - 1, self.__output_features)
        assert src_sequence_length.shape == src.shape[:-2]
        shifted_tgt = torch.concatenate(
            tensors=(tgt, torch.rand(size=src.shape[:-2] + (1, self.__output_features))),
            dim=-2,
        )
        assert shifted_tgt.shape == src.shape[:-2] + (self.__history_size, self.__output_features)
        assert torch.all(shifted_tgt[..., :-1, :] == tgt)
        assert torch.all(shifted_tgt[..., -1, :] >= 0)
        assert torch.all(shifted_tgt[..., -1, :] <= 1)
        src_key_padding_mask = self.__key_padding_mask(sequence_lengths=src_sequence_length)
        assert src_key_padding_mask.shape == src.shape[:-1]
        tgt_key_padding_mask = torch.BoolTensor(torch.concatenate(
            tensors=(src_key_padding_mask[..., :-1], torch.ones(size=src_key_padding_mask.shape[:-1] + (1,)),),
            dim=-1
        ).bool())
        assert tgt_key_padding_mask.shape == shifted_tgt.shape[:-1]
        assert torch.all(tgt_key_padding_mask[..., :-1] == src_key_padding_mask[..., :-1])
        assert torch.all(tgt_key_padding_mask[..., -1] == True)
        result = model.forward(
            src=src,
            tgt=shifted_tgt,
            tgt_mask=self.__tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        assert result.shape == src.shape[:-2] + (self.__history_size, self.__output_features)
        return result

    @abc.abstractmethod
    def forward_model(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_sequence_length: torch.IntTensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(
            src=src,
            tgt=tgt,
            src_sequence_length=src_sequence_length,
            model=self.__model,
        )

    @abc.abstractmethod
    def forward_target_model(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_sequence_length: torch.IntTensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(
            src=src,
            tgt=tgt,
            src_sequence_length=src_sequence_length,
            model=self.__target_model,
        )

    def __key_padding_mask(self, sequence_lengths: torch.IntTensor) -> torch.BoolTensor:
        assert torch.all(sequence_lengths >= 0)
        flat_sequence_lengths = sequence_lengths.flatten()
        assert flat_sequence_lengths.shape == (sequence_lengths.nelement(),)
        capped_flat_sequence_length = flat_sequence_lengths.clamp(min=0, max=self.__history_size)
        assert 0 <= capped_flat_sequence_length <= self.__history_size
        flat_padding_mask = torch.stack(
            [torch.concatenate((torch.ones(self.__history_size - sequence_length).bool(),
                                torch.zeros(sequence_length).bool()))
             for sequence_length in capped_flat_sequence_length]
        )
        assert flat_padding_mask.shape == (sequence_lengths.nelement(), self.__history_size)
        padding_mask = flat_padding_mask.reshape(sequence_lengths.shape + (self.__history_size,))
        assert padding_mask.shape == sequence_lengths.shape + (self.__history_size,)
        return torch.BoolTensor(padding_mask)

    def _update_target_model(self, target_update_proportion: float) -> None:
        assert 0 <= target_update_proportion <= 1
        for parameter, target_parameter in zip(self.__model.parameters(), self.__target_model.parameters()):
            target_parameter.data \
                = ((1 - target_update_proportion) * target_parameter.data + target_update_proportion * parameter.data)
