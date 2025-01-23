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
            model_input_shape: tuple[int, ...],
            model_output_shape: tuple[int, ...],
    ) -> None:
        for model_input_shape_dimension_length in model_input_shape:
            assert model_input_shape_dimension_length > 0
        for model_output_shape_dimension_length in model_output_shape:
            assert model_output_shape_dimension_length > 0
        self.__model = model
        self.__model_input_shape = model_input_shape
        self.__model_output_shape = model_output_shape
        try:
            self.__model.load_state_dict(torch.load(load_path))
            print("model loaded")
        except FileNotFoundError:
            self.__model.apply(self.__initialise_model)
            print("model initialised")
        self.__target_model = copy.deepcopy(self.__model)

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

    def __forward_model_base(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_sequence_length: torch.IntTensor,
            tgt_sequence_length: torch.IntTensor,
            model: Model,
    ) -> torch.Tensor:
        assert src.shape[-len(self.__model_input_shape):] == self.__model_input_shape
        result = model.forward(
            src=src,
            tgt=tgt,
            tgt_mask=self.__tgt_mask(tgt=tgt),
            src_key_padding_mask=self.__padding_mask(src=src, sequence_lengths=src_sequence_length),
            tgt_key_padding_mask=self.__padding_mask(src=tgt, sequence_lengths=tgt_sequence_length),
        )
        assert result.shape == src.shape[:-len(self.__model_input_shape)] + self.__model_output_shape
        return result

    def forward_model(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_sequence_length: torch.IntTensor,
            tgt_sequence_length: torch.IntTensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(
            src=src,
            tgt=tgt,
            src_sequence_length=src_sequence_length,
            tgt_sequence_length=tgt_sequence_length,
            model=self.__model,
        )

    def forward_target_model(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_sequence_length: torch.IntTensor,
            tgt_sequence_length: torch.IntTensor,
    ) -> torch.Tensor:
        return self.__forward_model_base(
            src=src,
            tgt=tgt,
            src_sequence_length=src_sequence_length,
            tgt_sequence_length=tgt_sequence_length,
            model=self.__target_model,
        )

    @staticmethod
    def __tgt_mask(tgt: torch.Tensor) -> torch.Tensor:
        assert tgt.ndim >= 2
        tgt_mask = torch.triu(-torch.inf * torch.ones(size=(tgt.shape[-2], tgt.shape[-2])), diagonal=1).flip(dims=(1,))
        assert tgt_mask.shape == (tgt.shape[-2], tgt.shape[-2])
        return tgt_mask

    @staticmethod
    def __padding_mask(src: torch.Tensor, sequence_lengths: torch.IntTensor) -> torch.BoolTensor:
        assert src.ndim >= 2
        assert sequence_lengths.shape == src.shape[:-2] + (1,)
        assert torch.all(sequence_lengths >= 0)
        history_length = src.shape[-2]
        flat_sequence_lengths = sequence_lengths.flatten()
        assert flat_sequence_lengths.shape == (sequence_lengths.nelement(),)
        capped_flat_sequence_length = flat_sequence_lengths.clamp(min=0, max=history_length)
        assert 0 <= capped_flat_sequence_length <= history_length
        flat_padding_mask = torch.stack(
            [torch.concatenate((torch.ones(history_length - sequence_length).bool(),
                                torch.zeros(sequence_length).bool()))
             for sequence_length in capped_flat_sequence_length]
        )
        assert flat_padding_mask.shape == (sequence_lengths.nelement(), src.shape[-2])
        padding_mask = flat_padding_mask.reshape(src.shape[:-1])
        assert padding_mask.shape == src.shape[:-1]
        return torch.BoolTensor(padding_mask)

    def _update_target_model(self, target_update_proportion: float) -> None:
        assert 0 <= target_update_proportion <= 1
        for parameter, target_parameter in zip(self.__model.parameters(), self.__target_model.parameters()):
            target_parameter.data \
                = ((1 - target_update_proportion) * target_parameter.data + target_update_proportion * parameter.data)
