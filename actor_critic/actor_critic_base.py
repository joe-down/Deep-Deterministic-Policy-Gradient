import abc
import copy
import pathlib
import torch
import typing


class ActorCriticBase(abc.ABC):
    def __init__(
            self,
            load_path: pathlib.Path,
            model: torch.nn.Module,
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
            self.__model.apply(self.__model_initialisation)
            print("model initialised")
        self.__target_model = copy.deepcopy(self.__model)
        self._update_target_model(target_update_proportion=1)

    @staticmethod
    def __model_initialisation(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    @property
    def _model_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__model.parameters()

    @property
    def model_state_dict(self) -> dict[str, typing.Any]:
        return self.__model.state_dict()

    @abc.abstractmethod
    def _src_key_padding_mask(self, src: torch.Tensor, step_number: int) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _tgt_key_padding_mask(self, tgt: torch.Tensor, step_number: int) -> torch.Tensor:
        raise NotImplementedError

    def __forward_model_base(self,
                             observations: torch.Tensor,
                             tgt: torch.Tensor,
                             step_number: int,
                             model: torch.nn.Module
                             ) -> torch.Tensor:
        assert observations.shape[-len(self.__model_input_shape):] == self.__model_input_shape
        assert tgt.shape == observations.shape
        result = model.forward(
            src=observations,
            tgt=tgt,
            src_key_padding_mask=self._src_key_padding_mask(src=observations, step_number=step_number),
            tgt_key_padding_mask=self._tgt_key_padding_mask(tgt=observations, step_number=step_number),
        )
        assert result.shape == observations.shape[:-len(self.__model_input_shape)] + self.__model_output_shape
        return result

    def forward_model(self, observations: torch.Tensor, tgt: torch.Tensor, step_number: int) -> torch.Tensor:
        return self.__forward_model_base(
            observations=observations,
            tgt=tgt,
            model=self.__model,
            step_number=step_number,
        )

    def forward_target_model(self, observations: torch.Tensor, tgt: torch.Tensor, step_number: int) -> torch.Tensor:
        return self.__forward_model_base(
            observations=observations,
            tgt=tgt,
            model=self.__target_model,
            step_number=step_number,
        )

    def _update_target_model(self, target_update_proportion: float) -> None:
        assert 0 <= target_update_proportion <= 1
        for parameter, target_parameter in zip(self.__model.parameters(), self.__target_model.parameters()):
            target_parameter.data = ((1 - target_update_proportion) * target_parameter.data
                                     + target_update_proportion * parameter.data)
