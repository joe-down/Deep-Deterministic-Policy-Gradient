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
            input_features: int,
            output_features: int,
            history_size: int,
    ) -> None:
        assert input_features > 0
        assert output_features > 0
        assert history_size > 0
        self.__input_features = input_features
        self.__output_features = output_features
        self.__history_size = history_size
        self.__model = model
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
    def _target_model_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__target_model.parameters()

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

    def _forward_model(self, *args, **kwargs) -> torch.Tensor:
        return self.__model.forward(*args, **kwargs)

    def _forward_target_model(self, *args, **kwargs) -> torch.Tensor:
        return self.__target_model.forward(*args, **kwargs)

    def _update_target_model(self, target_update_proportion: float) -> None:
        assert 0 <= target_update_proportion <= 1
        for parameter, target_parameter in zip(self._model_parameters, self._target_model_parameters):
            target_parameter.data = ((1 - target_update_proportion) * target_parameter.data
                                     + target_update_proportion * parameter.data)
