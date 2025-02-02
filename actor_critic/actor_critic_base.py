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
        self.__model_a = model
        try:
            self.__model_a.load_state_dict(torch.load(load_path))
            print("model loaded")
        except FileNotFoundError:
            self.__model_a.apply(self.__initialise_model)
            print("model initialised")
        self.__model_b = copy.deepcopy(self.__model_a)

    @staticmethod
    def __initialise_model(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    @property
    def _model_a_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__model_a.parameters()

    @property
    def _model_b_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__model_b.parameters()

    @property
    def model_a_state_dict(self) -> dict[str, typing.Any]:
        return self.__model_a.state_dict()

    @property
    def _input_features(self) -> int:
        return self.__input_features

    @property
    def _output_features(self) -> int:
        return self.__output_features

    @property
    def _history_size(self) -> int:
        return self.__history_size

    def _forward_model_a(self, *args, **kwargs) -> torch.Tensor:
        return self.__model_a.forward(*args, **kwargs)

    def _forward_model_b(self, *args, **kwargs) -> torch.Tensor:
        return self.__model_b.forward(*args, **kwargs)
