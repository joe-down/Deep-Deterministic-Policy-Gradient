import copy
import pathlib
import abc

import torch
import typing


class ActorCriticBase:
    def __init__(self, load_path: pathlib.Path, neural_network: torch.nn.Sequential) -> None:
        self.__neural_network: torch.nn.Sequential = neural_network
        try:
            self.__neural_network.load_state_dict(torch.load(load_path))
            print("model loaded")
        except FileNotFoundError:
            self.__neural_network.apply(self.__neural_network_initialisation)
            print("model initialised")
        self.__target_neural_network = copy.deepcopy(neural_network)
        self._update_target_network(target_update_proportion=1)

    @property
    def _parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__neural_network.parameters()

    @property
    def state_dict(self) -> dict[str, typing.Any]:
        return self.__neural_network.state_dict()

    @property
    @abc.abstractmethod
    def _nn_output_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def __neural_network_initialisation(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def __forward_network_base(self, observations: torch.Tensor, network: torch.nn.Sequential) -> torch.Tensor:
        assert observations.ndim == 3
        result = network(observations)
        assert result.shape[0] == observations.shape[0]
        assert result.shape[1:] == self._nn_output_shape
        return result

    def forward_network(self, observations: torch.Tensor) -> torch.Tensor:
        return self.__forward_network_base(observations=observations, network=self.__neural_network)

    def forward_target_network(self, observations: torch.Tensor) -> torch.Tensor:
        return self.__forward_network_base(observations=observations, network=self.__target_neural_network)

    def _update_target_network(self, target_update_proportion: float) -> None:
        assert 0 <= target_update_proportion <= 1
        for parameter, target_parameter in zip(self._parameters, self.__target_neural_network.parameters()):
            target_parameter.data = ((1 - target_update_proportion) * target_parameter.data
                                     + target_update_proportion * parameter.data)
