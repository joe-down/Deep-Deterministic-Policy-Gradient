import pathlib

import torch
import typing


class ActorCriticBase:
    def __init__(self, load_path: pathlib.Path, neural_network: torch.nn.Sequential, action_length: int) -> None:
        assert action_length >= 1
        self.__neural_network: torch.nn.Sequential = neural_network
        try:
            self.__neural_network.load_state_dict(torch.load(load_path))
            print("model loaded")
        except FileNotFoundError:
            self.__neural_network.apply(self.__neural_network_initialisation)
            print("model initialised")
        self.__action_length = action_length

    @property
    def _parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__neural_network.parameters()

    @property
    def state_dict(self) -> dict[str, typing.Any]:
        return self.__neural_network.state_dict()

    @staticmethod
    def __neural_network_initialisation(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward_network(self, observations: torch.Tensor) -> torch.Tensor:
        actions = self.__neural_network(observations)
        assert actions.shape[-1] == self.__action_length
        return actions
