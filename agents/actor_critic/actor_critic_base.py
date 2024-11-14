import torch
import copy
import typing


class ActorCriticBase:
    def __init__(self, load_path: str, neural_network: torch.nn.Sequential) -> None:
        self.__neural_network: torch.nn.Sequential = neural_network
        try:
            self.__neural_network.load_state_dict(torch.load(load_path))
            print("model loaded")
        except FileNotFoundError:
            self.__neural_network.apply(self.__neural_network_initialisation)
            print("model initialised")
        self.__target_neural_network = copy.deepcopy(self.__neural_network)

    @property
    def _neural_network_parameters(self) -> typing.Iterator[torch.nn.Parameter]:
        return self.__neural_network.parameters()

    @property
    def state_dict(self) -> dict[str, typing.Any]:
        return self.__neural_network.state_dict()

    @staticmethod
    def __neural_network_initialisation(module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def update_target_network(self, target_update_proportion: float) -> None:
        for parameter, target_parameter in zip(self.__neural_network.parameters(),
                                               self.__target_neural_network.parameters()):
            target_parameter.data = ((1 - target_update_proportion) * target_parameter.data
                                     + target_update_proportion * parameter.data)

    def forward_network(self, observations: torch.Tensor) -> torch.Tensor:
        return self.__neural_network(observations)

    def forward_target_network(self, observations: torch.Tensor) -> torch.Tensor:
        return self.__target_neural_network(observations)
