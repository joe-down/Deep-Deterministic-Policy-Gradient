import torch
import typing
from agents.actor_critic.actor_critic_base import ActorCriticBase

if typing.TYPE_CHECKING:
    from agents.actor_critic.critic import Critic


class Actor(ActorCriticBase):
    def __init__(self, load_path: str, observation_length: int, action_length: int, nn_width: int) -> None:
        self.__nn_width = nn_width
        super().__init__(load_path=load_path + "-action", neural_network=torch.nn.Sequential(
            torch.nn.Linear(observation_length, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, action_length),
            torch.nn.Sigmoid()
        ))
        self.__optimiser: torch.optim.Optimizer = torch.optim.Adam(params=self._neural_network_parameters)

    def update(self, observations: torch.Tensor, critic: "Critic") -> float:
        best_actions = self.forward_network(observations)
        best_observation_actions = torch.concatenate((observations, best_actions), dim=1)
        self.__optimiser.zero_grad()
        loss = (-critic.forward_network(best_observation_actions)).mean()
        loss.backward()
        self.__optimiser.step()
        return float(loss)
