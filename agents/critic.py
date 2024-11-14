import torch
import typing
from agents.actor_critic_base import ActorCriticBase
if typing.TYPE_CHECKING:
    from agents.actor import Actor


class Critic(ActorCriticBase):
    def __init__(self, load_path: str, observation_length: int, action_length: int, nn_width: int) -> None:
        self.__nn_width = nn_width
        super().__init__(load_path=load_path + "-q", observation_length=observation_length, action_length=action_length)
        self.__optimiser: torch.optim.Optimizer = torch.optim.Adam(params=self._neural_network.parameters())
        self.__loss_function: torch.nn.MSELoss = torch.nn.MSELoss()

    def _build_neural_network(self) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(self._observation_length + self._action_length, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, self.__nn_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__nn_width, 1),
        )

    def update(self,
               observation_actions: torch.Tensor,
               immediate_rewards: torch.Tensor,
               terminations: torch.Tensor,
               next_observations: torch.Tensor,
               discount_factor: float,
               actor: "Actor") -> float:
        best_next_actions = actor.forward_target_network(observations=next_observations).detach()
        best_next_observation_actions = torch.concatenate((next_observations, best_next_actions), dim=1)
        target = (immediate_rewards + discount_factor * (1 - terminations)
                  * self.forward_target_network(best_next_observation_actions))
        prediction = self.forward_network(observation_actions)
        self.__optimiser.zero_grad()
        loss = self.__loss_function(target, prediction)
        loss.backward()
        self.__optimiser.step()
        return float(loss)
