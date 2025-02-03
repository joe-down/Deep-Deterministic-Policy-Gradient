import pathlib
import torch

from actor_critic.actor import Actor
from actor_critic.actor_critic_base import ActorCriticBase


class SubCritic(ActorCriticBase):
    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
    ) -> None:
        assert observation_length > 0
        assert action_length > 0
        self.__observation_length = observation_length
        self.__action_length = action_length
        super().__init__(
            load_path=load_path,
            model=torch.nn.Sequential(
                torch.nn.Linear(in_features=(observation_length + action_length) * history_size, out_features=2 ** 8),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 8, out_features=2 ** 8),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 8, out_features=2 ** 8),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 8, out_features=2 ** 8),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 8, out_features=2 ** 8),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 8, out_features=1),
            ),
            input_features=observation_length + action_length,
            output_features=1,
            history_size=history_size,
        )
        self.__optimiser = torch.optim.AdamW(params=self._model_parameters)

    def __observation_action(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert observation.ndim >= 2
        assert observation.shape[-2:] == (self._history_size, self.__observation_length)
        assert action.shape == observation.shape[:-2] + (self._history_size, self.__action_length,)
        observation_actions = torch.concatenate((observation, action), dim=-1)
        assert observation_actions.shape == observation.shape[:-2] + (self._history_size, self._input_features,)
        assert torch.all(observation_actions[..., :self.__observation_length] == observation)
        assert torch.all(observation_actions[..., self.__observation_length:] == action)
        return observation_actions

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        observation_action = self.__observation_action(observation=observation, action=action)
        result = self._forward_model(observation_action.flatten(-2, -1))  # TODO
        assert result.shape == observation.shape[:-2] + (self._output_features,)
        return result

    def forward_target(self, observation: torch.Tensor, action: torch.Tensor):
        observation_action = self.__observation_action(observation=observation, action=action)
        result = self._forward_target_model(observation_action.flatten(-2, -1))  # TODO
        assert result.shape == observation.shape[:-2] + (self._output_features,)
        return result

    def update(
            self,
            observation: torch.Tensor,
            action: torch.Tensor,
            next_observation: torch.Tensor,
            immediate_reward: torch.Tensor,
            termination: torch.Tensor,
            discount_factor: float,
            loss_function: torch.nn.Module,
            actor: Actor,
            update_target_network: bool,
            target_model_update_proportion: float,
    ) -> float:
        no_history_next_action = actor.forward_target(observation=next_observation).unsqueeze(dim=-2)  # TODO
        assert no_history_next_action.shape == action.shape[:-2] + (1,) + action.shape[-1:]
        next_action = torch.concatenate(tensors=(action[..., 1:, :], no_history_next_action), dim=-2)
        assert next_action.shape == action.shape
        assert torch.all(next_action[..., :-1, :] == action[..., 1:, :])
        assert torch.all(next_action[..., -1:, :] == no_history_next_action)

        next_qs = self.forward_target(observation=next_observation, action=next_action).unsqueeze(-1)  # TODO
        assert next_qs.shape ==next_observation.shape[:-2] + (self._output_features, 1)
        q_targets = immediate_reward + discount_factor * (1 - termination) * next_qs

        prediction = self.forward(observation=observation, action=action)
        assert prediction.shape == observation.shape[:-2] + (self._output_features,)

        self.__optimiser.zero_grad()
        loss = loss_function.forward(input=prediction, target=q_targets.detach())
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()
        if update_target_network:
            self._update_target_model(target_update_proportion=target_model_update_proportion)

        return loss