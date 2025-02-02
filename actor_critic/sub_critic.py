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
                torch.nn.Linear(in_features=(observation_length + action_length) * history_size, out_features=2 ** 6),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 6, out_features=2 ** 6),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 6, out_features=2 ** 6),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 6, out_features=1),
            ),
            input_features=observation_length + action_length,
            output_features=1,
            history_size=history_size,
        )
        self.__optimiser_a = torch.optim.AdamW(params=self._model_a_parameters)
        self.__optimiser_b = torch.optim.AdamW(params=self._model_b_parameters)

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
        result = self._forward_model_a(observation_action.flatten(-2, -1))  # TODO
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
    ) -> float:
        no_history_next_action = actor.forward_target(observation=next_observation).unsqueeze(dim=-2)  # TODO
        assert no_history_next_action.shape == action.shape[:-2] + (1,) + action.shape[-1:]
        next_action = torch.concatenate(tensors=(action[..., 1:, :], no_history_next_action), dim=-2)
        assert next_action.shape == action.shape
        assert torch.all(next_action[..., :-1, :] == action[..., 1:, :])
        assert torch.all(next_action[..., -1:, :] == no_history_next_action)
        next_observation_action = self.__observation_action(observation=next_observation, action=next_action)

        next_qs_a = self._forward_model_a(next_observation_action.flatten(-2, -1).detach()).unsqueeze(-1)  # TODO
        assert next_qs_a.shape == next_observation_action.shape[:-2] + (self._output_features, 1)
        next_qs_b = self._forward_model_b(next_observation_action.flatten(-2, -1).detach()).unsqueeze(-1)  # TODO
        assert next_qs_b.shape == next_observation_action.shape[:-2] + (self._output_features, 1)
        next_qs, minimum_q_indices = torch.concatenate((next_qs_a, next_qs_b), dim=-1).min(dim=-1)
        assert next_qs.shape == next_observation_action.shape[:-2] + (self._output_features,)
        q_targets = immediate_reward + discount_factor * (1 - termination) * next_qs

        observation_action = self.__observation_action(observation=observation, action=action).flatten(-2, -1)  # TODO
        prediction_a = self._forward_model_a(observation_action.detach())
        assert prediction_a.shape == observation.shape[:-2] + (self._output_features,)
        prediction_b = self._forward_model_b(observation_action.detach())
        assert prediction_b.shape == prediction_a.shape

        self.__optimiser_a.zero_grad()
        loss_a = loss_function.forward(input=prediction_a, target=q_targets.detach())
        assert loss_a.shape == ()
        loss_a.backward()
        self.__optimiser_a.step()

        self.__optimiser_b.zero_grad()
        loss_b = loss_function.forward(input=prediction_b, target=q_targets.detach())
        assert loss_b.shape == ()
        loss_b.backward()
        self.__optimiser_b.step()

        return loss_a
