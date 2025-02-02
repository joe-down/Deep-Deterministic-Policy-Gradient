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
        self.__main_network_a = True

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
        assert immediate_reward.shape == observation.shape[:-2] + (self._output_features,)
        assert termination.shape == observation.shape[:-2] + (self._output_features,)
        assert 0 <= discount_factor <= 1

        no_history_next_action = actor.forward(observation=next_observation).unsqueeze(dim=-2)#TODO
        assert no_history_next_action.shape == action.shape[:-2] + (1,) + action.shape[-1:]
        next_action = torch.concatenate(tensors=(action[..., 1:, :], no_history_next_action), dim=-2)
        assert next_action.shape == action.shape
        assert torch.all(next_action[..., :-1, :] == action[..., 1:, :])
        assert torch.all(next_action[..., -1:, :] == no_history_next_action)
        next_observation_action \
            = self.__observation_action(observation=next_observation, action=next_action).flatten(-2, -1)  # TODO

        next_qs = self._forward_model_a(next_observation_action) if not self.__main_network_a \
            else self._forward_model_b(next_observation_action)
        assert next_qs.shape == observation.shape[:-2] + (self._output_features,)
        q_targets = immediate_reward + discount_factor * (1 - termination) * next_qs
        assert q_targets.shape == observation.shape[:-2] + (self._output_features,)

        observation_action = self.__observation_action(observation=observation, action=action).flatten(-2, -1)  # TODO
        prediction = self._forward_model_a(observation_action) if self.__main_network_a \
            else self._forward_model_b(observation_action)
        assert prediction.shape == observation.shape[:-2] + (self._output_features,)

        if self.__main_network_a:
            self.__optimiser_a.zero_grad()
        else:
            self.__optimiser_b.zero_grad()
        loss = loss_function.forward(input=prediction, target=q_targets)
        assert loss.shape == ()
        loss.backward()
        if self.__main_network_a:
            self.__optimiser_a.step()
        else:
            self.__optimiser_b.step()

        self.__main_network_a = not self.__main_network_a
        return loss
