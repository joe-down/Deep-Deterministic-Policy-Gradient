import pathlib
import torch
import typing
from actor_critic.actor_critic_base import ActorCriticBase

if typing.TYPE_CHECKING:
    from actor_critic.critic import Critic


class Actor(ActorCriticBase):
    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
    ) -> None:
        super().__init__(
            load_path=load_path / "action",
            model=torch.nn.Sequential(
                torch.nn.Linear(in_features=observation_length * history_size, out_features=2 ** 6),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 6, out_features=2 ** 6),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 6, out_features=2 ** 6),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=2 ** 6, out_features=action_length),
                torch.nn.Sigmoid(),
            ),
            input_features=observation_length,
            output_features=action_length,
            history_size=history_size,
        )
        self.__optimiser = torch.optim.AdamW(params=self._model_a_parameters)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        assert observation.ndim >= 2
        assert observation.shape[-2:] == (self._history_size, self._input_features)
        result = self._forward_model_a(observation.flatten(-2, -1))  # TODO
        assert result.shape == observation.shape[:-2] + (self._output_features,)
        return result

    def update(
            self,
            observations: torch.Tensor,
            previous_actions: torch.Tensor,
            target_model_update_proportion: float,
            update_target_network: bool,
            critic: "Critic",
    ) -> float:
        assert previous_actions.shape == observations.shape[:-2] + (self._history_size - 1, self._output_features,)
        assert 0 <= target_model_update_proportion <= 1

        unprocessed_best_actions = self.forward(observation=observations).unsqueeze(-2)  # TODO
        assert unprocessed_best_actions.shape == observations.shape[:-2] + (1, self._output_features,)
        best_actions = torch.concatenate(tensors=(previous_actions, unprocessed_best_actions), dim=-2)
        assert best_actions.shape == observations.shape[:-2] + (self._history_size, self._output_features,)
        assert torch.all(best_actions[..., :-1, :] == previous_actions)
        assert torch.all(best_actions[..., -1:, :] == unprocessed_best_actions)

        self.__optimiser.zero_grad()
        loss = (-critic.forward_model(observation=observations, action=best_actions)).mean()
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()

        if update_target_network:
            self._update_target_model(target_update_proportion=target_model_update_proportion)
        return -loss

    def _update_target_model(self, target_update_proportion: float) -> None:
        assert 0 <= target_update_proportion <= 1
        for parameter, target_parameter in zip(self._model_a_parameters, self._model_b_parameters):
            target_parameter.data = ((1 - target_update_proportion) * target_parameter.data
                                     + target_update_proportion * parameter.data)
