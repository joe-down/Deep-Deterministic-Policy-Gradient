import pathlib
import torch

from actor_critic.actor import Actor
from actor_critic.actor_critic_base import ActorCriticBase
from actor_critic.model import Model


class SubCritic(ActorCriticBase):
    def __init__(
            self,
            load_path: pathlib.Path,
            observation_length: int,
            action_length: int,
            history_size: int,
            embedding_dim: int,
            n_head: int,
    ) -> None:
        assert observation_length > 0
        assert action_length > 0
        self.__observation_length = observation_length
        self.__action_length = action_length
        super().__init__(
            load_path=load_path,
            model=Model(
                src_features=observation_length + action_length,
                tgt_features=1,
                history_size=history_size,
                embedding_dim=embedding_dim,
                n_head=n_head,
            ),
            input_features=observation_length + action_length,
            output_features=1,
            history_size=history_size,
        )
        self.__optimiser = torch.optim.AdamW(params=self._model_parameters)
        self.__main_network_a = True

    def __forward_model_postprocess(self, qs: torch.Tensor) -> torch.Tensor:
        assert qs.shape[-2:] + (self._history_size, self._output_features,)
        q = qs[..., -1, :]
        assert q.shape == qs.shape[:-2] + qs.shape[-1:]
        return q

    def forward_model(
            self,
            observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_postprocess(qs=self._forward_model_no_tgt(
            src=observation_actions,
            src_sequence_length=observation_actions_sequence_length,
        ))

    def __forward_model_b(
            self,
            observation_actions: torch.Tensor,
            observation_actions_sequence_length: torch.Tensor,
    ) -> torch.Tensor:
        return self.__forward_model_postprocess(qs=self._forward_target_model_no_tgt(
            src=observation_actions,
            src_sequence_length=observation_actions_sequence_length,
        ))

    def __observation_action(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert observation.ndim >= 2
        assert observation.shape[-2:] == (self._history_size, self.__observation_length)
        assert action.shape == observation.shape[:-1] + (self.__action_length,)
        observation_actions = torch.concatenate((observation, action), dim=-1)
        assert observation_actions.shape == observation.shape[:-1] + (self._input_features,)
        assert torch.all(observation_actions[..., :self.__observation_length] == observation)
        assert torch.all(observation_actions[..., self.__observation_length:] == action)
        return observation_actions

    def update(
            self,
            observation: torch.Tensor,
            action: torch.Tensor,
            next_observation: torch.Tensor,
            observation_sequence_length: torch.Tensor,
            next_observation_sequence_length: torch.Tensor,
            immediate_reward: torch.Tensor,
            termination: torch.Tensor,
            discount_factor: float,
            loss_function: torch.nn.Module,
            actor: Actor,
    ) -> float:
        assert immediate_reward.shape == observation.shape[:-2] + (1,)
        assert termination.shape == observation.shape[:-2] + (1,)
        assert 0 <= discount_factor <= 1
        observation_action = self.__observation_action(observation=observation, action=action)
        no_history_next_action = actor.forward_model(
            observations=next_observation,
            previous_actions=action[..., 1:, :],
            observations_sequence_length=next_observation_sequence_length,
        ).unsqueeze(dim=-2)
        assert no_history_next_action.shape == action.shape[:-2] + (1,) + action.shape[-1:]
        next_action = torch.concatenate(tensors=(action[..., 1:, :], no_history_next_action), dim=-2)
        assert next_action.shape == action.shape
        assert torch.all(next_action[..., :-1, :] == action[..., 1:, :])
        assert torch.all(next_action[..., -1:, :] == no_history_next_action)
        next_observation_action = self.__observation_action(observation=next_observation, action=next_action)
        next_qs = self.forward_model(
            observation_actions=next_observation_action,
            observation_actions_sequence_length=next_observation_sequence_length,
        ) if not self.__main_network_a else self.__forward_model_b(
            observation_actions=next_observation_action,
            observation_actions_sequence_length=next_observation_sequence_length,
        )
        q_targets = immediate_reward + discount_factor * (1 - termination) * next_qs
        assert q_targets.shape == observation_action.shape[:-2] + (1,)
        prediction = self.forward_model(
            observation_actions=observation_action,
            observation_actions_sequence_length=observation_sequence_length,
        ) if self.__main_network_a else self.__forward_model_b(
            observation_actions=observation_action,
            observation_actions_sequence_length=observation_sequence_length,
        )
        self.__optimiser.zero_grad()
        loss = loss_function.forward(input=prediction, target=q_targets)
        assert loss.shape == ()
        loss.backward()
        self.__optimiser.step()
        self.__main_network_a = not self.__main_network_a
        return loss
