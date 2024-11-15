import numpy
import torch
from agents.actor_critic.actor import Actor


class BasicAgent:
    @staticmethod
    def action(observation: numpy.ndarray, actor: Actor) -> numpy.ndarray:
        return actor.forward_network(observations=torch.tensor(observation)).detach().cpu().numpy()

    @staticmethod
    def reward(reward: float, terminated: bool) -> None:
        return
