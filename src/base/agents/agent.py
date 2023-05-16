from abc import ABC, abstractmethod
from typing import Any

class Agent(ABC):

    def __init__(self, env, policy, optimizer) -> None:
        self.env = env
        self.policy = policy
        self.optimizer = optimizer

    def sample_action(self, obs: Any):
        """Selects an action based on the given state."""
        return self.policy(obs).sample().item()

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def train(self, episodes):
        pass
