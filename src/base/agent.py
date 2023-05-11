from abc import ABC, abstractmethod
from typing import Any

class Agent(ABC):

    def __init__(self, env, policy) -> None:
        self.env = env
        self.policy = policy

    def sample_action(self, state: Any) -> Any:
        """Selects an action based on the given state."""
        return self.policy.sample_action(state)

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Updates the agent's policy based on recent experience."""
        pass
