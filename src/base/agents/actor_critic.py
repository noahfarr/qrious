from abc import ABC, abstractmethod
from typing import Any
from base.agents.agent import Agent

class ActorCritic(Agent):

    def __init__(self, env, actor, critic, actor_optimizer, critic_optimizer) -> None:
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

    def sample_action(self, obs: Any):
        """Selects an action based on the given state."""
        return self.policy(obs).sample().item()

    def compute_loss(self):
        raise NotImplementedError


    @abstractmethod
    def compute_actor_loss(self):
        pass

    @abstractmethod
    def compute_critic_loss(self):
        pass

    @abstractmethod
    def train(self, episodes):
        pass
