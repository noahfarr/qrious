import gymnasium as gym
class Environment:

    def __init__(self, env) -> None:
        self.env = gym.make(env)
        self.obs_space = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.n

    def step(self, action):
        return self.env.step(action)

    def reset(self, seed=None):
        return self.env.reset(seed=seed) if seed else self.env.reset()