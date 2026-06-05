"""Wrapper that terminates an episode when a safety violation (cost > 0) occurs."""

import gymnasium


class TerminateOnCollisionWrapper(gymnasium.Wrapper):
    """Terminates the episode when the environment reports a nonzero cost.

    Designed to wrap a Safety-Gymnasium env *before* SafetyGymnasium2Gymnasium,
    so ``step()`` still returns the 6-tuple ``(obs, reward, cost, terminated, truncated, info)``.
    """

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        if cost > 0:
            terminated = True
        return obs, reward, cost, terminated, truncated, info
