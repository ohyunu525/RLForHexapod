import torch
from isaaclab.envs import ManagerBasedRLEnv


class HexapodEnv(ManagerBasedRLEnv):
    """물리가 터진 step의 NaN 보상을 0으로 바꿈 (그 env는 sim_nan 종료로 리셋됨)."""

    def step(self, action):
        obs, rew, term, trunc, extras = super().step(action)
        return obs, torch.nan_to_num(rew, 0.0, 0.0, 0.0), term, trunc, extras
