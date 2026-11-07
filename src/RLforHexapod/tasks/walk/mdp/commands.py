"""몸통 높이 명령: 몸통 원점이 지면에서 얼마나 높아야 하는지 (N, 1) [m]."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import index_fill_
from isaaclab.utils.configclass import configclass

__all__ = ["BaseHeightCommand", "BaseHeightCommandCfg", "base_height"]


def base_height(env, asset: Articulation) -> torch.Tensor:
    """몸통 원점의 지면 기준 높이 (N,). 지금은 평지라 env 원점 기준 z다. 지형이 생기면 몸통 아래 지면 높이를 뺀다."""
    return asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


class BaseHeightCommand(CommandTerm):
    """몸통 높이를 범위 안에서 고르게 뽑는다. resampling_time_range마다 새로 뽑는다."""

    cfg: BaseHeightCommandCfg

    def __init__(self, cfg: BaseHeightCommandCfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        # 에피소드 평균 높이 오차 [m]. reset에서 확정해 Metrics/<명령 이름>/error_height로 기록된다
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self._error_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/height"
        self.cfg.element_names = self.cfg.element_names or ["base_height"]

    def __str__(self) -> str:
        return (f"BaseHeightCommand:\n\tHeight range: {self.cfg.height_range}\n"
                f"\tResampling time range: {self.cfg.resampling_time_range}")

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        self._error_sum += torch.abs(self.height_command[:, 0] - base_height(self._env, self.robot))
        self._step_count += 1.0

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        self.metrics["error_height"][env_ids] = self._error_sum[env_ids] / self._step_count[env_ids].clamp_min(1.0)
        extras = super().reset(env_ids)
        index_fill_(self._error_sum, env_ids, 0.0)
        index_fill_(self._step_count, env_ids, 0.0)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        num = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        self.height_command[env_ids, 0] = torch.empty(num, device=self.device).uniform_(*self.cfg.height_range)

    def _update_command(self):
        pass


@configclass
class BaseHeightCommandCfg(CommandTermCfg):
    class_type: type = BaseHeightCommand
    asset_name: str = MISSING
    height_range: tuple[float, float] = MISSING  # 몸통 원점 높이 [m]
