from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import action_cfg


class RaycastPitchOffsetAction(ActionTerm):
    """Action term that steers the ray-cast camera by pitch only, relative to the original offset."""

    cfg: action_cfg.RaycastPitchOffsetActionCfg

    def __init__(self, cfg: action_cfg.RaycastPitchOffsetActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._sensor = env.scene.sensors[self.cfg.sensor_name]
        self._extra_sensors = [env.scene.sensors[name] for name in self.cfg.extra_sensor_names]
        self._raw_actions = torch.zeros((env.num_envs, 1), device=env.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._base_offset_quat = self._sensor._offset_quat.clone()
        self._extra_base_offset_quat = [sensor._offset_quat.clone() for sensor in self._extra_sensors]
        base_roll, base_pitch, base_yaw = math_utils.euler_xyz_from_quat(self._base_offset_quat)
        self._base_roll = base_roll
        self._base_pitch = base_pitch
        self._base_yaw = base_yaw

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self.cfg.init_pitch
        self._base_offset_quat[env_ids] = self._sensor._offset_quat[env_ids].clone()
        base_roll, base_pitch, base_yaw = math_utils.euler_xyz_from_quat(self._base_offset_quat[env_ids])
        self._base_roll[env_ids] = base_roll
        self._base_pitch[env_ids] = base_pitch
        self._base_yaw[env_ids] = base_yaw
        for idx, sensor in enumerate(self._extra_sensors):
            self._extra_base_offset_quat[idx][env_ids] = sensor._offset_quat[env_ids].clone()
        self._apply_pitch(env_ids)

    def process_actions(self, action: torch.Tensor):
        self._raw_actions[:] = action
        action = torch.clamp(action.squeeze(-1), -1.0, 1.0)
        pitch = self.cfg.pitch_min + (action + 1.0) * 0.5 * (self.cfg.pitch_max - self.cfg.pitch_min)
        self._processed_actions[:] = pitch.unsqueeze(-1)

    def apply_actions(self):
        self._apply_pitch(slice(None))

    def _apply_pitch(self, env_ids):
        pitch = self._processed_actions[env_ids].squeeze(-1)
        delta_pitch = pitch - self._base_pitch[env_ids]
        delta_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(delta_pitch),
            delta_pitch,
            torch.zeros_like(delta_pitch),
        )
        quat = math_utils.quat_mul(self._base_offset_quat[env_ids], delta_quat)
        self._sensor._offset_quat[env_ids] = quat
        pos_w, quat_w = self._sensor._compute_camera_world_poses(env_ids)
        self._sensor._data.pos_w[env_ids] = pos_w
        self._sensor._data.quat_w_world[env_ids] = quat_w
        for idx, sensor in enumerate(self._extra_sensors):
            extra_quat = math_utils.quat_mul(self._extra_base_offset_quat[idx][env_ids], delta_quat)
            sensor._offset_quat[env_ids] = extra_quat
            pos_w, quat_w = sensor._compute_camera_world_poses(env_ids)
            sensor._data.pos_w[env_ids] = pos_w
            sensor._data.quat_w_world[env_ids] = quat_w
