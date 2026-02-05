from dataclasses import MISSING

from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from . import camera_actions, joint_actions


@configclass
class ActionOverridenJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the action overridden delayed joint position action term.

    See :class:`ActionOverridenointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.ActionOverridenJointPositionAction

    asset_cfg: SceneEntityCfg = MISSING
    """Whether to override the action with the delayed action. Defaults to False."""

    override_value: float = 0.0
    """Delay in frames before the action is overridden. Defaults to 0."""


@configclass
class RaycastPitchOffsetActionCfg(ActionTermCfg):
    """Configuration for a pitch action term that steers the ray-cast camera by offset."""

    class_type: type[ActionTerm] = camera_actions.RaycastPitchOffsetAction

    asset_name: str = "robot"
    """The scene entity used for action bookkeeping. Defaults to the robot asset."""

    sensor_name: str = "camera"
    """Sensor name to update. Defaults to "camera"."""

    extra_sensor_names: list[str] = []
    """Additional sensor names to keep in sync."""

    pitch_min: float = 0.0
    """Minimum pitch angle in radians."""

    pitch_max: float = 0.0
    """Maximum pitch angle in radians."""

    init_pitch: float = 0.0
    """Initial pitch angle in radians."""
