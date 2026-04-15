from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg, DelayedPDActuatorCfg, DelayedPDActuator
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedImplicitActuator(ImplicitActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedImplicitActuatorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # instantiate the delay buffers
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        # all of the envs
        self._ALL_INDICES = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # number of environments (since env_ids can be a slice)
        if env_ids is None or env_ids == slice(None):
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)
        # set a new random delay for environments in env_ids
        time_lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        # set delays
        self.positions_delay_buffer.set_time_lag(time_lags, env_ids)
        self.velocities_delay_buffer.set_time_lag(time_lags, env_ids)
        self.efforts_delay_buffer.set_time_lag(time_lags, env_ids)
        # reset buffers
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        # compte actuator model
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""


class AxiboActuator(DelayedPDActuator):
    """Explicit PD actuator with piecewise-linear torque-speed curve limits.

    Extends :class:`DelayedPDActuator` with a torque-speed envelope derived from
    Robstride motor T-N curves. The curve is defined by four parameters:

    * **Y1**: Peak torque when torque and velocity are in the same direction (driving).
    * **Y2**: Peak torque when torque opposes velocity (braking). Defaults to Y1.
    * **X1**: Speed knee-point (rad/s) below which full Y1 torque is available.
    * **X2**: No-load speed (rad/s) where available driving torque reaches zero.

    For joints driven through a mechanical linkage, ``linkage_ratio`` > 1 multiplies
    torque limits by the ratio and divides speed limits by the ratio.
    """

    cfg: AxiboActuatorCfg

    def __init__(self, cfg: AxiboActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        ratio = cfg.linkage_ratio
        self._Y1 = cfg.Y1 * ratio
        self._Y2 = (cfg.Y2 if cfg.Y2 is not None else cfg.Y1) * ratio
        self._X1 = cfg.X1 / ratio
        self._X2 = cfg.X2 / ratio

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        abs_vel = torch.abs(self._joint_vel)

        # Same-direction envelope: flat at Y1 up to X1, linear drop to 0 at X2
        same_limit = torch.where(
            abs_vel <= self._X1,
            self._Y1,
            self._Y1 * (self._X2 - abs_vel) / (self._X2 - self._X1),
        )
        same_limit = torch.clamp(same_limit, min=0.0)

        # Opposite-direction envelope: flat at Y2 (braking torque)
        opp_limit = self._Y2

        # Combine: when vel >= 0, positive torque is driving, negative is braking
        max_effort = torch.where(self._joint_vel >= 0, same_limit, opp_limit)
        min_effort = torch.where(self._joint_vel >= 0, -opp_limit, -same_limit)

        return torch.clamp(effort, min=min_effort, max=max_effort)


@configclass
class AxiboActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for Axibo torque-speed curve actuator.

    Uses a piecewise-linear T-N curve to clip computed PD torques based on
    instantaneous joint velocity. Suitable for Robstride RS02/RS03/RS04 motors.
    """

    class_type: type = AxiboActuator

    Y1: float = 120.0
    """Peak torque (N·m) when driving (same direction as velocity)."""

    Y2: float | None = None
    """Peak torque (N·m) when braking (opposite direction). Defaults to Y1 if None."""

    X1: float = 10.47
    """Speed knee-point (rad/s) where torque starts dropping."""

    X2: float = 20.94
    """No-load speed (rad/s) where driving torque reaches zero."""

    linkage_ratio: float = 1.0
    """Mechanical linkage multiplier. >1 increases torque limit and decreases speed limit."""
