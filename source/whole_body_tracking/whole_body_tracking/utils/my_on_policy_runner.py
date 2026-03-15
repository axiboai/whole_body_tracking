import os
import re
from pathlib import Path

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        # Ensure logger_type exists (some rsl_rl versions reference it in save())
        if not hasattr(self, "logger_type"):
            self.logger_type = "tensorboard"
        super().save(path, infos)
        # logger_type exists in rsl_rl <= 2.3.x but was removed in newer versions
        if getattr(self, "logger_type", None) in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # obs_normalizer exists in rsl_rl <= 2.3.x but was removed in newer versions
            obs_normalizer = getattr(self, "obs_normalizer", None)
            export_policy_as_onnx(self.alg.policy, normalizer=obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.in_progress_export_enabled = False
        self.in_progress_export_dir = None
        self.in_progress_export_base_name = None
        self.in_progress_obs_suffix = "full_obs"
        # Ensure logger_type exists (some rsl_rl versions reference it in save())
        if not hasattr(self, "logger_type"):
            self.logger_type = "tensorboard"

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        # Defensive: also set here in case __init__ was bypassed
        if not hasattr(self, "logger_type"):
            self.logger_type = "tensorboard"
        super().save(path, infos)
        self._maybe_export_in_progress_onnx(path)
        # logger_type exists in rsl_rl <= 2.3.x but was removed in newer versions
        if getattr(self, "logger_type", None) in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # obs_normalizer exists in rsl_rl <= 2.3.x but was removed in newer versions
            obs_normalizer = getattr(self, "obs_normalizer", None)
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run (skip for local files)
            if self.registry_name is not None and not self.registry_name.startswith("local:"):
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def _maybe_export_in_progress_onnx(self, checkpoint_path: str):
        """Export ONNX snapshots for intermediate checkpoints.

        SAFETY: This method must never mutate training state.  The ONNX
        exporter already deep-copies the actor and normalizer, so we do NOT
        call policy.eval() or touch the simulation.  Everything runs under
        torch.no_grad() as an extra safeguard.
        """
        if not bool(getattr(self, "in_progress_export_enabled", False)):
            return

        checkpoint_name = os.path.basename(checkpoint_path)
        match = re.match(r"model_(\d+)\.pt$", checkpoint_name)
        if match is None:
            return

        iteration = int(match.group(1))
        if iteration <= 0:
            return

        export_dir = getattr(self, "in_progress_export_dir", None)
        if not export_dir:
            if self.log_dir is None:
                return
            export_dir = self.log_dir
        os.makedirs(export_dir, exist_ok=True)

        base_name = getattr(self, "in_progress_export_base_name", None)
        if not base_name:
            base_name = Path(self.log_dir).name if self.log_dir else "policy"
        obs_suffix = getattr(self, "in_progress_obs_suffix", "full_obs")
        filename = f"{base_name}_{obs_suffix}_iter{iteration}_in_progress.onnx"

        try:
            import torch

            with torch.no_grad():
                # Build nominal gains from robot config for metadata only.
                robot = self.env.unwrapped.scene["robot"]
                robot_cfg = self.env.unwrapped.scene.cfg.robot
                joint_names = robot.data.joint_names
                nominal_stiffness = [0.0 for _ in joint_names]
                nominal_damping = [0.0 for _ in joint_names]
                for actuator_cfg in robot_cfg.actuators.values():
                    for i, joint_name in enumerate(joint_names):
                        for pattern in actuator_cfg.joint_names_expr:
                            if re.match(pattern, joint_name):
                                if isinstance(actuator_cfg.stiffness, dict):
                                    for k_pattern, k_value in actuator_cfg.stiffness.items():
                                        if re.match(k_pattern, joint_name):
                                            nominal_stiffness[i] = float(k_value)
                                            break
                                else:
                                    nominal_stiffness[i] = float(actuator_cfg.stiffness)

                                if isinstance(actuator_cfg.damping, dict):
                                    for d_pattern, d_value in actuator_cfg.damping.items():
                                        if re.match(d_pattern, joint_name):
                                            nominal_damping[i] = float(d_value)
                                            break
                                else:
                                    nominal_damping[i] = float(actuator_cfg.damping)
                                break

                # The exporter deep-copies the actor and normalizer internally,
                # so we pass the live policy without switching to eval mode.
                obs_normalizer = getattr(self, "obs_normalizer", None)
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    self.alg.policy,
                    export_dir,
                    normalizer=obs_normalizer,
                    filename=filename,
                )
                attach_onnx_metadata(
                    self.env.unwrapped,
                    f"local:{checkpoint_path}",
                    path=export_dir,
                    filename=filename,
                    joint_stiffness_override=nominal_stiffness,
                    joint_damping_override=nominal_damping,
                )
            print(f"[INFO]: ✓ Exported in-progress ONNX: {filename}")
        except Exception as exc:
            print(f"[WARNING]: In-progress ONNX export failed for {checkpoint_name}: {exc}")
