import os

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
        # Ensure logger_type exists (some rsl_rl versions reference it in save())
        if not hasattr(self, "logger_type"):
            self.logger_type = "tensorboard"

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        # Defensive: also set here in case __init__ was bypassed
        if not hasattr(self, "logger_type"):
            self.logger_type = "tensorboard"
        super().save(path, infos)
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
