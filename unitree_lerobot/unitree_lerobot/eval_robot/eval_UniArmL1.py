"""
Evaluation script for UniArmL1 robot arm on real hardware.

This script loads a trained policy and runs it on the UniArmL1 robot arm.
UniArmL1 is a 6-DOF robot arm (5 arm joints + 1 gripper) controlled via serial port.

Usage:
python unitree_lerobot/eval_robot/eval_UniArmL1.py \
    --repo_id test_uniarm \
    --policy.path=/home/unitree/unitree_IL_lerobot/unitree_lerobot/lerobot/outputs/train/2026-04-27/18-41-19_act/checkpoints/last/pretrained_model \
    --port /dev/ttyACM0 \
    --frequency 30 \
    --cameras '{"head": {"device": 2}, "wrist": {"device": 0}}'
Camera configuration via --cameras (JSON string):
- device: camera index (e.g., 0, 1, 2)
- fps: frames per second
- width: image width in pixels
- height: image height in pixels
- fourcc: video format ('MJPG', 'YUYV', etc.)
- warmup_s: warmup time in seconds
- flip_vertical: boolean to flip image vertically

Refer to:
    lerobot/lerobot/scripts/eval.py
    unitree_lerobot/eval_robot/eval_g1.py

"""

import time
import torch
import logging
import argparse
import numpy as np
import sys
import os
import cv2
from pprint import pformat
from dataclasses import asdict, dataclass, field
from torch import nn
from contextlib import nullcontext
from typing import Any
from pathlib import Path

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.configs.policies import PreTrainedConfig

from unitree_lerobot.eval_robot.robot_control.uniarml1.config_uniarm_l1 import TeleopConfig
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import (
    to_list,
    to_scalar,
    predict_action,
)

import logging_mp

logger_mp = logging_mp.getLogger(__name__)



@dataclass
class EvalUniArmConfig(TeleopConfig):
    """Configuration for UniArm evaluation.

    Inherits all hardware/camera/frequency fields from TeleopConfig.
    Only adds lerobot-specific fields (repo_id, policy).
    """
    repo_id: str = ""
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0

    rename_map: dict[str, str] = field(default_factory=dict)

    visualization: bool = True  # Add missing visualization field
    frequency: int = 30  # Control loop frequency in Hz
    use_dataset: bool = True

    def __post_init__(self):
        # Parse policy path from CLI if provided
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

def setup_uniarm_camera(cfg: EvalUniArmConfig):
    """Setup cameras for UniArm using LocalCamera with async read."""
    from unitree_lerobot.eval_robot.image_server.image_client import LocalCamera

    cameras_info = {}
    # cameras is now a dict[str, dict] format
    for name, cam_cfg in cfg.cameras.items():
        if isinstance(cam_cfg, int):
            cam_cfg = {"device": cam_cfg}

        camera = LocalCamera(
            device=f"/dev/video{cam_cfg.get('device', cam_cfg.get('idx', 0))}",
            fps=cam_cfg.get('fps', 30),
            width=cam_cfg.get('width', 640),
            height=cam_cfg.get('height', 480),
            fourcc=cam_cfg.get('fourcc', 'MJPG'),
            warmup_s=cam_cfg.get('warmup_s', 1.0),
        )
        camera.connect()
        logger_mp.info(f"Camera {name} (/dev/video{cam_cfg['device']}) connected: {camera.width}x{camera.height}")

        cameras_info[name] = {
            "camera": camera,
            "img_shape": (cam_cfg.get('height', 480), cam_cfg.get('width', 640), 3),
            "flip_vertical": cam_cfg.get('flip_vertical', False),
        }

    return cameras_info


def setup_uniarm_robot(cfg: EvalUniArmConfig):
    """Setup UniArmL1 robot controller."""
    from unitree_lerobot.eval_robot.robot_control.uniarml1.uniarm_l1 import UniArmL1
    from unitree_lerobot.eval_robot.robot_control.uniarml1.config_uniarm_l1 import UniArmL1Config

    urdf_path = Path(__file__).resolve().parent / "assets/uniarml1/urdf/UniArmL1.urdf"
    
    if not urdf_path.exists():
        logger_mp.error(f"URDF file not found: {urdf_path}")
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    logger_mp.info(f"Using URDF path: {urdf_path}")

    arm_config = UniArmL1Config(
        id='follower',
        port=cfg.port,
        urdf_path=str(urdf_path),
        init_move_duration=2.0,
    )

    # Create UniArmL1 controller
    arm = UniArmL1(arm_config)

    return {
        "arm": arm,
        "arm_dof": 6,  # 5 arm joints + 1 gripper
    }

def process_uniarm_observation(arm, cameras_info: dict[str, dict] | None):
    """Process images and get current arm state using LocalCamera.async_read()."""
    observation = {}
    camera_name_map = {"head": "cam_top", "wrist": "cam_wrist"}
    if cameras_info is not None:
        for name, info in cameras_info.items():
            camera = info.get("camera")
            if camera is not None:
                try:
                    frame_rgb = camera.async_read()
                    target_shape = info["img_shape"]
                    if frame_rgb.shape[0] != target_shape[0] or frame_rgb.shape[1] != target_shape[1]:
                        frame_rgb = cv2.resize(frame_rgb, (target_shape[1], target_shape[0]))
                    frame_rgb = cv2.flip(frame_rgb, 0)
                    if info.get("flip_vertical"):
                        frame_rgb = cv2.flip(frame_rgb, 0)

                    mapped_name = camera_name_map.get(name, name)
                    mapped_name = camera_name_map.get(name, name)

                    frame_rgb = np.ascontiguousarray(frame_rgb)
                    image_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()  # HWC -> CHW

                    observation[f"observation.images.{mapped_name}"] = image_tensor
                except (TimeoutError, RuntimeError) as e:
                    logger_mp.warning(f"Camera {name} read failed: {e}")

    # Get current arm state in sim space
    current_arm_q = None
    if arm is not None:
        current_arm_q = arm.get_cur_q_sim()

    return observation, current_arm_q


def wait_for_start_signal() -> None:
    """Wait for user to press 's' to start (supports single-key without Enter on TTY)."""
    try:
        if sys.stdin.isatty():
            import select
            import termios
            import tty

            print("Press 's' to initialize and start...", end=" ", flush=True)
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while True:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch.lower() == "s":
                            print("")
                            return
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        else:
            while True:
                user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
                logger_mp.info(f"user_input: {user_input}")
                if user_input.strip().lower().startswith("s"):
                    return
                logger_mp.info("Input not recognized, please enter 's' to start.")
    except Exception:
        while True:
            user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
            logger_mp.info(f"user_input: {user_input}")
            if user_input.strip().lower().startswith("s"):
                return
            logger_mp.info("Input not recognized, please enter 's' to start.")


def eval_uniarm_policy(
    cfg: EvalUniArmConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    """Main evaluation loop for UniArmL1."""
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    cameras_info = None
    try:
        # --- Setup Phase ---
        cameras_info = setup_uniarm_camera(cfg)
        robot_interface = setup_uniarm_robot(cfg)

        arm = robot_interface["arm"]
        arm_dof = robot_interface["arm_dof"]

        # Get initial pose from dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        logger_mp.info(f"Initial arm pose from dataset: {init_arm_pose}")

        wait_for_start_signal()

        idx = 0

        if True:
            # Start control loop
            arm.start_control_loop()

            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")

            # --- Main Loop ---
            while True:
                loop_start_time = time.perf_counter()

                # Ensure position control mode during evaluation
                if getattr(arm, "bus", None) is not None and arm.bus.control_mode != "position_control":
                    arm.bus.control_mode = "position_control"

                # 1. Get Observations
                observation, current_arm_q = process_uniarm_observation(arm, cameras_info)

                # Build state tensor
                state_tensor = torch.from_numpy(current_arm_q).float()
                observation["observation.state"] = state_tensor

                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                action_np = action.cpu().numpy()

                if idx % 30 == 0:
                    for img_key in observation:
                        if img_key.startswith("observation.images."):
                            img = observation[img_key]
                            logger_mp.info(
                                "Image stats - %s: min=%d max=%d mean=%.1f",
                                img_key, int(img.min()), int(img.max()), float(img.float().mean()),
                            )
                    logger_mp.info(
                        "Action stats: min=%.4f max=%.4f mean=%.4f",
                        float(action_np.min()),
                        float(action_np.max()),
                        float(action_np.mean()),
                    )

                # 3. Execute Action
                # Convert action to sim space joint angles
                arm_action = action_np[:arm_dof]

                # Map sim action to motor positions
                tgt_pos_rad = arm.map_sim_to_output_rad_all(arm_action)

                # Update target position
                arm.tgt_pos_rad = tgt_pos_rad
                arm.con_mode[:] = 1  # Enable position control


                # Visualization
                if cfg.visualization:
                    visualization_data(idx, observation, current_arm_q, action_np, rerun_logger)

                idx += 1

                # Maintain frequency
                elapsed = time.perf_counter() - loop_start_time
                time.sleep(max(0, (1.0 / cfg.frequency) - elapsed))

    except KeyboardInterrupt:
        logger_mp.info("Evaluation interrupted by user.")
    except Exception as e:
        logger_mp.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if cameras_info is not None:
            for name, info in cameras_info.items():
                camera = info.get("camera")
                if camera is not None:
                    camera.disconnect()
                    logger_mp.info(f"Camera {name} disconnected.")

        if 'robot_interface' in locals():
            arm = robot_interface["arm"]
            logger_mp.info("Stopping UniArmL1 control loop...")
            arm.stop_control_loop()


@parser.wrap()
def eval_main(cfg: EvalUniArmConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id='repo_pick_place_2')

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_uniarm_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
