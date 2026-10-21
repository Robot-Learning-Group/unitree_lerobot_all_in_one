"""Config-driven G1 arm joint policy; unselected joints hold the episode pose."""
import json
from pathlib import Path
import time

import numpy as np

from unitree_lerobot.utils.constants import (
    ROBOT_CONFIGS, G1_DEX1_CONFIG, G1_DEX3_CONFIG, G1_BRAINCO_CONFIG, G1_INSPIRE_CONFIG,
)


class ArmJointLayout:
    def __init__(self, cfg):
        if cfg.robot_type not in ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot_type: {cfg.robot_type}")
        if cfg.arm not in ("G1_29", "G1_23"):
            raise ValueError("Arm-joint inference requires a G1 arm")
        if cfg.use_dataset or cfg.visualization:
            raise ValueError("Configured live inference requires use_dataset=false and visualization=false")
        if not np.isfinite(cfg.frequency) or cfg.frequency <= 0:
            raise ValueError("frequency must be positive and finite")
        if not cfg.initial_pose_episode:
            raise ValueError("initial_pose_episode must specify a Unitree episode directory")
        self.config = ROBOT_CONFIGS[cfg.robot_type]
        parts = {"left_arm.qpos": list(range(7)), "right_arm.qpos": list(range(7, 14))}
        hand_configs = {"dex1": G1_DEX1_CONFIG, "dex3": G1_DEX3_CONFIG,
                        "brainco": G1_BRAINCO_CONFIG, "inspire1": G1_INSPIRE_CONFIG}
        fields = self.config.json_state_data_name + self.config.json_action_data_name
        has_hands = any(key in ("left_ee.qpos", "right_ee.qpos") for key in fields)
        self.ee = cfg.ee.lower() if has_hands else ""
        if has_hands and self.ee not in hand_configs:
            raise ValueError("Hand channels require ee=dex1, dex3, brainco or inspire1")
        names = (hand_configs[self.ee].motors if has_hands else G1_DEX1_CONFIG.motors[:14])
        self.ee_dof = (len(names) - 14) // 2 if has_hands else 0
        if has_hands:
            parts.update({"left_ee.qpos": list(range(14, 14 + self.ee_dof)),
                          "right_ee.qpos": list(range(14 + self.ee_dof, len(names)))})
        def indices(fields):
            if not fields or any(key not in parts for key in fields) or len(set(fields)) != len(fields):
                raise ValueError("Configured inference supports arm and hand qpos channels only")
            return [i for key in fields for i in parts[key]]
        self.state_indices = indices(self.config.json_state_data_name)
        self.action_indices = indices(self.config.json_action_data_name)
        if any([names[i] for i in order] != self.config.motors
               for order in (self.state_indices, self.action_indices)):
            raise ValueError("motors must match the configured G1 qpos order")
        self.dimension = len(self.config.motors)
        if set(self.config.camera_to_image_key.values()) != set(self.config.cameras):
            raise ValueError("Camera mapping must match cameras")
        ports = {"color_0": cfg.image_port, "color_1": 55556, "color_2": 55557}
        ports.update(cfg.camera_ports)
        if not self.config.cameras or any(key not in ports for key in self.config.camera_to_image_key):
            raise ValueError("Specify camera_ports for every configured camera source")
        self.camera_ports = {name: ports[key] for key, name in self.config.camera_to_image_key.items()}
        if len(set(self.camera_ports.values())) != len(self.camera_ports):
            raise ValueError("Each camera must have a separate ZMQ port")
        with (Path(cfg.initial_pose_episode) / "data.json").open() as f:
            states = json.load(f)["data"][0]["states"]
        self.initial_pose = np.concatenate([
            self.joints(states[part]["qpos"], part) for part in ("left_arm", "right_arm")
        ])
        self.initial_full_pose = self.initial_pose.copy()
        if has_hands:
            hand_pose = np.concatenate([
                self.joints(states[part]["qpos"], part, self.ee_dof)
                for part in ("left_ee", "right_ee")
            ])
            self.initial_full_pose = np.concatenate((self.initial_pose, hand_pose))

    @staticmethod
    def joints(values, label, size=7):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"{label} must contain {size} finite joint angles")
        return values

    def full_target(self, action):
        target = self.initial_full_pose.copy()
        target[self.action_indices] = self.joints(action, "policy action", self.dimension)
        return target

    def arm_target(self, action):
        return self.full_target(action)[:14]

    def observation_state(self, arm_state, hand_state=()):
        full = np.concatenate((self.joints(arm_state, "measured arms", 14),
                               self.joints(hand_state, "measured hands", 2 * self.ee_dof)))
        return full[self.state_indices]

    def read_hands(self, shared):
        if not self.ee_dof:
            return np.empty(0)
        with shared["lock"]:
            return self.joints(list(shared["state"][:]), "measured hands", 2 * self.ee_dof)

    def command_hands(self, shared, values):
        if not self.ee_dof:
            return
        values = self.joints(values, "hand target", 2 * self.ee_dof)
        for side, target in zip(("left", "right"), np.split(values, 2)):
            channel = shared[side]
            with channel.get_lock():
                if hasattr(channel, "value"):
                    channel.value = float(target[0])
                else:
                    channel[:] = target.tolist()

    def validate_features(self, features, policy_config):
        for key in ("observation.state", "action"):
            if tuple(features[key]["shape"]) != (self.dimension,):
                raise ValueError(f"Dataset {key} has the wrong joint dimension")
            names = features[key].get("names")
            if names and isinstance(names[0], list):
                names = names[0]
            if names and names != self.config.motors:
                raise ValueError(f"Dataset {key} joint names/order differ from robot_type")
        if tuple(policy_config.input_features["observation.state"].shape) != (self.dimension,):
            raise ValueError("Policy state has the wrong joint dimension")
        if tuple(policy_config.output_features["action"].shape) != (self.dimension,):
            raise ValueError("Policy action has the wrong joint dimension")
        expected = {f"observation.images.{name}" for name in self.config.cameras}
        for actual in (features, policy_config.input_features):
            if {key for key in actual if key.startswith("observation.images.")} != expected:
                raise ValueError(f"Expected exactly these cameras: {sorted(expected)}")


def eval_configured_arm(cfg, dataset, policy, preprocessor, postprocessor, layout):
    import cv2
    import torch
    import zmq
    from dataclasses import replace
    from lerobot.utils.utils import get_safe_torch_device
    from unitree_lerobot.eval_robot.make_robot import setup_robot_interface
    from unitree_lerobot.eval_robot.utils.utils import predict_action

    context = zmq.Context()
    sockets = {}
    arm_ctrl = arm_ik = None
    hand_shared = None
    try:
        for name, port in layout.camera_ports.items():
            sock = context.socket(zmq.SUB)
            sockets[name] = sock
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.CONFLATE, 1)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            sock.connect(f"tcp://{cfg.image_host}:{port}")

        def images():
            observation = {}
            for name, sock in sockets.items():
                frame = cv2.imdecode(np.frombuffer(sock.recv(), dtype=np.uint8), cv2.IMREAD_COLOR)
                key = f"observation.images.{name}"
                if frame is None or tuple(frame.shape) != tuple(dataset.meta.features[key]["shape"]):
                    raise ValueError(f"Invalid image or unexpected resolution: {name}")
                observation[key] = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return observation

        images()  # All streams must work before connecting to robot controllers.
        step = dataset[dataset.meta.episodes["dataset_from_index"][0]]
        if input("Enter 's' to move both arms to the episode pose and start configured arm control: ").lower() != "s":
            return
        robot = setup_robot_interface(replace(cfg, ee=layout.ee))
        arm_ctrl, arm_ik = robot["arm_ctrl"], robot["arm_ik"]
        hand_shared = robot.get("ee_shared_mem", {})
        if layout.ee_dof and robot["ee_dof"] != layout.ee_dof:
            raise ValueError("Controller hand dimension does not match configuration")
        hand_start = layout.read_hands(hand_shared)
        start = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float)
        if start.shape != (14,) or not np.isfinite(start).all():
            raise ValueError("Expected fourteen finite measured arm joints")
        # Limit the initial transition to 0.3 rad/s per joint.
        duration = max(3.0, float(np.max(np.abs(layout.initial_pose - start))) / 0.3)
        count = max(1, int(np.ceil(duration * cfg.frequency)))
        for i in range(1, count + 1):
            target = start + (layout.initial_pose - start) * (i / count)
            arm_ctrl.ctrl_dual_arm(target, arm_ik.solve_tau(target))
            hand_target = hand_start + (layout.initial_full_pose[14:] - hand_start) * (i / count)
            layout.command_hands(hand_shared, hand_target)
            time.sleep(duration / count)
        deadline = time.monotonic() + 10.0
        while True:
            arm_error = np.max(np.abs(layout.joints(arm_ctrl.get_current_dual_arm_q(), "measured arms", 14)
                                      - layout.initial_pose))
            hand_error = (np.max(np.abs(layout.read_hands(hand_shared) - layout.initial_full_pose[14:]))
                          if layout.ee_dof else 0.0)
            if arm_error <= 0.05 and hand_error <= 0.05:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Arms/hands did not reach the initial pose")
            time.sleep(1 / cfg.frequency)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        while True:
            tick = time.monotonic()
            observation = images()
            measured = layout.joints(arm_ctrl.get_current_dual_arm_q(), "measured arms", 14)
            state = layout.observation_state(measured, layout.read_hands(hand_shared))
            observation["observation.state"] = torch.tensor(state, dtype=torch.float32)
            action = predict_action(
                observation, policy, get_safe_torch_device(policy.config.device),
                preprocessor, postprocessor, policy.config.use_amp, step["task"],
                robot_type=cfg.robot_type,
            )
            target = layout.full_target(action.cpu().numpy())
            arm_ctrl.ctrl_dual_arm(target[:14], arm_ik.solve_tau(target[:14]))
            layout.command_hands(hand_shared, target[14:])
            time.sleep(max(0.0, 1 / cfg.frequency - (time.monotonic() - tick)))
    finally:
        try:
            if arm_ctrl is not None:
                # Hold measured pose on interruption or camera/policy failure.
                target = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float)
                if target.shape == (14,) and np.isfinite(target).all():
                    arm_ctrl.ctrl_dual_arm(target, arm_ik.solve_tau(target))
                if hand_shared is not None:
                    layout.command_hands(hand_shared, layout.read_hands(hand_shared))
        finally:
            for sock in sockets.values():
                sock.close(linger=0)
            context.term()
