"""UniArmL1 robot configuration — all-in-one config module.

Modify default values in this file to adjust all parameters without requiring additional YAML files.

Contains:
- RobotConfig: base class with subclass registry
- UniArmL1Config: UniArmL1 hardware parameters (serial port, motors, PD gains, etc.)
- UniArmL1RobotConfig: composed config (RobotConfig + UniArmL1Config)
- TeleopConfig: teleop flow configuration (input mode, cameras, recording, etc.)
"""

from dataclasses import dataclass, field
from typing import Callable, ClassVar, TypeAlias, TypeVar
from lerobot.cameras.configs import CameraConfig

T = TypeVar("T", bound="RobotConfig")


@dataclass
class RobotConfig:
    """Base class for robot configurations with subclass registry."""

    _registry: ClassVar[dict[str, type["RobotConfig"]]] = {}

    id: str | None = None

    @classmethod
    def register_subclass(cls, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a subclass with a key."""
        def decorator(subclass: type[T]) -> type[T]:
            cls._registry[key] = subclass
            return subclass
        return decorator

    @classmethod
    def get_subclass(cls, key: str) -> type["RobotConfig"] | None:
        return cls._registry.get(key)


T = TypeVar("T", bound="RobotConfig")


# ── UniArmL1 hardware config ────────────────────────────────────




@dataclass
class UniArmL1Config:
    """UniArmL1 robot arm hardware configuration."""

    port: str

    disable_torque_on_disconnect: bool = True

    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    use_degrees: bool = False

    urdf_path: str | None = "../assets/uniarml1/urdf/UniArmL1.urdf"

    use_vr: bool = False
    id: str | None = None

    # Initialization movement duration (seconds): smooth transition from current position to initial position
    # Set to 0 for immediate jump (may cause jitter)
    init_move_duration: float = 2.0

    # Motor configuration: joint name -> list of motor IDs (supports dual motors)
    # Dual-motor joint: two motors drive the same joint synchronously
    joint_motor_ids: dict[str, list[int]] = field(default_factory=lambda: {
        "shoulder_pan": [0],
        "shoulder_lift": [1, 6],  # dual motor
        "elbow_flex": [2, 7],     # dual motor
        "wrist_flex": [3],
        "wrist_roll": [4],
        "gripper": [5],
    })

    # Motor PD control parameters (in motor ID order, index equals motor ID)
    kp_loop: list[float] | None = None
    kd_loop: list[float] | None = None

    # Motor default kp/kd (for motor control mode)
    kp_default: list[float] | None = None
    kd_default: list[float] | None = None

    # Whether to run without a real robot arm (simulation mode)
    no_real_robot: bool = False


@RobotConfig.register_subclass("uniarm_l1_follower")
@dataclass
class UniArmL1RobotConfig(RobotConfig, UniArmL1Config):
    pass


UniArmL1ConfigType: TypeAlias = UniArmL1RobotConfig


# ── Teleop flow config ─────────────────────────────────────────

@dataclass
class TeleopConfig:
    """Teleoperation flow configuration — modify default values to adjust parameters."""

    input: str = "vr"  # vr | keyboard | leader
    port: str = "/dev/ttyACM0"  # Follower arm serial port
    leader_port: str = "/dev/ttyACM3"  # Leader arm serial port
    urdf_path: str = "../assets/uniarml1/urdf/UniArmL1.urdf"
    cameras: dict[str, dict] = field(default_factory=lambda: {
        "head": {"device": 2, "fps": 30, "width": 640, "height": 480, "fourcc": "MJPG", "warmup_s": 1.0, "flip_vertical": True},
        "wrist": {"device": 0, "fps": 30, "width": 640, "height": 480, "fourcc": "MJPG", "warmup_s": 1.0, "flip_vertical": True},
    })
    no_camera: bool = False
    record: bool = False
    task_dir: str = "./data/teleop"
    task_goal: str = ""
    record_hz: int = 50
    meshcat: bool = False
    no_real_robot: bool = False
