"""Offline build checks. Never create a robot controller or a DDS participant."""

import importlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader
from torchcodec.decoders import VideoDecoder

torch.set_num_threads(2)

# Decord's official 0.6.0 wheel has incorrect internal platform tags. Preserve
# the upstream package and allow only this exact pip-check diagnostic; validate
# its native decoder below. All missing/conflicting dependencies remain fatal.
# https://github.com/dmlc/decord/issues/366
check = subprocess.run(["python", "-m", "pip", "check"], text=True, capture_output=True)
diagnostic = check.stdout.strip()
if check.returncode and not (
    check.returncode == 1 and diagnostic == "decord 0.6.0 is not supported on this platform"
):
    raise RuntimeError(check.stdout + check.stderr)
print(diagnostic, flush=True)


for module in (
    "lerobot.policies.act.modeling_act",
    "lerobot.policies.groot.modeling_groot",
    "unitree_sdk2py.core.channel",
    "unitree_lerobot.eval_robot.eval_g1",
    "unitree_lerobot.eval_robot.eval_g1_dataset",
    "unitree_lerobot.utils.convert_unitree_json_to_lerobot",
    "pinocchio.casadi",
    "cv2",
    "zmq",
):
    importlib.import_module(module)
    print(f"import OK: {module}", flush=True)

with tempfile.TemporaryDirectory() as tmp:
    video = Path(tmp) / "smoke.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=c=red:s=64x64:r=5", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    assert VideoDecoder(str(video), device="cpu")[0].shape == (3, 64, 64)
    assert VideoReader(str(video))[0].shape == (64, 64, 3)
    print("FFmpeg encode / TorchCodec and Decord decode OK", flush=True)

from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK

ik = G1_29_ArmIK(Unit_Test=True, Visualization=False)
tau = np.asarray(ik.solve_tau(np.zeros(14)))
assert tau.shape == (14,) and np.isfinite(tau).all()
print("G1 URDF / CasADi model / gravity torque OK", flush=True)

for module in (
    "lerobot.scripts.lerobot_train",
    "unitree_lerobot.utils.convert_unitree_json_to_lerobot",
    "unitree_lerobot.eval_robot.eval_g1_dataset",
    "unitree_lerobot.eval_robot.eval_g1",
):
    subprocess.run(["python", "-m", module, "--help"], check=True, stdout=subprocess.DEVNULL)
    print(f"CLI OK: {module}", flush=True)

print(f"Offline checks passed; torch={torch.__version__}. GPU and real robot not tested.")
