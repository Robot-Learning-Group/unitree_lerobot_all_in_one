"""Convert recorded G1 Dex3 episodes with three cameras using the LeRobot API.

This adapter leaves the vendored Unitree converter and the raw recordings intact.
It rejects other joint/camera layouts instead of padding or fabricating inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from unitree_lerobot.utils.constants import ROBOT_CONFIGS


CAMERAS = {
    "color_0": "cam_high",
    "color_1": "cam_left_wrist",
    "color_2": "cam_right_wrist",
}
PARTS = ("left_arm", "right_arm", "left_ee", "right_ee")


def joints(frame, key):
    values = [np.asarray(frame[key][part]["qpos"], dtype=np.float32) for part in PARTS]
    if any(value.shape != (7,) for value in values):
        raise ValueError(f"Expected 7 joints for each of {PARTS}; got {[v.shape for v in values]}")
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite {key}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing episode_XXXX folders")
    parser.add_argument("--repo-id", required=True, help="Local dataset identifier, e.g. local/pick_cube_dex3_v3")
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="Explicit raw episode numbers")
    args = parser.parse_args()
    repo_path = Path(args.repo_id)
    if repo_path.is_absolute() or ".." in repo_path.parts or len(repo_path.parts) != 2:
        parser.error("--repo-id must have the form namespace/name")
    if len(set(args.episodes)) != len(args.episodes):
        parser.error("--episodes must not contain duplicates")
    destination = HF_LEROBOT_HOME / args.repo_id
    if destination.exists():
        raise FileExistsError(f"Dataset already exists: {destination}; choose a new --repo-id")

    # Validate every selected recording before creating the destination.
    episodes = []
    for number in args.episodes:
        path = args.raw_dir / f"episode_{number:04d}" / "data.json"
        raw = path.read_bytes()
        episode = json.loads(raw)
        if not episode["data"] or episode["info"]["image"]["fps"] != 30:
            raise ValueError(f"Expected a nonempty 30 FPS recording: {path}")
        for frame in episode["data"]:
            joints(frame, "states")
            joints(frame, "actions")
            if set(frame["colors"]) != set(CAMERAS):
                raise ValueError(f"Expected exactly three cameras in {path}")
            for relative in frame["colors"].values():
                image_path = (path.parent / relative).resolve()
                if not image_path.is_relative_to(path.parent.resolve()) or not image_path.is_file():
                    raise ValueError(f"Missing or out-of-directory image: {image_path}")
        episodes.append((path, episode, hashlib.sha256(raw).hexdigest()))

    motors = ROBOT_CONFIGS["Unitree_G1_Dex3"].motors
    features = {
        key: {"dtype": "float32", "shape": (28,), "names": motors}
        for key in ("observation.state", "action")
    }
    features.update({
        f"observation.images.{name}": {
            "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]
        }
        for name in CAMERAS.values()
    })
    cv2.setNumThreads(1)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, root=destination, fps=30,
        robot_type="Unitree_G1_Dex3_Sim", features=features,
        image_writer_threads=4, video_backend="torchcodec",
    )
    provenance = []
    try:
        for path, episode, digest in episodes:
            for frame in episode["data"]:
                sample = {
                    "observation.state": joints(frame, "states"),
                    "action": joints(frame, "actions"),
                    "task": episode.get("text", {}).get("goal", ""),
                }
                for key, name in CAMERAS.items():
                    image = cv2.imread(str(path.parent / frame["colors"][key]))
                    if image is None or image.shape != (480, 640, 3):
                        raise ValueError(f"Expected a readable 480x640 RGB image: {path.parent / frame['colors'][key]}")
                    sample[f"observation.images.{name}"] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                dataset.add_frame(sample)
            dataset.save_episode()
            provenance.append({"raw_episode": path.parent.name, "frames": len(episode["data"]), "json_sha256": digest})
            print(f"Converted {path.parent.name}: {len(episode['data'])} frames", flush=True)
        dataset.finalize()
    finally:
        dataset.stop_image_writer()
    (destination / "meta" / "conversion_source.json").write_text(
        json.dumps({"camera_mapping": CAMERAS, "episodes": provenance}, indent=2) + "\n"
    )
    print(f"Saved {len(episodes)} episodes to {destination}", flush=True)


if __name__ == "__main__":
    main()
