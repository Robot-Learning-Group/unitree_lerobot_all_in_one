from pathlib import Path

import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def local_push_to_hub(
    repo_id: str,
    root_path: Path = None,
):
    dataset = LeRobotDataset(repo_id=repo_id, root=root_path)
    dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(local_push_to_hub)
