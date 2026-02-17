# src/bayesianoptimization/data.py

from __future__ import annotations

from pathlib import Path

import typer
from torchvision.datasets import Omniglot
<<<<<<< HEAD
<<<<<<< HEAD


def download(
    data_path: Path = typer.Argument(Path("data"), help="Folder to download/store Omniglot (default: data/)"),
    background: bool = typer.Option(True, help="Download background split (train-like)."),
    evaluation: bool = typer.Option(True, help="Download evaluation split (test-like)."),
) -> None:
    """Download Omniglot with torchvision into data_path."""
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading into: {data_path.resolve()}")

    if not background and not evaluation:
        raise typer.BadParameter("At least one of --background / --evaluation must be True.")

    if background:
        Omniglot(root=str(data_path), background=True, download=True)
        print("✓ Downloaded images_background")

    if evaluation:
        Omniglot(root=str(data_path), background=False, download=True)
        print("✓ Downloaded images_evaluation")

    print("Done.")

=======
from torch.utils.data import Dataset
=======
>>>>>>> e4e4809 (working data.py)


def download(
    data_path: Path = typer.Argument(Path("data"), help="Folder to download/store Omniglot (default: data/)"),
    background: bool = typer.Option(True, help="Download background split (train-like)."),
    evaluation: bool = typer.Option(True, help="Download evaluation split (test-like)."),
) -> None:
    """Download Omniglot with torchvision into data_path."""
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading into: {data_path.resolve()}")

    if not background and not evaluation:
        raise typer.BadParameter("At least one of --background / --evaluation must be True.")

    if background:
        Omniglot(root=str(data_path), background=True, download=True)
        print("✓ Downloaded images_background")

    if evaluation:
        Omniglot(root=str(data_path), background=False, download=True)
        print("✓ Downloaded images_evaluation")

    print("Done.")


>>>>>>> f03d60b (changes in data.py)
if __name__ == "__main__":
    typer.run(download)
