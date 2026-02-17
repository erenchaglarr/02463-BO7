# src/bayesianoptimization/data.py

from __future__ import annotations

from pathlib import Path
import typer
from torchvision.datasets import Omniglot
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

class MyDataset(Dataset):
    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        return 0  # you can implement later

    def __getitem__(self, index: int):
        raise IndexError  # you can implement later

    def preprocess(self, output_folder: Path) -> None:
        output_folder.mkdir(parents=True, exist_ok=True)

        # download Omniglot into self.data_path
        Omniglot(root=str(self.data_path), background=True, download=True)
        Omniglot(root=str(self.data_path), background=False, download=True)

        print("Omniglot downloaded into:", self.data_path.resolve())

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)

>>>>>>> f03d60b (changes in data.py)
if __name__ == "__main__":
    typer.run(download)
