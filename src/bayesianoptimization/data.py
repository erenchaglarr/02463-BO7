from pathlib import Path
import typer
from torchvision.datasets import Omniglot
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

if __name__ == "__main__":
    typer.run(preprocess)
