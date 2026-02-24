from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from torchvision.datasets import Omniglot

app = typer.Typer()


def download(
    data_path: Path = Path("data"),
    background: bool = True,
    evaluation: bool = True,
) -> None:
    """Download Omniglot with torchvision into data_path."""
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading into: {data_path.resolve()}")

    if not background and not evaluation:
        raise ValueError("At least one of background/evaluation must be True.")

    if background:
        Omniglot(root=str(data_path), background=True, download=True)
        print("✓ Downloaded images_background")

    if evaluation:
        Omniglot(root=str(data_path), background=False, download=True)
        print("✓ Downloaded images_evaluation")

    print("Done.")


@app.command()
def download_cli(
    data_path: Annotated[
        Path, typer.Argument(help="Folder to download/store Omniglot (default: data/)")
    ] = Path("data"),
    background: Annotated[
        bool, typer.Option(help="Download background split (train-like).")
    ] = True,
    evaluation: Annotated[
        bool, typer.Option(help="Download evaluation split (test-like).")
    ] = True,
) -> None:
    """CLI wrapper around download()."""
    try:
        download(data_path=data_path, background=background, evaluation=evaluation)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


if __name__ == "__main__":
    app()