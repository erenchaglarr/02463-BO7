from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from skopt import dummy_minimize, gp_minimize
from skopt.space import Categorical, Integer, Real
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Omniglot

# Robust imports (works both as package and direct script)
try:
    from bayesianoptimization.model import CNNModel
    from bayesianoptimization.data import download
except ImportError:
    from model import CNNModel
    from data import download


# -----------------------
# Reproducibility / device
# -----------------------
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------
# Hyperparameter search space
# -----------------------
# IMPORTANT:
# - depth > 6 with repeated MaxPool2d(2) will collapse Omniglot spatial dims
# - kernel_size should be odd so padding is symmetric
search_space = [
    Integer(1, 6, name="depth"),
    Categorical([3, 5, 7, 9], name="kernel_size"),
    Real(0.0, 0.5, name="dropout_rate"),
]


# -----------------------
# Data (cached globally)
# -----------------------
TRAIN_LOADER = None
VAL_LOADER = None
NUM_CLASSES = None


def build_dataloaders(
    data_root: str = "data",
    batch_size: int = 32,
    val_ratio: float = 0.2,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train/val loaders from Omniglot background split."""
    transform = transforms.ToTensor()  # PIL -> Tensor [1, H, W]

    dataset = Omniglot(
        root=data_root,
        background=True,   # train-like split
        download=False,
        transform=transform,
    )

    # torchvision Omniglot stores character classes in _characters
    # background split should be 964 classes
    if hasattr(dataset, "_characters"):
        num_classes = len(dataset._characters)
    else:
        # fallback (older/newer torchvision internals)
        labels = [item[1] for item in dataset._flat_character_images]
        num_classes = len(set(labels))

    generator = torch.Generator().manual_seed(42)
    n_train = int((1 - val_ratio) * len(dataset))
    n_val = len(dataset) - n_train

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # safer on macOS/MPS
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Dataset size: {len(dataset)} | Train: {len(train_set)} | Val: {len(val_set)} | Classes: {num_classes}")
    return train_loader, val_loader, num_classes


# -----------------------
# Objective for BO
# -----------------------
def objective(params):
    depth, kernel_size, dropout_rate = params
    try:
        validation_score = train_model(depth, kernel_size, dropout_rate)
        return -validation_score  # skopt minimizes
    except RuntimeError as e:
        # If a trial fails unexpectedly, penalize it instead of crashing BO
        print(f"Trial failed for params={params}: {e}")
        return 1.0  # bad loss (since objective is -accuracy, valid values are usually <= 0)


def train_model(depth: int, kernel_size: int, dropout_rate: float) -> float:
    global TRAIN_LOADER, VAL_LOADER, NUM_CLASSES

    # Make each trial deterministic
    torch.manual_seed(42)
    np.random.seed(42)

    if TRAIN_LOADER is None or VAL_LOADER is None or NUM_CLASSES is None:
        TRAIN_LOADER, VAL_LOADER, NUM_CLASSES = build_dataloaders(data_root="data", batch_size=32)

    model = CNNModel(
        depth=depth,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        num_classes=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 3  # keep small while tuning (increase later)

    # Train
    for _ in range(epochs):
        model.train()
        for x, y in TRAIN_LOADER:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    # Validate
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in VAL_LOADER:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    print(f"depth={depth}, kernel={kernel_size}, dropout={dropout_rate:.3f} -> val_acc={accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    # Safe to call repeatedly; won't re-download if already present
    download(data_path=Path("data"), background=True, evaluation=True)

    # Build loaders once
    TRAIN_LOADER, VAL_LOADER, NUM_CLASSES = build_dataloaders(data_root="data", batch_size=32)

    # Bayesian optimization with Expected Improvement
    results_ei = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        n_random_starts=5,
        acq_func="EI",
        random_state=42,
    )

    # Bayesian optimization with Probability of Improvement
    results_pi = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        n_random_starts=5,
        acq_func="PI",
        random_state=42,
    )

    # Random search baseline
    results_random = dummy_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        random_state=42,
    )

    print("\n=== Results ===")
    print("Best EI accuracy:", -results_ei.fun, "params:", results_ei.x)
    print("Best PI accuracy:", -results_pi.fun, "params:", results_pi.x)
    print("Best Random accuracy:", -results_random.fun, "params:", results_random.x)