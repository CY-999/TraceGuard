"""Dataset loading for the minimal FedAvg loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import datasets, transforms


CIFAR_NORMALIZATION = (
    (0.4914, 0.4822, 0.4465),
    (0.2470, 0.2435, 0.2616),
)
TINY_IMAGENET_NORMALIZATION = (
    (0.4802, 0.4481, 0.3975),
    (0.2302, 0.2265, 0.2262),
)
DATASET_NORMALIZATION = {
    "fakedata": CIFAR_NORMALIZATION,
    "cifar10": CIFAR_NORMALIZATION,
    "cifar100": CIFAR_NORMALIZATION,
    "tinyimagenet": TINY_IMAGENET_NORMALIZATION,
}


def get_normalized_input_bounds(
    config: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel-wise model-input bounds after optional Normalize."""
    dataset_name = str(config.get("dataset", {}).get("name", "")).lower()
    normalization = DATASET_NORMALIZATION.get(dataset_name)
    if normalization is None:
        lower = torch.zeros(3, 1, 1, device=device, dtype=dtype)
        upper = torch.ones(3, 1, 1, device=device, dtype=dtype)
        return lower, upper

    mean, std = normalization
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(-1, 1, 1)
    lower = (torch.zeros_like(mean_tensor) - mean_tensor) / std_tensor
    upper = (torch.ones_like(mean_tensor) - mean_tensor) / std_tensor
    return lower, upper


def _limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def _fake_vision_data(
    config: dict[str, Any],
    train_transform,
    test_transform,
    *,
    num_classes: int,
) -> tuple[Dataset, Dataset]:
    train_size = int(config["dataset"].get("max_train_samples") or 320)
    test_size = int(config["dataset"].get("max_test_samples") or 256)
    train_set = datasets.FakeData(
        size=train_size,
        image_size=(3, 32, 32),
        num_classes=num_classes,
        transform=train_transform,
    )
    test_set = datasets.FakeData(
        size=test_size,
        image_size=(3, 32, 32),
        num_classes=num_classes,
        transform=test_transform,
    )
    return train_set, test_set


def _cifar_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*CIFAR_NORMALIZATION),
        ]
    )
    return train_transform, test_transform


def _tiny_imagenet_transforms(image_size: int = 64):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*TINY_IMAGENET_NORMALIZATION),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(*TINY_IMAGENET_NORMALIZATION),
        ]
    )
    return train_transform, test_transform


class TinyImageNetValDataset(Dataset):
    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform=None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.image_dir = root / "val" / "images"
        annotation_path = root / "val" / "val_annotations.txt"
        if not self.image_dir.is_dir() or not annotation_path.is_file():
            raise FileNotFoundError(
                "Tiny-ImageNet validation data is incomplete. Expected "
                f"{self.image_dir} and {annotation_path}."
            )

        samples: list[tuple[Path, int]] = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                samples.append((self.image_dir / filename, class_to_idx[wnid]))

        if not samples:
            raise RuntimeError(
                "Tiny-ImageNet validation annotations produced no samples. "
                "Please check val/val_annotations.txt and wnids.txt."
            )
        self.samples = samples
        self.targets = [label for _, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def load_fakedata(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    train_transform, test_transform = _cifar_transforms()
    num_classes = int(config.get("model", {}).get("num_classes", 10))
    return _fake_vision_data(
        config,
        train_transform,
        test_transform,
        num_classes=num_classes,
    )


def _load_cifar(config: dict[str, Any], *, name: str) -> tuple[Dataset, Dataset]:
    data_dir = config["dataset"]["data_dir"]
    download = bool(config["dataset"].get("download", False))
    train_transform, test_transform = _cifar_transforms()
    dataset_cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    num_classes = 10 if name == "cifar10" else 100

    root = Path(data_dir)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*dtype\(\): align should be passed.*",
                category=Warning,
            )
            train_set = dataset_cls(
                root=str(root),
                train=True,
                transform=train_transform,
                download=download,
            )
            test_set = dataset_cls(
                root=str(root),
                train=False,
                transform=test_transform,
                download=download,
            )
    except RuntimeError as exc:
        if bool(config.get("debug", {}).get("enabled", False)) and bool(
            config["dataset"].get("fake_data_on_missing", False)
        ):
            print(
                f"{name} was not found under dataset.data_dir and debug download is disabled; "
                "using torchvision.datasets.FakeData fallback."
            )
            return _fake_vision_data(
                config,
                train_transform,
                test_transform,
                num_classes=num_classes,
            )
        raise RuntimeError(
            f"{name} was not found under dataset.data_dir. "
            "Place the configured dataset under dataset.data_dir, set dataset.download=true, "
            "or use debug mode with dataset.fake_data_on_missing=true."
        ) from exc

    train_set = _limit_dataset(train_set, config["dataset"].get("max_train_samples"))
    test_set = _limit_dataset(test_set, config["dataset"].get("max_test_samples"))
    return train_set, test_set


def _find_tiny_imagenet_root(data_dir: str | Path) -> Path:
    base = Path(data_dir)
    candidates = [
        base / "tiny-imagenet-200",
        base / "tinyimagenet",
        base,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir() and (candidate / "wnids.txt").is_file():
            return candidate
    raise FileNotFoundError(
        "Tiny-ImageNet was not found or has an unsupported layout. Expected a directory like:\n"
        "  data/tiny-imagenet-200/\n"
        "    train/<wnid>/images/*.JPEG\n"
        "    val/images/*.JPEG\n"
        "    val/val_annotations.txt\n"
        "    wnids.txt\n"
        f"Configured dataset.data_dir was: {base}"
    )


def load_tiny_imagenet(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    if bool(config["dataset"].get("download", False)):
        raise ValueError(
            "Automatic Tiny-ImageNet download is not supported. "
            "Please prepare the dataset under dataset.data_dir."
        )

    root = _find_tiny_imagenet_root(config["dataset"]["data_dir"])
    image_size = int(config["dataset"].get("image_size", 64))
    train_transform, test_transform = _tiny_imagenet_transforms(image_size=image_size)

    train_root = root / "train"
    train_set = ImageFolder(str(train_root), transform=train_transform)
    if len(train_set.classes) != 200:
        raise RuntimeError(
            f"Tiny-ImageNet train split should contain 200 classes, found {len(train_set.classes)}."
        )
    test_set = TinyImageNetValDataset(
        root,
        class_to_idx=train_set.class_to_idx,
        transform=test_transform,
    )

    train_set = _limit_dataset(train_set, config["dataset"].get("max_train_samples"))
    test_set = _limit_dataset(test_set, config["dataset"].get("max_test_samples"))
    return train_set, test_set


def load_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset]:
    name = config["dataset"]["name"].lower()
    if name == "fakedata":
        return load_fakedata(config)
    if name in {"cifar10", "cifar100"}:
        return _load_cifar(config, name=name)
    if name == "tinyimagenet":
        return load_tiny_imagenet(config)
    raise ValueError(f"Supported datasets are fakedata, cifar10, cifar100, and tinyimagenet; got: {name}")
