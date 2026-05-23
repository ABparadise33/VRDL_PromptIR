"""HW4 rain/snow datasets for PromptIR training and submission."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def image_to_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1))


def image_to_uint8_chw(tensor):
    tensor = tensor.detach().clamp(0.0, 1.0).cpu()
    array = tensor.numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    return array


def list_image_files(root):
    root = Path(root)
    return sorted(
        [path for path in root.iterdir() if path.suffix.lower() in IMG_EXTENSIONS],
        key=lambda path: path.name,
    )


def natural_image_key(path):
    stem = Path(path).stem
    try:
        return int(stem)
    except ValueError:
        return stem


def paired_clean_name(degraded_name):
    stem = Path(degraded_name).stem
    suffix = Path(degraded_name).suffix
    if stem.startswith("rain-"):
        return f"rain_clean-{stem.split('-', 1)[1]}{suffix}"
    if stem.startswith("snow-"):
        return f"snow_clean-{stem.split('-', 1)[1]}{suffix}"
    raise ValueError(f"Unsupported degraded filename: {degraded_name}")


def build_paired_samples(dataset_root):
    dataset_root = Path(dataset_root)
    degraded_dir = dataset_root / "train" / "degraded"
    clean_dir = dataset_root / "train" / "clean"
    samples = []
    for degraded_path in list_image_files(degraded_dir):
        clean_path = clean_dir / paired_clean_name(degraded_path.name)
        if clean_path.exists():
            samples.append((degraded_path, clean_path))
    if not samples:
        raise RuntimeError(f"No training pairs found under {dataset_root}")
    return samples


def split_train_val_samples(samples, val_ratio=0.1, seed=42):
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    samples = list(samples)
    if val_ratio == 0.0:
        return samples, []

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_ratio)))
    val_indices = set(indices[:val_count])
    train_samples = [
        sample for index, sample in enumerate(samples) if index not in val_indices
    ]
    val_samples = [
        sample for index, sample in enumerate(samples) if index in val_indices
    ]
    return train_samples, val_samples


def degradation_from_sample(sample):
    degraded_path, _ = sample
    name = degraded_path.name
    if name.startswith("rain-"):
        return "rain"
    if name.startswith("snow-"):
        return "snow"
    return "unknown"


def oversample_rain_samples(samples, rain_oversample=1):
    if rain_oversample < 1:
        raise ValueError("--rain-oversample must be >= 1")
    if rain_oversample == 1:
        return list(samples)

    oversampled = []
    for sample in samples:
        oversampled.append(sample)
        if degradation_from_sample(sample) == "rain":
            oversampled.extend([sample] * (rain_oversample - 1))
    return oversampled


def pad_if_smaller(array, patch_size):
    h, w = array.shape[:2]
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return array
    return np.pad(
        array,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )


def random_crop_pair(degraded, clean, patch_size):
    degraded = pad_if_smaller(degraded, patch_size)
    clean = pad_if_smaller(clean, patch_size)
    h, w = degraded.shape[:2]
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    degraded = degraded[top:top + patch_size, left:left + patch_size]
    clean = clean[top:top + patch_size, left:left + patch_size]
    return degraded, clean


def augment_pair(degraded, clean):
    if random.random() < 0.5:
        degraded = np.flip(degraded, axis=1)
        clean = np.flip(clean, axis=1)
    if random.random() < 0.5:
        degraded = np.flip(degraded, axis=0)
        clean = np.flip(clean, axis=0)
    rotations = random.randint(0, 3)
    if rotations:
        degraded = np.rot90(degraded, rotations)
        clean = np.rot90(clean, rotations)
    return np.ascontiguousarray(degraded), np.ascontiguousarray(clean)


class HW4RainSnowTrainDataset(Dataset):
    def __init__(self, dataset_root, patch_size=128, augment=True, samples=None):
        self.dataset_root = Path(dataset_root)
        self.patch_size = patch_size
        self.augment = augment
        self.samples = samples if samples is not None else build_paired_samples(
            self.dataset_root
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        degraded_path, clean_path = self.samples[index]
        degraded = np.asarray(Image.open(degraded_path).convert("RGB"))
        clean = np.asarray(Image.open(clean_path).convert("RGB"))

        degraded, clean = random_crop_pair(degraded, clean, self.patch_size)
        if self.augment:
            degraded, clean = augment_pair(degraded, clean)

        degraded_tensor = torch.from_numpy(
            degraded.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        clean_tensor = torch.from_numpy(
            clean.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        return degraded_path.name, degraded_tensor, clean_tensor


class HW4RainSnowValDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        degraded_path, clean_path = self.samples[index]
        degraded = image_to_tensor(Image.open(degraded_path))
        clean = image_to_tensor(Image.open(clean_path))
        return degraded_path.name, degraded, clean


class HW4TestDataset(Dataset):
    def __init__(self, dataset_root):
        self.degraded_dir = Path(dataset_root) / "test" / "degraded"
        self.paths = sorted(list_image_files(self.degraded_dir), key=natural_image_key)
        if not self.paths:
            raise RuntimeError(f"No test images found under {self.degraded_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = image_to_tensor(Image.open(path))
        return path.name, tensor
