"""Run PromptIR inference on HW4 test images and create pred.npz."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from promptir.data import HW4TestDataset, image_to_uint8_chw
from promptir.model import PromptIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="hw4_realse_dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="pred.npz")
    parser.add_argument("--save-images", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument("--tile-size", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_to_multiple(input_tensor, multiple=8):
    height, width = input_tensor.shape[-2:]
    padded_h = ((height + multiple - 1) // multiple) * multiple
    padded_w = ((width + multiple - 1) // multiple) * multiple
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h == 0 and pad_w == 0:
        return input_tensor, height, width
    padded = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, height, width


def tile_forward(model, input_tensor, tile_size, tile_overlap):
    _, channels, height, width = input_tensor.shape
    tile = min(tile_size, height, width)
    if tile % 8 != 0:
        raise ValueError("--tile-size must be a multiple of 8")
    stride = tile - tile_overlap
    if stride <= 0:
        raise ValueError("--tile-overlap must be smaller than --tile-size")

    h_idx_list = list(range(0, height - tile, stride)) + [height - tile]
    w_idx_list = list(range(0, width - tile, stride)) + [width - tile]
    output = torch.zeros_like(input_tensor)
    weight = torch.zeros_like(input_tensor)

    for top in h_idx_list:
        for left in w_idx_list:
            patch = input_tensor[..., top:top + tile, left:left + tile]
            restored_patch = model(patch)
            mask = torch.ones(
                1,
                channels,
                restored_patch.shape[-2],
                restored_patch.shape[-1],
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
            output[..., top:top + tile, left:left + tile] += restored_patch
            weight[..., top:top + tile, left:left + tile] += mask

    return output / weight


def load_model(checkpoint_path, device):
    model = PromptIR(decoder=True).to(device)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    if "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {}
    for key, value in state.items():
        if key.startswith("net."):
            key = key[len("net."):]
        cleaned_state[key] = value

    model.load_state_dict(cleaned_state, strict=True)
    model.eval()
    return model


def save_png(array_chw, path):
    from PIL import Image

    image = Image.fromarray(array_chw.transpose(1, 2, 0))
    image.save(path)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = HW4TestDataset(args.dataset_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    save_images_dir = None
    if args.save_images is not None:
        save_images_dir = Path(args.save_images)
        save_images_dir.mkdir(parents=True, exist_ok=True)

    predictions = {}
    print(f"Device: {device}")
    print(f"Test images: {len(dataset)}")

    with torch.no_grad():
        progress = tqdm(loader, desc="Infer")
        for index, (names, degraded) in enumerate(progress, start=1):
            degraded = degraded.to(device)
            padded, height, width = pad_to_multiple(degraded)
            if args.tile_size > 0:
                restored = tile_forward(
                    model,
                    padded,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                )
            else:
                restored = model(padded)
            restored = restored[..., :height, :width]
            array_chw = image_to_uint8_chw(restored[0])
            predictions[names[0]] = array_chw

            if save_images_dir is not None:
                save_png(array_chw, save_images_dir / names[0])
            if 0 < args.max_images <= index:
                break

    np.savez(args.output, **predictions)
    print(f"Saved {len(predictions)} restored images to {args.output}")


if __name__ == "__main__":
    main()
