# HW4 PromptIR Baseline

This is a clean baseline for the HW4 image restoration task. The model is the
original PromptIR architecture trained from scratch on the provided rain/snow
training pairs. No pretrained weights or external data are used.

## Source

- Paper: PromptIR: Prompting for All-in-One Image Restoration, NeurIPS 2023
- Official code: https://github.com/va1shn9v/PromptIR
- HW4 requirement: train a single PromptIR model for both Rain and Snow.

## Environment

```bash
pip install -r requirements.txt
```

## Vast.ai RTX 3090 Training Steps

Use a PyTorch CUDA image on Vast.ai. A single RTX 3090 with 24 GB VRAM should be
enough for this baseline with `patch-size 128` and a batch size around `4` to
`8`. If you hit CUDA out-of-memory, lower `--batch-size` first.

### 1. SSH into the instance

Copy the SSH command from Vast.ai. It usually looks like this:

```bash
ssh -p <PORT> root@<HOST>
```

### 2. Clone the code

```bash
cd /workspace
git clone https://github.com/ABparadise33/VRDL_PromptIR.git
cd VRDL_PromptIR
```

### 3. Install dependencies

Most Vast.ai PyTorch templates already include CUDA-enabled PyTorch. Install the
remaining Python dependencies from the repo:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that PyTorch can see the 3090:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### 4. Download the HW4 dataset

Do not commit the dataset to GitHub. Download it directly on the Vast.ai
instance with the provided Google Drive link:

```bash
python -m pip install gdown
gdown 1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w -O hw4_realse_dataset.zip
unzip hw4_realse_dataset.zip
```

If the zip extracts into a nested directory, move or rename it so the final
dataset path is `hw4_realse_dataset`:

```bash
find . -maxdepth 3 -type d | sort | head -50
```

As an alternative, upload the dataset from your local machine:

```bash
scp -P <PORT> -r /path/to/hw4_realse_dataset root@<HOST>:/workspace/VRDL_PromptIR/
```

After upload, the expected layout is:

```text
VRDL_PromptIR/
  hw4_realse_dataset/
    train/
      degraded/
      clean/
    test/
      degraded/
```

### 5. Start a persistent terminal session

This keeps training alive if your SSH connection disconnects:

```bash
tmux new -s promptir
```

If `tmux` is not installed:

```bash
apt-get update
apt-get install -y tmux
tmux new -s promptir
```

### 6. Train from scratch

Baseline command:

```bash
python train.py \
  --dataset-root hw4_realse_dataset \
  --output-dir runs/promptir_baseline \
  --epochs 120 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --device cuda
```

Safer 3090 command if memory is tight:

```bash
python train.py \
  --dataset-root hw4_realse_dataset \
  --output-dir runs/promptir_baseline \
  --epochs 120 \
  --batch-size 4 \
  --patch-size 128 \
  --num-workers 4 \
  --device cuda
```

Resume training from the latest checkpoint:

```bash
python train.py \
  --dataset-root hw4_realse_dataset \
  --output-dir runs/promptir_baseline \
  --epochs 120 \
  --batch-size 8 \
  --patch-size 128 \
  --num-workers 4 \
  --device cuda \
  --resume runs/promptir_baseline/latest.pt
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach later with:

```bash
tmux attach -t promptir
```

Training writes the epoch loss log and the report-ready loss curve here:

```text
runs/promptir_baseline/latest.pt
runs/promptir_baseline/best.pt
runs/promptir_baseline/epoch_010.pt
runs/promptir_baseline/epoch_020.pt
runs/promptir_baseline/metrics.csv
runs/promptir_baseline/loss_curve.png
```

By default, `latest.pt` is overwritten every epoch, `best.pt` is overwritten
when train L1 loss improves, and numbered checkpoints are saved every 10 epochs.
Change the interval with `--save-every`.

### 7. Generate `pred.npz`

Run inference with the trained checkpoint:

```bash
python infer.py \
  --dataset-root hw4_realse_dataset \
  --checkpoint runs/promptir_baseline/best.pt \
  --output pred.npz \
  --device cuda
```

If inference runs out of memory, use tiled inference:

```bash
python infer.py \
  --dataset-root hw4_realse_dataset \
  --checkpoint runs/promptir_baseline/best.pt \
  --output pred.npz \
  --device cuda \
  --tile-size 256 \
  --tile-overlap 32
```

Quickly verify the submission file:

```bash
python - <<'PY'
import numpy as np
data = np.load("pred.npz")
print(len(data.files))
print(data.files[:5])
first = data[data.files[0]]
print(first.shape, first.dtype)
PY
```

Expected output: `100` files, each array with shape `(3, H, W)` and dtype
`uint8`.

### 8. Download the result

From your local machine:

```bash
scp -P <PORT> root@<HOST>:/workspace/VRDL_PromptIR/pred.npz .
scp -P <PORT> root@<HOST>:/workspace/VRDL_PromptIR/runs/promptir_baseline/loss_curve.png .
scp -P <PORT> root@<HOST>:/workspace/VRDL_PromptIR/runs/promptir_baseline/metrics.csv .
```

## Train

From this directory:

```bash
python train.py \
  --dataset-root ../hw4_realse_dataset \
  --output-dir runs/promptir_baseline \
  --epochs 120 \
  --batch-size 8 \
  --patch-size 128
```

Use a smaller batch size if the GPU runs out of memory.

Training writes `metrics.csv` and `loss_curve.png` under the output directory.

## Inference And Submission

```bash
python infer.py \
  --dataset-root ../hw4_realse_dataset \
  --checkpoint runs/promptir_baseline/best.pt \
  --output pred.npz
```

If full-image inference runs out of memory, use tiling:

```bash
python infer.py \
  --dataset-root ../hw4_realse_dataset \
  --checkpoint runs/promptir_baseline/best.pt \
  --output pred.npz \
  --tile-size 256 \
  --tile-overlap 32
```

The generated `pred.npz` uses the required format: each key is the original test
filename, and each value is a restored `uint8` image array with shape `(3, H, W)`.
