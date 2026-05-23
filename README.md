# HW4 PromptIR Baseline
This is a clean baseline for the HW4 image restoration task. The model is the
original PromptIR architecture trained from scratch on the provided rain/snow
training pairs. No pretrained weights or external data are used.

## Source
- Paper: PromptIR: Prompting for All-in-One Image Restoration, NeurIPS 2023
- Official code: https://github.com/va1shn9v/PromptIR
- HW4 requirement: train a single PromptIR model for both Rain and Snow.

## Training Steps
### 1. Clone the code

```bash
git clone https://github.com/ABparadise33/VRDL_PromptIR.git
cd VRDL_PromptIR
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Check that PyTorch:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

### 3. Download the HW4 dataset

```bash
python -m pip install gdown
gdown 1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w -O hw4_realse_dataset.zip
unzip hw4_realse_dataset.zip
```

the expected layout is:

```text
VRDL_PromptIR/
  hw4_realse_dataset/
    train/
      degraded/
      clean/
    test/
      degraded/
```

### 4. Train from scratch

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

if memory is tight:

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

To run the PSNR-oriented loss experiment, keep a separate output directory:

```bash
python train.py \
  --dataset-root hw4_realse_dataset \
  --output-dir runs/promptir_l1_mse \
  --epochs 120 \
  --batch-size 4 \
  --patch-size 128 \
  --num-workers 4 \
  --device cuda \
  --loss-type l1_mse \
  --mse-weight 1.0
```

By default, 10% of the training pairs are held out for validation PSNR after
each epoch. Change this with `--val-ratio`; set `--val-ratio 0` to disable
validation.

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

Training writes the epoch loss log and the report-ready loss curve here:

```text
runs/promptir_baseline/latest.pt
runs/promptir_baseline/best.pt
runs/promptir_baseline/best_psnr.pt
runs/promptir_baseline/best_loss.pt
runs/promptir_baseline/epoch_010.pt
runs/promptir_baseline/epoch_020.pt
runs/promptir_baseline/metrics.csv
runs/promptir_baseline/loss_curve.png
runs/promptir_baseline/psnr_curve.png
```

By default, `latest.pt` is overwritten every epoch, `best.pt` is overwritten
when validation PSNR improves, `best_psnr.pt` stores the highest validation
PSNR checkpoint, `best_loss.pt` stores the lowest training loss checkpoint, and
numbered checkpoints are saved every 10 epochs. Change the interval with
`--save-every`.

`metrics.csv` records overall validation PSNR plus separate rain/snow validation
PSNR columns:

```text
epoch,train_loss,train_l1,train_mse,val_psnr,val_psnr_rain,val_psnr_snow,lr,steps
```

### 5. Generate `pred.npz`

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

The generated `pred.npz` uses the required format: each key is the original test
filename, and each value is a restored `uint8` image array with shape `(3, H, W)`.
