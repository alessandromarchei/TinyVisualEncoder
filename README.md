# Visual frontend KD — Step 1

Goal: distil the original SEANet/AVSR visual frontend into a much smaller
frontend while keeping the same interface:

- input: `[T,B,1,112,112]`
- temporal receptive field in the first Conv3D: 5 frames
- output: `[T,B,512]`

The preprocessing intentionally matches the embedding-generation script:
OpenCV grayscale, resize 224, center crop 112, temporal sampling before
Conv3D, and `(x/255 - 0.4161)/0.1688`.

Place this directory in the SEANet repository root (or run from there), so
`pretrain_networks.visual_frontend` is importable.

## Train

```bash
python train_kd.py \
  --video-root /scratch_nvme/VoxCeleb2-2Mix/orig \
  --data-list configs/data_list.csv \
  --teacher pretrain_networks/visual_frontend.pt \
  --output runs/visual_kd_5f_512 \
  --fps 25 \
  --source-fps 25 \
  --frames 50 \
  --batch-size 32 \
  --workers 8 \
  --epochs 100 \
  --temporal-kernel 5 \
  --embedding-dim 512
```

If your `video-root` already points directly to `/orig/train`, the resolver
also supports that layout.

## Evaluate

```bash
python eval_kd.py \
  --video-root /scratch_nvme/VoxCeleb2-2Mix/orig \
  --data-list configs/data_list.csv \
  --split test \
  --teacher pretrain_networks/visual_frontend.pt \
  --student runs/visual_kd_5f_512/checkpoints/best.pt \
  --output runs/visual_kd_5f_512/eval \
  --fps 25
```

## Step 2

The model already supports `temporal_kernel=1`. For the actual Step-2
experiment the teacher remains 5-frame and the student becomes single-frame.
Do not simply run the current trainer with `--temporal-kernel 1` and call that
a complete single-frame experiment without documenting the temporal target:
the teacher target at t still contains context t-2..t+2, while the student
sees only t. That is intentional KD, but should be evaluated separately.

## Step 3

For PCA-128, fit PCA on TRAIN teacher embeddings only, save the PCA mean and
components, transform each teacher target as:

`z128 = (z512 - pca_mean) @ components[:128].T`

and instantiate the student with `embedding_dim=128`. No projection head is
needed. This should be a separate trainer/target mode to avoid accidentally
mixing raw 512-D teacher coordinates and PCA coordinates.

## AV-HuBERT Teacher

Training reads a packed dataset that is built once from the mouth-ROI input
LMDB shards and the AV-HuBERT teacher LMDB. Each utterance is one contiguous
record: uint8 mouth frames and float16 teacher embeddings, interleaved per
frame, so a 50-frame training window is a single byte range. See
[packed_dataset.py](packed_dataset.py) for the exact layout.

### 1. Convert once

```bash
python prepare_dataset.py \
  --input-dir /kaggle/input/mouth-input-lmdb \
  --teacher-lmdb /kaggle/input/avhubert-targets/data.lmdb \
  --data-list configs/data_list.csv \
  --output-dir /kaggle/working/avhubert_packed \
  --teacher-dim 1024 \
  --compare-legacy 64

python prepare_dataset.py --check --output-dir /kaggle/working/avhubert_packed
```

The conversion reuses the legacy split (`--val-utterances`, `--seed`) and the
SEANet val/test exclusion. `--compare-legacy N` checks N random samples against
the legacy loader: frames must match exactly, and the report includes the
float16 teacher error and the KD-loss change for a fixed student. Skipped
utterances are listed in `errors.tsv`. The packed format is uncompressed, so
its size can be larger than the zlib-compressed input; check the output
location's free space before starting.

### 2. Train

```bash
python train_kd_avhubert.py \
  --dataset-dir /kaggle/working/avhubert_packed \
  --output runs/avhubert_kd \
  --accelerator cuda \
  --batch-size 64 \
  --workers 4 \
  --prefetch-factor 2
```

Pixel mean/std and the teacher dimension come from the packed `meta.json`, so
training cannot drift from how the data was packed. Normalization happens on the
GPU. `--gpus N` uses DistributedDataParallel; batch size and workers are per GPU.

On a Kaggle TPU v5e-8 runtime with a compatible PyTorch/XLA installation:

```bash
python train_kd_avhubert.py \
  --dataset-dir /kaggle/working/avhubert_packed \
  --output runs/avhubert_kd_tpu \
  --accelerator tpu \
  --tpu-cores 8
```

### 3. Benchmark the input pipeline

```bash
# Packed dataset, random order, with GPU step timing
python benchmark_dataloader.py --dataset-dir /kaggle/working/avhubert_packed \
  --batch-size 64 --workers 4 --batches 200 --train-steps --cache-state cold

# Original pipeline on the same machine, for comparison
python benchmark_dataloader.py --legacy-input-dir /kaggle/input/mouth-input-lmdb \
  --legacy-teacher-lmdb /kaggle/input/avhubert-targets/data.lmdb \
  --legacy-data-list configs/data_list.csv \
  --batch-size 64 --workers 4 --batches 200 --train-steps --cache-state cold
```

Run each benchmark once with a cold page cache (`--drop-caches` if passwordless
sudo is available) and once warm. Peak USS, not RSS, reflects Python memory,
because RSS also counts memory-mapped file pages.

### Tests

```bash
python -m pytest -q test_packed_dataset.py
```

The tests build a small legacy fixture from synthetic data, convert it, and check
the packed reader against the legacy loader. They need no Kaggle data.

Training logs loss, Huber, MSE, and cosine similarity to stdout, `history.json`,
and W&B (unless `--no-wandb` is set), and saves `last.pt` and `best.pt` checkpoints.
