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
