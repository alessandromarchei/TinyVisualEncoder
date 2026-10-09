#!/usr/bin/env python3
"""Convert the legacy input LMDB shards + teacher LMDB into a packed dataset.

Run once (see README). Training then reads only the packed directory; see
packed_dataset.py for the on-disk format.

The train/validation split is produced by build_avhubert_samples, the same
function the legacy trainer used, so the split and the SEANet val/test
exclusion are unchanged.

Teacher embeddings are stored as float16. Quantization error is measured on
every converted sample and written to meta.json; use --compare-legacy to also
check KD loss against the original float32 targets.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from avhubert_dataset import (
    AVHubertLMDBDataset,
    build_avhubert_samples,
    decode_input,
    decode_target,
    _open_lmdb,
)
from losses import kd_loss
from models.tiny_visual_frontend import TinyVisualFrontend
from packed_dataset import (
    PackedAVHubertDataset,
    PackedWriter,
    load_meta,
    normalize_frames,
)


FP16_MAX = float(np.finfo(np.float16).max)


def load_pair(input_env, teacher_txn, sample, compression, mouth_shape, teacher_dim):
    """Return (uint8 frames [T,H,W], float32 teacher [T,D]) or raise ValueError."""
    with input_env.begin(write=False) as txn:
        payload = txn.get(sample["key"].encode("utf-8"))
    if payload is None:
        raise ValueError("input key missing from shard")
    target_payload = teacher_txn.get(sample["target_key"].encode("utf-8"))
    if target_payload is None:
        raise ValueError("teacher key missing from teacher LMDB")
    frames = decode_input(payload, compression)
    target = decode_target(target_payload)

    if frames.ndim != 3 or frames.shape[1:] != tuple(mouth_shape):
        raise ValueError(f"input shape {frames.shape}, expected [T,{mouth_shape}]")
    if frames.dtype != np.uint8:
        raise ValueError(f"input dtype {frames.dtype}, expected uint8")
    if target.ndim != 2 or target.shape[1] != teacher_dim:
        raise ValueError(f"teacher shape {target.shape}, expected [T,{teacher_dim}]")
    if len(frames) == 0 or len(frames) != len(target):
        raise ValueError(f"frame mismatch: {len(frames)} inputs vs {len(target)} teacher")
    if not np.isfinite(target).all():
        raise ValueError("non-finite teacher values")
    if np.abs(target).max() > FP16_MAX:
        raise ValueError("teacher values exceed float16 range")
    return frames, target


class QuantizationStats:
    """Running error of float16 teacher storage relative to the float32 source."""

    def __init__(self):
        self.max_abs_err = 0.0
        self.max_abs_value = 0.0
        self.sq_err = 0.0
        self.sq_ref = 0.0

    def update(self, target32, target16):
        ref = target32.astype(np.float64)
        err = ref - target16.astype(np.float64)
        self.max_abs_err = max(self.max_abs_err, float(np.abs(err).max()))
        self.max_abs_value = max(self.max_abs_value, float(np.abs(ref).max()))
        self.sq_err += float((err ** 2).sum())
        self.sq_ref += float((ref ** 2).sum())

    def as_dict(self):
        rel_rms = (self.sq_err / max(self.sq_ref, 1e-30)) ** 0.5
        return {
            "max_abs_err": self.max_abs_err,
            "max_abs_value": self.max_abs_value,
            "relative_rms_err": rel_rms,
        }


def convert(args):
    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    train_pool, val_samples, legacy_meta = build_avhubert_samples(
        input_dir=args.input_dir,
        teacher_lmdb=args.teacher_lmdb,
        data_list=args.data_list,
        val_utterances=args.val_utterances,
        max_train_utterances=args.max_train_utterances,
        seed=args.seed,
    )
    compression = legacy_meta.get("compression", "none")
    mouth_shape = (legacy_meta["crop_size"], legacy_meta["crop_size"])
    jobs = [(sample, "train") for sample in train_pool]
    jobs += [(sample, "val") for sample in val_samples]
    # Sorting by shard reads each input LMDB sequentially.
    jobs.sort(key=lambda job: (job[0]["shard"], job[0]["key"]))

    writer = PackedWriter(
        out, mouth_shape, args.teacher_dim, int(args.shard_size_gb * 1024 ** 3)
    )
    quant = QuantizationStats()
    errors = []
    input_env = None
    input_shard = None
    teacher_env = _open_lmdb(args.teacher_lmdb)
    try:
        with teacher_env.begin(write=False) as teacher_txn:
            for sample, split in tqdm(jobs, desc="Packing", unit="utt", dynamic_ncols=True):
                if sample["shard"] != input_shard:
                    if input_env is not None:
                        input_env.close()
                    input_env = _open_lmdb(Path(args.input_dir) / sample["shard"])
                    input_shard = sample["shard"]
                try:
                    frames, target32 = load_pair(
                        input_env, teacher_txn, sample, compression,
                        mouth_shape, args.teacher_dim,
                    )
                except Exception as exc:  # one bad utterance must not stop a long conversion
                    errors.append((sample["key"], str(exc)))
                    continue
                target16 = target32.astype(np.float16)
                quant.update(target32, target16)
                writer.add(sample["key"], split, frames, target16)
    finally:
        if input_env is not None:
            input_env.close()
        teacher_env.close()

    if errors:
        with (out / "errors.tsv").open("w", newline="") as stream:
            csv.writer(stream, delimiter="\t").writerows([("key", "error"), *errors])
    if not writer.index["keys"]:
        raise RuntimeError("No utterances were packed; see the errors above.")

    meta = writer.close({
        "crop_size": legacy_meta["crop_size"],
        "pixel_mean": args.pixel_mean,
        "pixel_std": args.pixel_std,
        "teacher_quantization": quant.as_dict(),
        "skipped_utterances": len(errors),
        "split_seed": args.seed,
        "val_utterances": args.val_utterances,
        "max_train_utterances": args.max_train_utterances,
        "input_dir": str(Path(args.input_dir).expanduser().resolve()),
        "teacher_lmdb": str(Path(args.teacher_lmdb).expanduser().resolve()),
        "input_compression": compression,
        "dataset_roots": legacy_meta.get("dataset_roots", {}),
    })
    print(f"Packed dataset written to {out}")
    print(f"  utterances : {meta['num_utterances']:,} {meta['split_counts']}")
    print(f"  frames     : {meta['num_frames']:,}")
    print(f"  size       : {sum(s['bytes'] for s in meta['shards']) / 1024 ** 3:.2f} GiB")
    print(f"  skipped    : {len(errors):,} (see errors.tsv)")
    print(f"  fp16 error : {meta['teacher_quantization']}")

    if args.compare_legacy > 0:
        report = compare_with_legacy(
            out, args.compare_legacy, jobs, args, compression, mouth_shape,
            student_checkpoint=args.student_checkpoint,
        )
        print("Legacy comparison:")
        for name, value in report.items():
            print(f"  {name}: {value}")


def compare_with_legacy(root, num_samples, jobs, args, compression, mouth_shape,
                        student_checkpoint=None):
    """Compare packed samples against the legacy loader on a random subset.

    Frames must match exactly after normalization; teacher error is the float16
    rounding; KD loss is evaluated with one fixed student on both targets.
    """
    meta = load_meta(root)
    mean, std = meta["pixel_mean"], meta["pixel_std"]
    chosen = random.Random(args.seed).sample(jobs, min(num_samples, len(jobs)))
    legacy = AVHubertLMDBDataset(
        samples=[sample for sample, _ in chosen],
        input_dir=args.input_dir,
        teacher_lmdb=args.teacher_lmdb,
        compression=compression,
        pixel_mean=mean,
        pixel_std=std,
        frames_per_sample=args.frames,
        random_crop=False,
    )
    datasets = {
        split: PackedAVHubertDataset(root, split, frames=args.frames, random_crop=False)
        for split in ("train", "val")
    }
    positions = {
        split: {dataset.key(p): p for p in range(len(dataset))}
        for split, dataset in datasets.items()
    }

    student = build_student(meta["teacher_dim"], student_checkpoint)
    max_frame_diff = max_target_diff = 0.0
    loss_diffs, loss_refs = [], []
    with torch.no_grad():
        for position, (sample, split) in enumerate(chosen):
            legacy_frames, legacy_target, _ = legacy[position]
            mouth, teacher = datasets[split][positions[split][sample["key"]]]
            frames = normalize_frames(mouth, mean, std)
            target = teacher.float()

            max_frame_diff = max(max_frame_diff, float((frames - legacy_frames).abs().max()))
            max_target_diff = max(max_target_diff, float((target - legacy_target).abs().max()))

            model_input = legacy_frames.unsqueeze(1).unsqueeze(2)  # [T,1,1,H,W]
            prediction = student(model_input)                      # [T,1,D]
            ref_loss, _ = kd_loss(prediction, legacy_target.unsqueeze(1),
                                  feature_mean=None, feature_std=None)
            new_loss, _ = kd_loss(prediction, target.unsqueeze(1),
                                  feature_mean=None, feature_std=None)
            loss_refs.append(float(ref_loss))
            loss_diffs.append(abs(float(new_loss) - float(ref_loss)))
    return {
        "samples": len(chosen),
        "max_frame_abs_diff": max_frame_diff,
        "max_teacher_abs_diff": max_target_diff,
        "kd_loss_mean_fp32": float(np.mean(loss_refs)),
        "kd_loss_max_abs_diff": float(np.max(loss_diffs)),
        "student": student_checkpoint or "random init (seed 0)",
    }


def build_student(teacher_dim, checkpoint_path=None):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        widths = tuple(int(v) for v in args["widths"].split(","))
        student = TinyVisualFrontend(
            embedding_dim=args["embedding_dim"],
            temporal_kernel=args["temporal_kernel"],
            stem_channels=args["stem_channels"],
            widths=widths,
            expand=args["expand"],
        )
        student.load_state_dict(checkpoint["model"], strict=True)
    else:
        torch.manual_seed(0)
        student = TinyVisualFrontend(embedding_dim=teacher_dim)
    return student.eval()


def check(args):
    """Structural and numeric checks on an existing packed directory."""
    root = Path(args.output_dir).expanduser().resolve()
    meta = load_meta(root)
    for split in ("train", "val"):
        dataset = PackedAVHubertDataset(root, split, frames=args.frames, random_crop=False)
        if len(dataset) == 0:
            raise RuntimeError(f"{split} split is empty")
        lengths = dataset.length[dataset.positions]
        if (lengths <= 0).any():
            raise RuntimeError(f"{split} split has empty utterances")
        generator = random.Random(args.seed)
        for position in generator.sample(range(len(dataset)), min(args.check_samples, len(dataset))):
            mouth, teacher = dataset[position]
            if not torch.isfinite(teacher.float()).all():
                raise RuntimeError(f"non-finite teacher in {split} sample {dataset.key(position)}")
            if mouth.shape != (args.frames, *meta["mouth_shape"]):
                raise RuntimeError(f"bad mouth shape {tuple(mouth.shape)}")
        print(f"{split}: {len(dataset):,} utterances, {int(lengths.sum()):,} frames OK")
    shard_sizes = {s["file"]: s["bytes"] for s in meta["shards"]}
    print(f"Shards: {len(shard_sizes)}; total {sum(shard_sizes.values()) / 1024 ** 3:.2f} GiB")
    print("Check passed.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Validate an existing packed dataset instead of converting")
    parser.add_argument("--input-dir", help="Legacy input directory (input_*.lmdb, manifest.tsv, metadata.json)")
    parser.add_argument("--teacher-lmdb", help="Legacy teacher LMDB with AV-HuBERT embeddings")
    parser.add_argument("--data-list", help="SEANet data_list.csv used to exclude val/test utterances")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-utterances", type=int, default=5000)
    parser.add_argument("--max-train-utterances", type=int, default=None)
    parser.add_argument("--teacher-dim", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=50, help="Window length used by --compare-legacy and --check")
    parser.add_argument("--pixel-mean", type=float, default=0.421)
    parser.add_argument("--pixel-std", type=float, default=0.165)
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare-legacy", type=int, default=0,
                        help="After converting, compare this many random samples with the legacy loader")
    parser.add_argument("--student-checkpoint", default=None,
                        help="Optional trained student for the KD-loss comparison (default: fixed random init)")
    parser.add_argument("--check-samples", type=int, default=64)
    args = parser.parse_args()
    if args.check:
        return args
    missing = [name for name in ("input_dir", "teacher_lmdb", "data_list") if not getattr(args, name)]
    if missing:
        parser.error("conversion requires " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if args.shard_size_gb <= 0 or args.teacher_dim <= 0:
        parser.error("--shard-size-gb and --teacher-dim must be positive")
    return args


def main():
    args = parse_args()
    if args.check:
        check(args)
    else:
        convert(args)


if __name__ == "__main__":
    main()
