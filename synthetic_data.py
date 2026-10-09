#!/usr/bin/env python3
"""Synthetic fixtures for tests and loader benchmarks. No real data is needed.

make_legacy_fixture  writes the legacy format (zlib input LMDB shard, manifest,
                     teacher LMDB, SEANet data list) so prepare_dataset.py can
                     be exercised end to end.
make_packed_fixture  writes a packed dataset directly with random contents;
                     useful for large throughput benchmarks.
"""

import csv
import io
import json
import zlib
from pathlib import Path

import lmdb
import numpy as np

from packed_dataset import PackedWriter


def _npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def make_legacy_fixture(root, num_utterances=40, teacher_dim=16, crop_size=88,
                        min_frames=20, max_frames=120, num_excluded=3, seed=0):
    """Return a dict of paths and the keys of utterances that the data list excludes."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    input_dir = root / "legacy_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir = root / "legacy_teacher.lmdb"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    data_list = root / "data_list.csv"

    keys = [f"lrs2/vid{idx:04d}/clip{idx % 7:02d}" for idx in range(num_utterances)]
    shard_name = "input_0000.lmdb"
    input_env = lmdb.open(str(input_dir / shard_name), subdir=False, map_size=1 << 36)
    teacher_env = lmdb.open(str(teacher_dir), subdir=True, map_size=1 << 36)
    manifest_rows = []
    lengths = {}
    with input_env.begin(write=True) as in_txn, teacher_env.begin(write=True) as t_txn:
        for key in keys:
            frames = int(rng.integers(min_frames, max_frames + 1))
            lengths[key] = frames
            mouth = rng.integers(0, 256, size=(frames, crop_size, crop_size), dtype=np.uint8)
            teacher = rng.standard_normal((frames, teacher_dim)).astype(np.float32) * 3.0
            in_txn.put(key.encode(), zlib.compress(_npy_bytes(mouth), level=1))
            t_txn.put(key.encode(), _npy_bytes(teacher))
            manifest_rows.append([key.split("/")[0], key, shard_name, frames, crop_size, crop_size, key.split("/", 1)[1] + ".npz"])
    input_env.close()
    teacher_env.close()

    (input_dir / "manifest.tsv").write_text(
        "\t".join(["dataset", "key", "shard", "frames", "height", "width", "source_relative"]) + "\n"
        + "\n".join("\t".join(str(v) for v in row) for row in manifest_rows) + "\n"
    )
    (input_dir / "metadata.json").write_text(json.dumps({
        "format": "numpy-npy-uint8", "compression": "zlib", "crop_size": crop_size,
        "frame_layout": "T,H,W", "dataset_roots": {"lrs2": "/nonexistent/lrs2"},
    }) + "\n")

    excluded = keys[-num_excluded:] if num_excluded else []
    with data_list.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for key in keys[: num_utterances - num_excluded]:
            vid, clip = key.split("/")[1:]
            writer.writerow(["train", "train", "id00001", vid, 0, "train", "id00002", clip, 0.1, 4.0])
        for key in excluded:
            _, vid, clip = key.split("/")
            writer.writerow(["test", "test", "lrs2", f"{vid}/{clip}", 0, "test", "x", "y", 0.1, 4.0])
    return {
        "input_dir": str(input_dir),
        "teacher_lmdb": str(teacher_dir),
        "data_list": str(data_list),
        "excluded": excluded,
        "lengths": lengths,
    }


def make_packed_fixture(root, num_utterances=2000, mean_frames=200, teacher_dim=1024,
                        val_fraction=0.05, shard_bytes=512 * 1024 ** 2, seed=0):
    """Write a random packed dataset (same format as prepare_dataset.py) for benchmarks."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    writer = PackedWriter(root, (88, 88), teacher_dim, shard_bytes)
    num_val = max(1, int(num_utterances * val_fraction))
    for idx in range(num_utterances):
        frames = int(np.clip(rng.normal(mean_frames, mean_frames / 4), 30, mean_frames * 3))
        mouth = rng.integers(0, 256, size=(frames, 88, 88), dtype=np.uint8)
        teacher = (rng.standard_normal((frames, teacher_dim)) * 3.0).astype(np.float16)
        split = "val" if idx < num_val else "train"
        writer.add(f"synthetic/{idx:07d}", split, mouth, teacher)
    return writer.close({
        "crop_size": 88, "pixel_mean": 0.421, "pixel_std": 0.165,
        "synthetic": True, "seed": seed,
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--num-utterances", type=int, default=2000)
    parser.add_argument("--mean-frames", type=int, default=200)
    parser.add_argument("--teacher-dim", type=int, default=1024)
    args = parser.parse_args()
    meta = make_packed_fixture(args.output_dir, args.num_utterances, args.mean_frames, args.teacher_dim)
    print(json.dumps({k: meta[k] for k in ("num_utterances", "num_frames", "split_counts")}, indent=2))
