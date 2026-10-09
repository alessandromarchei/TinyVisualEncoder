#!/usr/bin/env python3
"""Packed AV-HuBERT distillation dataset: one contiguous record per utterance.

Directory layout (written by prepare_dataset.py):

    meta.json        format description, normalization, split sizes, sources
    index.npz       per-utterance arrays: keys, split, shard, offset, length
    shard_0000.bin  raw records, concatenated; shards are ~--shard-size-gb each

An utterance with T frames is stored as T interleaved rows of ROW_BYTES:

    row t = [mouth uint8 H*W bytes (row-major) | teacher float16 D values (LE)]

Frames are contiguous, so a temporal crop of F frames is ONE contiguous byte
range. A sample therefore costs one memmap slice and one copy: no zlib, no
np.load, no dtype conversion and no normalization in the DataLoader workers.
Pixel normalization ((x / 255 - mean) / std) runs on the GPU in normalize_frames.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


FORMAT = "avhubert-packed"
FORMAT_VERSION = 1
SPLIT_IDS = {"train": 0, "val": 1}
SPLIT_NAMES = {value: key for key, value in SPLIT_IDS.items()}


def row_bytes(mouth_shape, teacher_dim):
    return int(np.prod(mouth_shape)) + 2 * int(teacher_dim)


def load_meta(root):
    meta = json.loads((Path(root) / "meta.json").read_text())
    if meta.get("format") != FORMAT or meta.get("version") != FORMAT_VERSION:
        raise ValueError(f"Not a {FORMAT} v{FORMAT_VERSION} dataset: {root}")
    return meta


def normalize_frames(frames, mean, std):
    """uint8 [..., H, W] -> float32 ((x / 255) - mean) / std, in place on the result.

    Same operation order as the legacy loader, so outputs match it exactly.
    """
    return frames.float().div_(255.0).sub_(mean).div_(std)


class PackedWriter:
    """Append utterances to fixed-size shard files and keep the index in memory."""

    def __init__(self, root, mouth_shape, teacher_dim, shard_bytes):
        self.root = Path(root)
        self.mouth_shape = tuple(int(v) for v in mouth_shape)
        self.mouth_bytes = int(np.prod(self.mouth_shape))
        self.teacher_dim = int(teacher_dim)
        self.row_bytes = row_bytes(self.mouth_shape, self.teacher_dim)
        self.shard_bytes = int(shard_bytes)
        self.shard_id = -1
        self.shard_offset = 0
        self.handle = None
        self.index = {"keys": [], "split": [], "shard": [], "offset": [], "length": []}
        self._open_next_shard()

    def _open_next_shard(self):
        if self.handle is not None:
            self.handle.close()
        self.shard_id += 1
        self.shard_offset = 0
        self.handle = open(self.root / shard_name(self.shard_id), "wb")

    def add(self, key, split, mouth, teacher):
        """mouth: uint8 [T,H,W]; teacher: float16 [T,D] (already validated)."""
        length = mouth.shape[0]
        block = np.empty((length, self.row_bytes), dtype=np.uint8)
        block[:, :self.mouth_bytes] = mouth.reshape(length, self.mouth_bytes)
        block[:, self.mouth_bytes:] = np.ascontiguousarray(
            teacher, dtype="<f2"
        ).reshape(length, -1).view(np.uint8)

        size = block.nbytes
        if self.shard_offset and self.shard_offset + size > self.shard_bytes:
            self._open_next_shard()
        self.index["keys"].append(key)
        self.index["split"].append(SPLIT_IDS[split])
        self.index["shard"].append(self.shard_id)
        self.index["offset"].append(self.shard_offset)
        self.index["length"].append(length)
        self.handle.write(block)
        self.shard_offset += size

    def close(self, extra_meta):
        self.handle.close()
        index = self.index
        np.savez(
            self.root / "index.npz",
            keys=np.array(index["keys"], dtype="S"),  # ASCII bytes: 1 byte per character
            split=np.array(index["split"], dtype=np.uint8),
            shard=np.array(index["shard"], dtype=np.uint32),
            offset=np.array(index["offset"], dtype=np.int64),
            length=np.array(index["length"], dtype=np.int32),
        )
        split_counts = {
            name: int(sum(1 for s in index["split"] if s == value))
            for name, value in SPLIT_IDS.items()
        }
        meta = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "mouth_shape": list(self.mouth_shape),
            "mouth_dtype": "uint8",
            "teacher_dim": self.teacher_dim,
            "teacher_dtype": "float16 (little-endian)",
            "row_bytes": self.row_bytes,
            "frame_layout": "per frame: mouth bytes, then teacher bytes",
            "num_utterances": len(index["keys"]),
            "num_frames": int(sum(index["length"])),
            "split_counts": split_counts,
            "shards": [
                {"file": shard_name(i), "bytes": os.path.getsize(self.root / shard_name(i))}
                for i in range(self.shard_id + 1)
            ],
            **extra_meta,
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta


def shard_name(shard_id):
    return f"shard_{shard_id:04d}.bin"


class PackedAVHubertDataset(Dataset):
    """Reads one split of a packed dataset. Opens shard memmaps lazily per process."""

    def __init__(self, root, split, frames=50, random_crop=True):
        if split not in SPLIT_IDS:
            raise ValueError(f"split must be one of {sorted(SPLIT_IDS)}")
        self.root = Path(root).expanduser().resolve()
        self.meta = load_meta(self.root)
        self.frames = int(frames)
        self.random_crop = random_crop
        self.mouth_shape = tuple(self.meta["mouth_shape"])
        self.teacher_dim = int(self.meta["teacher_dim"])
        self.row_bytes = int(self.meta["row_bytes"])
        self.mouth_bytes = int(np.prod(self.mouth_shape))
        if self.row_bytes != row_bytes(self.mouth_shape, self.teacher_dim):
            raise ValueError("meta.json row_bytes does not match its shapes")

        with np.load(self.root / "index.npz", allow_pickle=False) as index:
            split_ids = index["split"]
            self.shard = index["shard"]
            self.offset = index["offset"]
            self.length = index["length"]
            self.keys = index["keys"]
        self.positions = np.flatnonzero(split_ids == SPLIT_IDS[split])
        self._maps = {}

    def __len__(self):
        return len(self.positions)

    def key(self, position):
        return self.keys[self.positions[position]].decode("utf-8")

    def _memmap(self, shard_id):
        if shard_id not in self._maps:
            path = self.root / shard_name(shard_id)
            self._maps[shard_id] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._maps[shard_id]

    def __getitem__(self, position):
        index = self.positions[position]
        length = int(self.length[index])
        base = int(self.offset[index])
        data = self._memmap(int(self.shard[index]))
        row = self.row_bytes

        if length >= self.frames:
            if self.random_crop:
                start = random.randint(0, length - self.frames)
            else:
                start = (length - self.frames) // 2
            block = data[base + start * row: base + (start + self.frames) * row]
            block = block.reshape(self.frames, row)
        else:
            # Short utterances: repeat the last frame, like the legacy edge padding.
            clip = np.minimum(np.arange(self.frames), length - 1)
            block = data[base: base + length * row].reshape(length, row)[clip]

        mouth = torch.from_numpy(np.ascontiguousarray(block[:, :self.mouth_bytes]))
        teacher = torch.from_numpy(np.ascontiguousarray(block[:, self.mouth_bytes:]))
        return (
            mouth.view(self.frames, *self.mouth_shape),
            teacher.view(torch.float16).view(self.frames, self.teacher_dim),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state
