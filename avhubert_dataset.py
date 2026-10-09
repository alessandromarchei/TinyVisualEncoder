#!/usr/bin/env python3
"""LMDB dataset for distilling the AV-HuBERT visual encoder."""

import csv
import io
import json
import random
import zlib
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


_DATASET_NAMES = {"lrs2", "lrs3", "voxceleb2"}


def canonical_key(key):
    """Normalize input manifest IDs and absolute mouth-file paths."""
    value = key.decode("utf-8") if isinstance(key, bytes) else str(key)
    value = value.replace("\\", "/")

    if "/mouths/" in value:
        value = value.rsplit("/mouths/", 1)[1]

    parts = [part for part in value.split("/") if part and part != "."]
    if parts and parts[0] in _DATASET_NAMES:
        parts = parts[1:]

    normalized = "/".join(parts)
    for suffix in (".npz", ".npy"):
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
            break
    return normalized.lstrip("/")


def _decode_input(payload, compression):
    if compression == "zlib":
        payload = zlib.decompress(payload)
    return np.load(io.BytesIO(payload), allow_pickle=False)


def _decode_target(payload):
    try:
        target = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError):
        try:
            target = torch.load(
                io.BytesIO(payload),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            target = torch.load(io.BytesIO(payload), map_location="cpu")

    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    return np.asarray(target)


def _resolve_lmdb_path(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"LMDB not found: {path}")
    if path.is_dir() and not (path / "data.mdb").is_file():
        nested = path / "data.lmdb"
        if nested.is_file() or (nested / "data.mdb").is_file():
            path = nested
        else:
            candidates = sorted(
                child for child in path.iterdir()
                if child.is_dir() and (child / "data.mdb").is_file()
            )
            if len(candidates) == 1:
                path = candidates[0]
            elif candidates:
                choices = ", ".join(str(candidate) for candidate in candidates)
                raise ValueError(
                    f"Multiple LMDB directories found under {path}: {choices}. "
                    "Pass the intended directory with --teacher-lmdb."
                )
            else:
                raise FileNotFoundError(
                    f"{path} is a directory, but it is not an LMDB environment "
                    "(missing data.mdb). Pass the LMDB directory, its parent "
                    "containing one LMDB child, or data.lmdb inside the dataset."
                )
    return path


def _open_lmdb(path):
    path = _resolve_lmdb_path(path)
    return lmdb.open(
        str(path),
        subdir=path.is_dir(),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )


def _candidate_teacher_keys(row, dataset_roots):
    normalized = canonical_key(row["key"])
    dataset = row.get("dataset", "")
    source_relative = row.get("source_relative", "")
    candidates = [row["key"], normalized, f"{normalized}.npz"]

    root = dataset_roots.get(dataset)
    if root and source_relative:
        candidates.insert(
            0,
            str((Path(root).expanduser() / source_relative).resolve()),
        )
    return list(dict.fromkeys(candidates))


def _teacher_key_index(path, manifest_rows, dataset_roots):
    path = _resolve_lmdb_path(path)
    print(f"Opening teacher LMDB: {path}", flush=True)
    env = _open_lmdb(path)
    index = {}
    print(
        f"Looking up teacher targets for {len(manifest_rows):,} input rows...",
        flush=True,
    )
    try:
        with env.begin(write=False) as txn:
            for row in tqdm(
                manifest_rows,
                desc="Matching AV-HuBERT targets",
                unit="rows",
                dynamic_ncols=True,
            ):
                normalized = canonical_key(row["key"])
                for key in _candidate_teacher_keys(row, dataset_roots):
                    if txn.get(key.encode("utf-8")) is not None:
                        index[normalized] = key
                        break
    finally:
        env.close()
    print(f"Matched {len(index):,} teacher keys to input manifest.", flush=True)
    return index


def build_avhubert_samples(
    input_dir,
    teacher_lmdb,
    data_list,
    val_utterances=5000,
    max_train_utterances=None,
    seed=42,
):
    """Join input-shard manifest rows to a single AV-HuBERT target LMDB."""
    input_dir = Path(input_dir).expanduser().resolve()
    manifest_path = input_dir / "manifest.tsv"
    metadata_path = input_dir / "metadata.json"
    data_list = Path(data_list).expanduser().resolve()

    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected manifest.tsv and metadata.json in {input_dir}"
        )
    if not data_list.is_file():
        raise FileNotFoundError(f"SEANet data list not found: {data_list}")
    if val_utterances < 1:
        raise ValueError("--val-utterances must be >= 1")

    metadata = json.loads(metadata_path.read_text())
    compression = metadata.get("compression", "none")
    if compression not in {"none", "zlib"}:
        raise ValueError(f"Unsupported input compression: {compression}")

    excluded = set()
    with data_list.open("r", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) >= 4 and row[0].strip() in {"val", "test"}:
                excluded.add(canonical_key(f"{row[2].strip()}/{row[3].strip()}"))

    with manifest_path.open("r", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        required_columns = {"key", "shard"}
        if not rows.fieldnames or not required_columns.issubset(rows.fieldnames):
            raise ValueError("Input manifest must contain key and shard columns")
        manifest_rows = list(rows)

    print(
        f"Loaded {len(manifest_rows):,} input manifest rows; "
        "matching AV-HuBERT targets...",
        flush=True,
    )
    teacher_index = _teacher_key_index(
        teacher_lmdb,
        manifest_rows,
        metadata.get("dataset_roots", {}),
    )
    samples = []
    missing_targets = 0

    for row in manifest_rows:
        key = row["key"]
        normalized = canonical_key(key)
        target_key = teacher_index.get(normalized)
        if target_key is None:
            missing_targets += 1
            continue
        if normalized in excluded:
            continue
        samples.append({
            "key": key,
            "canonical_key": normalized,
            "shard": row["shard"],
            "target_key": target_key,
        })

    if not samples:
        raise RuntimeError(
            "No input/teacher pairs were found. Check that teacher LMDB keys "
            "refer to the same utterances as input manifest.tsv."
        )

    rng = random.Random(seed)
    rng.shuffle(samples)
    if val_utterances >= len(samples):
        raise ValueError(
            f"--val-utterances={val_utterances}, but only {len(samples)} "
            "paired utterances remain after filtering."
        )

    val_samples = samples[:val_utterances]
    train_pool = samples[val_utterances:]
    if max_train_utterances is not None:
        if max_train_utterances <= 0:
            raise ValueError("--max-train-utterances must be > 0")
        train_pool = train_pool[:max_train_utterances]

    print("AV-HuBERT LMDB dataset")
    print("----------------------")
    print(f"Input LMDB directory : {input_dir}")
    print(f"Teacher LMDB         : {Path(teacher_lmdb).expanduser().resolve()}")
    print(f"Paired teacher keys  : {len(teacher_index):,}")
    print(f"Unpaired input rows  : {missing_targets:,}")
    print(f"SEANet val/test IDs  : {len(excluded):,}")
    print(f"KD train             : {len(train_pool):,}")
    print(f"KD validation        : {len(val_samples):,}")
    print(f"Split seed           : {seed}")
    return train_pool, val_samples, metadata


class AVHubertLMDBDataset(Dataset):
    """Read sharded uint8 mouth inputs and teacher embeddings lazily."""

    def __init__(
        self,
        samples,
        input_dir,
        teacher_lmdb,
        compression="none",
        pixel_mean=0.421,
        pixel_std=0.165,
        frames_per_sample=50,
        random_crop=True,
    ):
        self.samples = list(samples)
        self.input_dir = Path(input_dir)
        self.teacher_lmdb = Path(teacher_lmdb)
        self.compression = compression
        self.pixel_mean = float(pixel_mean)
        self.pixel_std = float(pixel_std)
        self.frames_per_sample = int(frames_per_sample)
        self.random_crop = random_crop
        self.input_envs = {}
        self.teacher_env = None

    def __len__(self):
        return len(self.samples)

    def _get_input_env(self, shard):
        if shard not in self.input_envs:
            self.input_envs[shard] = _open_lmdb(self.input_dir / shard)
        return self.input_envs[shard]

    def _get_teacher_env(self):
        if self.teacher_env is None:
            self.teacher_env = _open_lmdb(self.teacher_lmdb)
        return self.teacher_env

    def __getitem__(self, index):
        sample = self.samples[index]
        with self._get_input_env(sample["shard"]).begin(write=False) as txn:
            input_payload = txn.get(sample["key"].encode("utf-8"))
        if input_payload is None:
            raise KeyError(f"Input LMDB key missing: {sample['key']}")

        with self._get_teacher_env().begin(write=False) as txn:
            target_payload = txn.get(sample["target_key"].encode("utf-8"))
        if target_payload is None:
            raise KeyError(f"Teacher LMDB key missing: {sample['target_key']}")

        frames = _decode_input(input_payload, self.compression)
        target = _decode_target(target_payload)
        if frames.ndim != 3:
            raise ValueError(f"Expected input [T,H,W], got {frames.shape}: {sample['key']}")
        if target.ndim != 2:
            raise ValueError(
                f"Expected AV-HuBERT embedding [T,D], got {target.shape}: "
                f"{sample['target_key']}"
            )
        if len(frames) != len(target):
            raise ValueError(
                f"Input/teacher frame mismatch for {sample['key']}: "
                f"{len(frames)} input frames vs {len(target)} embeddings"
            )
        if len(frames) == 0:
            raise ValueError(f"Empty utterance: {sample['key']}")

        count = self.frames_per_sample
        if len(frames) < count:
            pad = count - len(frames)
            frames = np.pad(frames, ((0, pad), (0, 0), (0, 0)), mode="edge")
            target = np.pad(target, ((0, pad), (0, 0)), mode="edge")
            start = 0
        elif len(frames) == count:
            start = 0
        elif self.random_crop:
            start = random.randint(0, len(frames) - count)
        else:
            start = (len(frames) - count) // 2

        frames = np.asarray(frames[start:start + count], dtype=np.float32) / 255.0
        target = np.asarray(target[start:start + count], dtype=np.float32)
        frames = (frames - self.pixel_mean) / self.pixel_std
        return torch.from_numpy(frames), torch.from_numpy(target), sample["key"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["input_envs"] = {}
        state["teacher_env"] = None
        return state

    def close(self):
        for env in self.input_envs.values():
            env.close()
        self.input_envs.clear()
        if self.teacher_env is not None:
            self.teacher_env.close()
            self.teacher_env = None