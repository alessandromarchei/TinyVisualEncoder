#!/usr/bin/env python3
"""Correctness tests: packed dataset versus the legacy LMDB loader.

Run with:  python -m pytest -q test_packed_dataset.py
Uses synthetic data only (see synthetic_data.py).
"""

import argparse

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import prepare_dataset
from avhubert_dataset import decode_input, _open_lmdb
from packed_dataset import PackedAVHubertDataset, load_meta, normalize_frames
from synthetic_data import make_legacy_fixture


FRAMES = 50
TEACHER_DIM = 16


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    base = tmp_path_factory.mktemp("packed")
    fixture = make_legacy_fixture(base, num_utterances=40, teacher_dim=TEACHER_DIM,
                                  min_frames=20, max_frames=120, num_excluded=3)
    out = base / "packed"
    args = argparse.Namespace(
        input_dir=fixture["input_dir"],
        teacher_lmdb=fixture["teacher_lmdb"],
        data_list=fixture["data_list"],
        output_dir=str(out),
        val_utterances=5,
        max_train_utterances=None,
        teacher_dim=TEACHER_DIM,
        frames=FRAMES,
        pixel_mean=0.421,
        pixel_std=0.165,
        shard_size_gb=0.0005,  # ~500 KiB: forces several shard files
        seed=42,
        compare_legacy=0,
        student_checkpoint=None,
        check_samples=64,
    )
    prepare_dataset.convert(args)
    return {"args": args, "fixture": fixture, "out": out}


def _legacy_samples(fixture):
    """Rebuild the legacy sample list with the same split the converter used."""
    from avhubert_dataset import build_avhubert_samples
    train_pool, val_samples, meta = build_avhubert_samples(
        input_dir=fixture["input_dir"], teacher_lmdb=fixture["teacher_lmdb"],
        data_list=fixture["data_list"], val_utterances=5,
        max_train_utterances=None, seed=42,
    )
    jobs = [(s, "train") for s in train_pool] + [(s, "val") for s in val_samples]
    return jobs, meta


def test_split_and_exclusions(converted):
    meta = load_meta(converted["out"])
    fixture = converted["fixture"]
    assert meta["split_counts"]["val"] == 5
    assert meta["num_utterances"] == 40 - len(fixture["excluded"])
    # Excluded SEANet test utterances must not be packed.
    train = PackedAVHubertDataset(converted["out"], "train")
    val = PackedAVHubertDataset(converted["out"], "val")
    packed_keys = {train.key(p) for p in range(len(train))} | {val.key(p) for p in range(len(val))}
    assert not packed_keys & set(fixture["excluded"])
    assert meta["teacher_quantization"]["relative_rms_err"] < 1e-3
    assert len(meta["shards"]) > 1, "fixture should exercise shard rotation"


def test_frames_and_teacher_match_legacy(converted):
    """Every packed utterance equals the legacy loader's centre crop."""
    jobs, legacy_meta = _legacy_samples(converted["fixture"])
    report = prepare_dataset.compare_with_legacy(
        converted["out"], len(jobs), jobs, converted["args"],
        legacy_meta.get("compression", "none"), (88, 88),
    )
    assert report["samples"] == len(jobs)
    # Frames are stored exactly, so normalized frames must match to float rounding.
    assert report["max_frame_abs_diff"] < 1e-6
    # Teacher goes through float16: bounded by half-precision rounding.
    assert report["max_teacher_abs_diff"] < 2e-2
    # KD loss is essentially unchanged by the float16 targets.
    assert report["kd_loss_max_abs_diff"] <= 1e-2 * max(1.0, report["kd_loss_mean_fp32"])


def test_random_crop_is_a_contiguous_window_of_the_source(converted):
    fixture = converted["fixture"]
    meta = load_meta(converted["out"])
    dataset = PackedAVHubertDataset(converted["out"], "train", frames=FRAMES, random_crop=True)
    in_env = None
    for position in range(min(10, len(dataset))):
        key = dataset.key(position)
        mouth, teacher = dataset[position]
        shard = "input_0000.lmdb"
        in_env = _open_lmdb(f"{fixture['input_dir']}/{shard}")
        with in_env.begin() as txn:
            raw = decode_input(txn.get(key.encode()), "zlib")
        in_env.close()
        length = len(raw)
        if length < FRAMES:
            continue
        starts = [s for s in range(length - FRAMES + 1)
                  if np.array_equal(raw[s:s + FRAMES], mouth.numpy())]
        assert starts, f"window for {key} is not a contiguous slice of its source"
        assert meta["teacher_dim"] == TEACHER_DIM


def test_short_utterances_repeat_the_last_frame(converted):
    fixture = converted["fixture"]
    dataset = PackedAVHubertDataset(converted["out"], "train", frames=FRAMES, random_crop=False)
    short = [dataset.key(p) for p in range(len(dataset))
             if dataset.length[dataset.positions[p]] < FRAMES]
    assert short, "fixture should contain utterances shorter than the window"
    key = short[0]
    in_env = _open_lmdb(f"{fixture['input_dir']}/input_0000.lmdb")
    with in_env.begin() as txn:
        raw = decode_input(txn.get(key.encode()), "zlib")
    in_env.close()
    position = next(p for p in range(len(dataset)) if dataset.key(p) == key)
    mouth, _ = dataset[position]
    assert mouth.shape[0] == FRAMES
    assert np.array_equal(mouth.numpy()[: len(raw)], raw)
    assert np.array_equal(mouth.numpy()[len(raw):], np.repeat(raw[-1:], FRAMES - len(raw), axis=0))


def test_dataloader_batches_stay_compact(converted):
    dataset = PackedAVHubertDataset(converted["out"], "train", frames=FRAMES, random_crop=True)
    loader = DataLoader(dataset, batch_size=8, num_workers=2, shuffle=True)
    mouth, teacher = next(iter(loader))
    assert mouth.dtype == torch.uint8 and mouth.shape == (8, FRAMES, 88, 88)
    assert teacher.dtype == torch.float16 and teacher.shape == (8, FRAMES, TEACHER_DIM)
    normalized = normalize_frames(mouth, 0.421, 0.165)
    assert normalized.dtype == torch.float32
    assert abs(float(normalized.mean())) < 3.0


def test_check_command_passes(converted):
    args = argparse.Namespace(output_dir=str(converted["out"]), frames=FRAMES,
                              seed=0, check_samples=16)
    prepare_dataset.check(args)


def test_packed_format_rejects_foreign_directory(tmp_path):
    (tmp_path / "meta.json").write_text('{"format": "other", "version": 1}')
    with pytest.raises(ValueError):
        load_meta(tmp_path)
