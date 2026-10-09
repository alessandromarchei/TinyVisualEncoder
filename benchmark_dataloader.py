#!/usr/bin/env python3
"""Benchmark the AV-HuBERT distillation input pipeline (and optionally training steps).

Measures, for each dataset variant (packed and/or legacy LMDB):

  * __getitem__ latency in one process (p50 / p95), no DataLoader involved;
  * DataLoader batches/s, samples/s and the time spent blocked in next();
  * process-tree RSS and USS peaks (USS = memory private to the process tree;
    RSS also counts shared/file-backed pages, including the page cache mapped
    by memmap, so USS is the number that reflects Python heap pressure);
  * with --train-steps, host-to-device copy, GPU step time (CUDA-synchronized)
    and peak CUDA allocation.

Cold vs warm cache: --drop-caches evicts the dataset files from the page cache
before each variant (global drop with passwordless sudo, otherwise per-file
posix_fadvise). Without it, label the run with --cache-state yourself.

Example (Kaggle, after prepare_dataset.py):
  python benchmark_dataloader.py --dataset-dir /kaggle/working/packed \
      --batch-size 64 --workers 4 --batches 200 --cache-state cold --train-steps
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from packed_dataset import PackedAVHubertDataset, load_meta, normalize_frames


class MemoryMonitor:
    """Samples RSS and USS of this process and its children on a background thread."""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_rss = 0
        self.peak_uss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process()

    def _snapshot(self):
        procs = [self._proc] + self._proc.children(recursive=True)
        rss = uss = 0
        for proc in procs:
            try:
                rss += proc.memory_info().rss
                uss += proc.memory_full_info().uss
            except (psutil.Error, OSError):
                continue
        return rss, uss

    def _run(self):
        while not self._stop.is_set():
            rss, uss = self._snapshot()
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_uss = max(self.peak_uss, uss)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        rss, uss = self._snapshot()
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_uss = max(self.peak_uss, uss)


def evict_page_cache(paths):
    """Make the next reads of these files come from storage.

    Tries a global drop (needs passwordless sudo). Otherwise evicts only the
    given files with posix_fadvise(DONTNEED), which needs no privileges.
    Returns the label to report for the cache state, or None if nothing was done.
    """
    try:
        subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       check=True, capture_output=True)
        return "cold (global drop)"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    evicted = 0
    for path in paths:
        for file in ([path] if Path(path).is_file() else sorted(Path(path).glob("*"))):
            if not file.is_file():
                continue
            fd = os.open(file, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                evicted += 1
            finally:
                os.close(fd)
    return f"cold (fadvise on {evicted} files)" if evicted else None


def getitem_latency(dataset, samples, seed):
    rng = random.Random(seed)
    times = []
    for position in rng.sample(range(len(dataset)), min(samples, len(dataset))):
        start = time.perf_counter()
        dataset[position]
        times.append((time.perf_counter() - start) * 1e3)
    times.sort()
    return {
        "p50_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * (len(times) - 1))],
        "mean_ms": statistics.fmean(times),
        "samples": len(times),
    }


def make_loader(dataset, args):
    sampler = (SequentialSampler(dataset) if args.order == "sequential"
               else RandomSampler(dataset))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )


def make_packed_dataset(args):
    return PackedAVHubertDataset(args.dataset_dir, "train", frames=args.frames,
                                 random_crop=True)


def make_legacy_dataset(args):
    from avhubert_dataset import AVHubertLMDBDataset, build_avhubert_samples
    train_pool, _, meta = build_avhubert_samples(
        input_dir=args.legacy_input_dir, teacher_lmdb=args.legacy_teacher_lmdb,
        data_list=args.legacy_data_list, val_utterances=args.legacy_val_utterances,
        max_train_utterances=None, seed=args.seed,
    )
    return AVHubertLMDBDataset(
        samples=train_pool,
        input_dir=args.legacy_input_dir,
        teacher_lmdb=args.legacy_teacher_lmdb,
        compression=meta.get("compression", "none"),
        pixel_mean=0.421,
        pixel_std=0.165,
        frames_per_sample=args.frames,
        random_crop=True,
    )


def build_train_step(args, device):
    from losses import kd_loss
    from models.tiny_visual_frontend import TinyVisualFrontend

    widths = tuple(int(v) for v in args.widths.split(","))
    model = TinyVisualFrontend(embedding_dim=args.teacher_dim, widths=widths).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def step(batch, packed):
        frames, target = batch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        frames = frames.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if packed:
            frames = normalize_frames(frames, 0.421, 0.165)
        target = target.permute(1, 0, 2).float()
        model_input = frames.permute(1, 0, 2, 3).unsqueeze(2)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model(model_input)
        loss, _ = kd_loss(prediction.float(), target, 1.0, 0.5)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    return step


def run_loader(name, dataset, args, packed, device):
    loader = make_loader(dataset, args)
    step = build_train_step(args, device) if args.train_steps and device.type == "cuda" else None
    if args.train_steps and step is None:
        print("--train-steps needs CUDA; measuring the loader only.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_batches = args.warmup + args.batches
    waits, transfers, computes = [], [], []
    samples = 0
    iterator = iter(loader)
    with MemoryMonitor() as monitor:
        start = None
        measured = 0
        for index in range(total_batches):
            t0 = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:  # epoch boundary: start a new pass over the data
                iterator = iter(loader)
                batch = next(iterator)
            t1 = time.perf_counter()
            if index < args.warmup:
                continue
            if start is None:
                start = t0
            if step is not None:
                transfer, compute = step(batch, packed)
                transfers.append(transfer)
                computes.append(compute)
            waits.append(t1 - t0)
            samples += batch[0].shape[0]
            measured += 1
        if start is None:
            raise RuntimeError(f"{name}: no batches measured; check --batches and --warmup")
        elapsed = time.perf_counter() - start
    del iterator

    result = {
        "variant": name,
        "batches": measured,
        "batches_per_s": measured / elapsed,
        "samples_per_s": samples / elapsed,
        "wait_fraction": sum(waits) / elapsed,
        "wait_ms_p50": statistics.median(waits) * 1e3,
        "wait_ms_p95": sorted(waits)[int(0.95 * (len(waits) - 1))] * 1e3,
        "peak_rss_gb": monitor.peak_rss / 1024 ** 3,
        "peak_uss_gb": monitor.peak_uss / 1024 ** 3,
    }
    if step is not None:
        result.update({
            "h2d_ms_mean": statistics.fmean(transfers) * 1e3,
            "gpu_step_ms_mean": statistics.fmean(computes) * 1e3,
            "peak_cuda_alloc_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
        })
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", help="Packed dataset directory")
    parser.add_argument("--legacy-input-dir", help="Legacy input LMDB directory (for the original pipeline)")
    parser.add_argument("--legacy-teacher-lmdb")
    parser.add_argument("--legacy-data-list")
    parser.add_argument("--legacy-val-utterances", type=int, default=5000)
    parser.add_argument("--make-synthetic", default=None,
                        help="Write a random packed dataset here first (no real data needed)")
    parser.add_argument("--synthetic-utterances", type=int, default=2000)
    parser.add_argument("--synthetic-mean-frames", type=int, default=200)
    parser.add_argument("--teacher-dim", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--order", choices=["random", "sequential"], default="random")
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--getitem-samples", type=int, default=300)
    parser.add_argument("--train-steps", action="store_true", help="Also time H2D copy and GPU steps")
    parser.add_argument("--widths", default="24,32,64,96")
    parser.add_argument("--cache-state", choices=["cold", "warm", "unknown"], default="unknown")
    parser.add_argument("--drop-caches", action="store_true",
                        help="Evict these dataset files from the page cache before measuring (see evict_page_cache)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if args.make_synthetic:
        from synthetic_data import make_packed_fixture
        make_packed_fixture(args.make_synthetic, args.synthetic_utterances,
                            args.synthetic_mean_frames, args.teacher_dim, seed=args.seed)
        args.dataset_dir = args.make_synthetic
    if not args.dataset_dir and not args.legacy_input_dir:
        parser.error("pass --dataset-dir and/or --legacy-input-dir")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"cache state: {args.cache_state} | device: {device} | batch {args.batch_size} | "
          f"workers {args.workers} | prefetch {args.prefetch_factor} | order {args.order}")
    results = []
    if args.dataset_dir:
        dataset = make_packed_dataset(args)
        if args.drop_caches:
            args.cache_state = evict_page_cache([args.dataset_dir]) or args.cache_state
        meta = load_meta(args.dataset_dir)
        print(f"packed: {len(dataset):,} utterances in {args.dataset_dir} "
              f"({sum(s['bytes'] for s in meta['shards']) / 1024 ** 3:.2f} GiB)")
        latency = getitem_latency(dataset, args.getitem_samples, args.seed)
        result = run_loader("packed", dataset, args, True, device)
        result["getitem"] = latency
        results.append(result)
    if args.legacy_input_dir:
        dataset = make_legacy_dataset(args)
        if args.drop_caches:
            state = evict_page_cache([args.legacy_input_dir, args.legacy_teacher_lmdb])
            args.cache_state = state or args.cache_state
        print(f"legacy: {len(dataset):,} utterances")
        latency = getitem_latency(dataset, args.getitem_samples, args.seed)
        result = run_loader("legacy", dataset, args, False, device)
        result["getitem"] = latency
        results.append(result)

    for result in results:
        print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"cache_state": args.cache_state, "args": vars(args), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
