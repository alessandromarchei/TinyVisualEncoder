#!/usr/bin/env python3
"""Distill TinyVisualFrontend from precomputed AV-HuBERT embeddings."""

import argparse
import json
import os
import random
import socket
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from losses import kd_loss
from models.tiny_visual_frontend import TinyVisualFrontend, count_parameters
from packed_dataset import PackedAVHubertDataset, load_meta, normalize_frames


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True,
                        help="Packed dataset written by prepare_dataset.py (meta.json, index.npz, shard_*.bin)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Must equal the packed teacher dimension")
    parser.add_argument("--temporal-kernel", type=int, choices=[1, 3, 5], default=5)
    parser.add_argument("--stem-channels", type=int, default=16)
    parser.add_argument("--widths", default="24,32,64,96")
    parser.add_argument("--expand", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--huber-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--teacher-stats", default=None, help="Optional NPZ with mean[D] and std[D]")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accelerator", choices=["auto", "cuda", "cpu", "tpu"], default="auto")
    parser.add_argument("--gpus", type=int, default=1, help="Number of CUDA GPUs for DDP; batch size and workers are per GPU")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False, help="Compile the student with torch.compile")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--tpu-cores", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="lip-embedding-frontend")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def set_seed(seed, rank=0):
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_for(accelerator, xm=None):
    if accelerator == "tpu":
        return xm.xla_device()
    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but CUDA is not available.")
        return torch.device("cuda")
    if accelerator == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _make_loader(split, args, train, rank, world_size):
    dataset = PackedAVHubertDataset(
        args.dataset_dir,
        split,
        frames=args.frames,
        random_crop=train,
    )
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            seed=args.seed,
            drop_last=train,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    return loader, sampler


def _run_epoch(loader, sampler, student, optimizer, scaler, device, args,
               feature_mean, feature_std, train, epoch, accelerator, xm=None):
    if sampler is not None and train:
        sampler.set_epoch(epoch)
    student.train(train)
    metric_names = ("loss", "huber", "mse", "cosine_similarity")
    totals = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in metric_names
    }
    num_samples = 0
    label = "train" if train else "val"
    pixel_mean = loader.dataset.meta["pixel_mean"]
    pixel_std = loader.dataset.meta["pixel_std"]

    for frames, target in tqdm(loader, desc=label, dynamic_ncols=True,
                               disable=xm is not None and not xm.is_master_ordinal()):
        # uint8 [B,T,H,W] and float16 [B,T,D] cross the bus; normalize on the device.
        frames = normalize_frames(frames.to(device, non_blocking=True), pixel_mean, pixel_std)
        target = target.to(device, non_blocking=True).permute(1, 0, 2).float()
        if target.shape[-1] != args.embedding_dim:
            raise ValueError(
                f"Teacher dimension is {target.shape[-1]}, but --embedding-dim="
                f"{args.embedding_dim}. Set --embedding-dim to the LMDB feature size."
            )
        model_input = frames.permute(1, 0, 2, 3).unsqueeze(2)

        if accelerator == "cuda" and args.amp:
            autocast = torch.autocast("cuda", dtype=torch.float16)
        elif accelerator == "tpu" and args.amp:
            autocast = torch.autocast("xla", dtype=torch.bfloat16)
        else:
            autocast = nullcontext()

        with torch.set_grad_enabled(train):
            with autocast:
                prediction = student(model_input)
            loss, metrics = kd_loss(
                prediction.float(),
                target.float(),
                args.huber_weight,
                args.cosine_weight,
                feature_mean,
                feature_std,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                elif xm is not None:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()

        batch_size = frames.shape[0]
        num_samples += batch_size
        totals["loss"] += loss.detach().to(torch.float64) * batch_size
        for key, value in metrics.items():
            totals[key] += value.to(torch.float64) * batch_size

    sums = torch.stack([totals[key] for key in metric_names] + [
        torch.tensor(num_samples, device=device, dtype=torch.float64)
    ])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    if xm is not None and xm.xrt_world_size() > 1:
        values = xm.mesh_reduce(
            f"{label}_metrics",
            sums.cpu().tolist(),
            lambda results: [
                sum(result[index] for result in results)
                for index in range(len(results[0]))
            ],
        )
        sums = torch.tensor(values, dtype=torch.float64)
    values = sums.cpu().tolist()
    count = max(values[-1], 1)
    return {
        key: values[index] / count
        for index, key in enumerate(metric_names)
    }


def _unwrap_student(student):
    while True:
        if hasattr(student, "module"):
            student = student.module
        elif hasattr(student, "_orig_mod"):
            student = student._orig_mod
        else:
            return student


def _save_checkpoint(path, student, optimizer, scheduler, scaler, epoch, best_val, args, xm):
    checkpoint = {
        "epoch": epoch,
        "model": _unwrap_student(student).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else {},
        "best_val": best_val,
        "args": vars(args),
    }
    if xm is None:
        torch.save(checkpoint, path)
    else:
        xm.save(checkpoint, path)


def _train(args, accelerator, xm=None, rank=0, world_size=1, local_rank=0):
    if xm is not None:
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
        is_master = xm.is_master_ordinal()
    else:
        is_master = rank == 0
    set_seed(args.seed, rank)
    if accelerator == "cuda":
        torch.cuda.set_device(local_rank)
    device = _device_for(accelerator, xm)

    meta = load_meta(args.dataset_dir)
    if meta["teacher_dim"] != args.embedding_dim:
        raise ValueError(
            f"Packed teacher dimension is {meta['teacher_dim']}, but --embedding-dim={args.embedding_dim}."
        )
    train_loader, train_sampler = _make_loader("train", args, True, rank, world_size)
    val_loader, val_sampler = _make_loader("val", args, False, rank, world_size)
    train_samples, val_samples = train_loader.dataset, val_loader.dataset

    widths = tuple(int(value) for value in args.widths.split(","))
    student = TinyVisualFrontend(
        embedding_dim=args.embedding_dim,
        temporal_kernel=args.temporal_kernel,
        stem_channels=args.stem_channels,
        widths=widths,
        expand=args.expand,
    ).to(device)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0 or newer")
        if accelerator == "tpu":
            raise ValueError("--compile is currently supported here for CUDA/CPU, not the XLA TPU path")
        student = torch.compile(student, mode=args.compile_mode)
    if world_size > 1 and accelerator == "cuda":
        student = DistributedDataParallel(
            student,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.amp and accelerator == "cuda"
    ) if accelerator == "cuda" else None

    feature_mean = feature_std = None
    if args.teacher_stats:
        stats = np.load(args.teacher_stats)
        feature_mean = torch.as_tensor(stats["mean"], device=device, dtype=torch.float32)
        feature_std = torch.as_tensor(stats["std"], device=device, dtype=torch.float32)
        expected = (args.embedding_dim,)
        if feature_mean.shape != expected or feature_std.shape != expected:
            raise ValueError(f"Teacher statistics must have shape {expected}")

    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    if is_master:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
        print("\nAV-HuBERT visual knowledge distillation")
        print("=" * 72)
        print(f"Accelerator         : {accelerator}")
        print(f"Device              : {device}")
        print(f"TPU processes       : {world_size if xm is not None else 0}")
        print(f"CUDA GPUs           : {world_size if accelerator == 'cuda' else 0}")
        print(f"Global batch size   : {args.batch_size * world_size}")
        print(f"Workers per GPU     : {args.workers}")
        print(f"torch.compile       : {args.compile}")
        print(f"Train/val samples   : {len(train_samples):,} / {len(val_samples):,}")
        print(f"Train/val batches   : {len(train_loader):,} / {len(val_loader):,}")
        print(f"Teacher dimension   : {args.embedding_dim}")
        print(f"Student parameters  : {count_parameters(student):,}")
        print("=" * 72)
    if xm is not None:
        xm.rendezvous("avhubert-output-ready")

    start_epoch = 1
    best_val = float("inf")
    history_path = output_dir / "history.json"
    history = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        _unwrap_student(student).load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", best_val)
        if is_master and history_path.is_file():
            history = json.loads(history_path.read_text())

    wandb_run = None
    if is_master and not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or output_dir.name,
            config={
                **vars(args),
                "num_train_samples": len(train_samples),
                "num_val_samples": len(val_samples),
                "student_parameters": count_parameters(student),
                "accelerator": accelerator,
                "world_size": world_size,
            },
            dir=str(output_dir),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            train_loader, train_sampler, student, optimizer, scaler, device,
            args, feature_mean, feature_std, True, epoch, accelerator, xm,
        )
        val_metrics = _run_epoch(
            val_loader, val_sampler, student, optimizer, scaler, device,
            args, feature_mean, feature_std, False, epoch, accelerator, xm,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if is_master:
            row = {"epoch": epoch, "lr": current_lr, "train": train_metrics, "val": val_metrics}
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2) + "\n")
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d} | lr={current_lr:.8g}")
            for label, metrics in (("Train", train_metrics), ("Val", val_metrics)):
                print(
                    f"{label} | loss={metrics['loss']:.6f} "
                    f"huber={metrics['huber']:.6f} mse={metrics['mse']:.6f} "
                    f"cos={metrics['cosine_similarity']:.5f}"
                )
            if wandb_run is not None:
                import wandb

                wandb.log({
                    "epoch": epoch,
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                    "lr": current_lr,
                }, step=epoch)

            _save_checkpoint(
                checkpoint_dir / "last.pt", student, optimizer, scheduler,
                scaler, epoch, best_val, args, xm,
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                _save_checkpoint(
                    checkpoint_dir / "best.pt", student, optimizer, scheduler,
                    scaler, epoch, best_val, args, xm,
                )
                print(f"New best validation loss: {best_val:.6f}")
        if xm is not None:
            xm.rendezvous(f"avhubert-epoch-{epoch}-saved")
        elif dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_master:
        print(f"\nTraining complete. Best validation loss: {best_val:.6f}")
        print(f"Best checkpoint: {checkpoint_dir / 'best.pt'}")
        if wandb_run is not None:
            import wandb

            wandb.finish()


def main():
    args = parse_args()
    if args.embedding_dim <= 0 or args.frames <= 0:
        raise ValueError("Embedding dimension and frame count must be positive")
    if args.gpus < 1 or args.workers < 0 or args.prefetch_factor < 1:
        raise ValueError("--gpus must be >= 1, --workers >= 0, and --prefetch-factor >= 1")
    accelerator = args.accelerator
    if accelerator == "auto":
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    if accelerator == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but CUDA is not available.")
        available_gpus = torch.cuda.device_count()
        if args.gpus > available_gpus:
            raise ValueError(
                f"Requested --gpus {args.gpus}, but only {available_gpus} CUDA GPUs are available."
            )
        if args.gpus > 1:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            mp.spawn(
                _cuda_worker,
                args=(args,),
                nprocs=args.gpus,
                join=True,
            )
        else:
            _train(args, accelerator)
    elif accelerator == "tpu":
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp
        except ImportError as exc:
            raise RuntimeError(
                "--accelerator tpu requires a Kaggle TPU runtime with PyTorch/XLA installed"
            ) from exc
        xmp.spawn(
            _xla_worker,
            args=(args,),
            nprocs=args.tpu_cores,
            start_method="fork",
        )
    else:
        if args.gpus != 1:
            raise ValueError("--gpus is only valid with --accelerator cuda")
        _train(args, accelerator)


def _cuda_worker(local_rank, args):
    dist.init_process_group(
        backend="nccl",
        rank=local_rank,
        world_size=args.gpus,
    )
    try:
        _train(
            args,
            "cuda",
            rank=local_rank,
            world_size=args.gpus,
            local_rank=local_rank,
        )
    finally:
        dist.destroy_process_group()


def _xla_worker(index, args):
    import torch_xla.core.xla_model as xm

    _train(args, "tpu", xm)


if __name__ == "__main__":
    main()