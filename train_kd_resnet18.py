#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VisualKDDataset, samples_for_kd
from losses import kd_loss
from models.tiny_visual_frontend import (
    TinyVisualFrontend,
    count_parameters,
)
from models.visual_frontend import VisualFrontend
import wandb

# ============================================================
# Arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Visual frontend knowledge distillation"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    p.add_argument(
        "--video-root",
        required=True,
        type=str,
        help="VoxCeleb2 origin/train directory.",
    )

    p.add_argument(
        "--data-list",
        required=True,
        type=str,
        help=(
            "SEANet data_list.csv. "
            "Utterances belonging to SEANet val/test are excluded."
        ),
    )

    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing precomputed teacher "
            "embeddings (.npy [T,512]). Directory structure must "
            "mirror --video-root."
        ),
    )
    p.add_argument(
        "--frame-cache-dir",
        type=str,
        default=None,
        help=(
            "Persistent cache for preprocessed visual frames. "
            "Each utterance is stored as uint8 [T,112,112]. "
            "Missing entries are automatically generated from MP4."
        ),
    )


    p.add_argument(
        "--val-utterances",
        type=int,
        default=5000,
        help=(
            "Number of utterances reserved for KD validation "
            "after removing SEANet val/test utterances."
        ),
    )

    p.add_argument(
        "--max-train-utterances",
        type=int,
        default=None,
        help=(
            "Maximum number of utterances used for KD training. "
            "Randomly sampled from the available training pool "
            "after val/test exclusion and KD validation split. "
            "Default: use all available utterances."
        ),
    )
    # --------------------------------------------------------
    # Teacher
    # --------------------------------------------------------

    p.add_argument(
        "--teacher",
        type=str,
        default=None,
        help=(
            "Teacher VisualFrontend checkpoint. Required only "
            "when --cache-dir is not specified."
        ),
    )

    p.add_argument(
        "--teacher-stats",
        type=str,
        default=None,
        help="Optional NPZ containing mean[512] and std[512].",
    )

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    p.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
    )

    p.add_argument(
        "--temporal-kernel",
        type=int,
        default=5,
        choices=[1, 3, 5],
    )

    p.add_argument(
        "--stem-channels",
        type=int,
        default=16,
    )

    p.add_argument(
        "--widths",
        type=str,
        default="24,32,64,96",
    )

    p.add_argument(
        "--expand",
        type=float,
        default=2.0,
    )

    # --------------------------------------------------------
    # Visual preprocessing
    # --------------------------------------------------------

    p.add_argument(
        "--source-fps",
        type=float,
        default=25.0,
    )

    p.add_argument(
        "--fps",
        type=float,
        default=25.0,
    )

    p.add_argument(
        "--frames",
        type=int,
        default=50,
    )

    p.add_argument(
        "--normalization",
        type=str,
        default="mean_std",
        choices=[
            "mean_std",
            "zero_one",
            "minus_one_one",
            "raw",
        ],
    )

    p.add_argument(
        "--pixel-mean",
        type=float,
        default=0.4161,
    )

    p.add_argument(
        "--pixel-std",
        type=float,
        default=0.1688,
    )

    p.add_argument(
        "--spatial-preprocess",
        type=str,
        default="center_crop",
        choices=[
            "center_crop",
            "resize",
        ],
    )

    p.add_argument(
        "--grayscale",
        type=str,
        default="opencv",
        choices=[
            "opencv",
            "average",
            "red",
            "green",
            "blue",
        ],
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--min-lr",
        type=float,
        default=5e-6,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--huber-weight",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--cosine-weight",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    p.add_argument(
        "--output",
        required=True,
        type=str,
    )

    p.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    p.add_argument(
        "--wandb-project",
        type=str,
        default="lip-embedding-frontend",
    )

    p.add_argument(
        "--wandb-name",
        type=str,
        default=None,
    )

    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
    )

    p.add_argument(
        "--no-wandb",
        action="store_true",
    )


    return p.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Teacher
# ============================================================

def load_teacher(path, device):

    if path is None:
        raise ValueError(
            "--teacher is required when --cache-dir "
            "is not specified."
        )

    print()
    print("Loading teacher")
    print("---------------")
    print(f"Checkpoint: {path}")

    model = VisualFrontend()

    state = torch.load(
        path,
        map_location="cpu",
    )

    if (
        isinstance(state, dict)
        and "state_dict" in state
    ):
        state = state["state_dict"]

    model.load_state_dict(
        state,
        strict=True,
    )

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    return model


# ============================================================
# DataLoader
# ============================================================
def make_loader(
    samples,
    args,
    train,
):

    dataset = VisualKDDataset(
        samples=samples,

        frames_per_sample=args.frames,

        source_fps=args.source_fps,
        target_fps=args.fps,

        normalization=args.normalization,
        pixel_mean=args.pixel_mean,
        pixel_std=args.pixel_std,

        spatial=args.spatial_preprocess,
        gray=args.grayscale,

        random_crop=train,

        frame_cache_dir=args.frame_cache_dir,
    )

    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": train,
        "num_workers": args.workers if train else 8,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": train,
        "persistent_workers": train,
    }

    if args.workers > 0 and train:
        kwargs["prefetch_factor"] = 2

    return DataLoader(
        dataset,
        **kwargs,
    )

# ============================================================
# Tensor layout
# ============================================================

def to_frontend_input(
    x,
    device,
):

    # Dataset / DataLoader:
    #
    #   [B,T,H,W]
    #
    # Visual frontend:
    #
    #   [T,B,1,H,W]

    return (
        x
        .to(
            device,
            non_blocking=True,
        )
        .permute(
            1,
            0,
            2,
            3,
        )
        .unsqueeze(2)
    )


def cached_target_to_device(
    target,
    device,
):

    # Dataset / DataLoader:
    #
    #   [B,T,512]
    #
    # Teacher frontend output:
    #
    #   [T,B,512]

    return (
        target
        .to(
            device,
            non_blocking=True,
        )
        .permute(
            1,
            0,
            2,
        )
        .float()
    )


# ============================================================
# Epoch
# ============================================================

def run_epoch(
    loader,
    student,
    teacher,
    optimizer,
    scaler,
    device,
    args,
    feature_mean,
    feature_std,
    train,
):

    if train:
        student.train()
    else:
        student.eval()

    if teacher is not None:
        teacher.eval()

    totals = {
        "loss": 0.0,
        "huber": 0.0,
        "mse": 0.0,
        "cosine_similarity": 0.0,
    }

    num_samples = 0

    description = (
        "train"
        if train
        else "val"
    )

    progress = tqdm(
        loader,
        desc=description,
        dynamic_ncols=True,
    )

    for (
        x,
        cached_target,
        _keys,
    ) in progress:

        # ====================================================
        # Student input
        # ====================================================

        x = to_frontend_input(
            x,
            device,
        )

        # ====================================================
        # Teacher target
        # ====================================================

        if cached_target.numel() > 0:

            target = cached_target_to_device(
                cached_target,
                device,
            )

        else:

            if teacher is None:
                raise RuntimeError(
                    "Dataset returned no cached teacher target "
                    "but no teacher model is loaded."
                )

            # no_grad rather than inference_mode:
            # target will participate in a loss whose backward
            # computes gradients with respect to the student.

            with torch.no_grad():

                target = (
                    teacher(x)
                    .float()
                )

        # ====================================================
        # Forward student + KD loss
        # ====================================================

        with torch.set_grad_enabled(train):

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    args.amp
                    and device.type == "cuda"
                ),
            ):

                prediction = student(x)

            # Keep KD loss itself in FP32.

            loss, metrics = kd_loss(
                prediction.float(),
                target.float(),
                args.huber_weight,
                args.cosine_weight,
                feature_mean,
                feature_std,
            )

            # =================================================
            # Backward
            # =================================================

            if train:

                optimizer.zero_grad(
                    set_to_none=True,
                )

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    args.grad_clip,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

        # ====================================================
        # Statistics
        # ====================================================

        batch_size = x.shape[1]

        num_samples += batch_size

        totals["loss"] += (
            loss.item()
            * batch_size
        )

        for key, value in metrics.items():

            totals[key] += (
                value.item()
                * batch_size
            )

        progress.set_postfix(
            loss=(
                f"{totals['loss'] / num_samples:.5f}"
            ),
            cos=(
                f"{totals['cosine_similarity'] / num_samples:.4f}"
            ),
        )

    return {
        key: value / max(num_samples, 1)
        for key, value in totals.items()
    }


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    student,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val,
    args,
):

    checkpoint = {
        "epoch": epoch,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val": best_val,
        "args": vars(args),
    }

    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# History
# ============================================================

def save_history(
    history,
    path,
):

    with open(
        path,
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Configuration checks
    # ========================================================

    if args.embedding_dim != 512:
        raise ValueError(
            "This trainer performs direct KD against the "
            "512-dimensional teacher representation. "
            "--embedding-dim must therefore be 512."
        )

    if args.cache_dir is None and args.teacher is None:
        raise ValueError(
            "Specify either --cache-dir with precomputed "
            "teacher embeddings or --teacher for online "
            "teacher inference."
        )

    if args.val_utterances <= 0:
        raise ValueError(
            "--val-utterances must be > 0."
        )

    # ========================================================
    # Seed
    # ========================================================

    set_seed(
        args.seed
    )

    # ========================================================
    # Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 72)
    print("VISUAL FRONTEND KNOWLEDGE DISTILLATION")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Seed                : {args.seed}")
    print(f"Target FPS          : {args.fps}")
    print(f"Frames/sample       : {args.frames}")
    print(f"Validation utterances: {args.val_utterances:,}")

    if args.cache_dir is not None:
        print("Teacher targets     : precomputed cache")
        print(f"Cache               : {args.cache_dir}")
    else:
        print("Teacher targets     : online teacher")

    print("=" * 72)
    print()

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = Path(
        args.output
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "config.json",
        "w",
    ) as f:

        json.dump(
            vars(args),
            f,
            indent=2,
        )


    # ========================================================
    # Build dataset
    # ========================================================

    train_samples, val_samples = samples_for_kd(
        video_root=args.video_root,
        data_list=args.data_list,
        cache_dir=args.cache_dir,
        val_utterances=args.val_utterances,
        max_train_utterances=args.max_train_utterances,
        seed=args.seed,
    )

    if len(train_samples) == 0:
        raise RuntimeError(
            "No training samples available."
        )

    if len(val_samples) == 0:
        raise RuntimeError(
            "No validation samples available."
        )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = make_loader(
        train_samples,
        args,
        train=True,
    )

    val_loader = make_loader(
        val_samples,
        args,
        train=False,
    )

    print("DataLoaders")
    print("-----------")
    print(
        f"Train samples       : "
        f"{len(train_samples):,}"
    )
    print(
        f"Validation samples  : "
        f"{len(val_samples):,}"
    )
    print(
        f"Train batches       : "
        f"{len(train_loader):,}"
    )
    print(
        f"Validation batches  : "
        f"{len(val_loader):,}"
    )
    print()

    # ========================================================
    # Teacher
    # ========================================================

    if args.cache_dir is not None:

        teacher = None

        print(
            "Teacher model       : not loaded "
            "(using cached targets)"
        )

    else:

        teacher = load_teacher(
            args.teacher,
            device,
        )

    # ========================================================
    # Student
    # ========================================================

    widths = tuple(
        int(x)
        for x in args.widths.split(",")
    )

    student = TinyVisualFrontend(
        embedding_dim=args.embedding_dim,
        temporal_kernel=args.temporal_kernel,
        stem_channels=args.stem_channels,
        widths=widths,
        expand=args.expand,
    ).to(device)

    student_parameters = count_parameters(
        student
    )

    print()
    print("Student")
    print("-------")
    print(
        f"Parameters          : "
        f"{student_parameters:,}"
    )
    print(
        f"Embedding dim       : "
        f"{args.embedding_dim}"
    )
    print(
        f"Temporal kernel     : "
        f"{args.temporal_kernel}"
    )

    if teacher is not None:

        teacher_parameters = count_parameters(
            teacher
        )

        print(
            f"Teacher parameters  : "
            f"{teacher_parameters:,}"
        )

        print(
            f"Parameter reduction : "
            f"{teacher_parameters / student_parameters:.2f}x"
        )

    print()

    # ========================================================
    # Optional teacher statistics
    # ========================================================

    feature_mean = None
    feature_std = None

    if args.teacher_stats is not None:

        stats = np.load(
            args.teacher_stats
        )

        feature_mean = torch.as_tensor(
            stats["mean"],
            device=device,
            dtype=torch.float32,
        )

        feature_std = torch.as_tensor(
            stats["std"],
            device=device,
            dtype=torch.float32,
        )

        if feature_mean.shape != (512,):
            raise ValueError(
                f"Teacher mean must have shape (512,), "
                f"got {feature_mean.shape}"
            )

        if feature_std.shape != (512,):
            raise ValueError(
                f"Teacher std must have shape (512,), "
                f"got {feature_std.shape}"
            )

        print(
            f"Teacher statistics  : "
            f"{args.teacher_stats}"
        )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            args.amp
            and device.type == "cuda"
        ),
    )

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 1
    best_val = float("inf")

    if args.resume is not None:

        print()
        print(
            f"Resuming from: "
            f"{args.resume}"
        )

        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
        )

        student.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        scaler.load_state_dict(
            checkpoint["scaler"]
        )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        best_val = checkpoint.get(
            "best_val",
            float("inf"),
        )

        print(
            f"Resume epoch        : "
            f"{start_epoch}"
        )

        print(
            f"Best validation     : "
            f"{best_val:.6f}"
        )

    # ========================================================
    # History
    # ========================================================

    history_path = (
        output_dir
        / "history.json"
    )

    if (
        args.resume is not None
        and history_path.is_file()
    ):

        with open(
            history_path,
            "r",
        ) as f:

            history = json.load(f)

    else:

        history = []



    # ========================================================
    # LOGGER
    # ========================================================

    wandb_run = None

    if not args.no_wandb:
        
        if args.wandb_name is None:
            args.wandb_name = Path(args.output).name

            
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                **vars(args),

                # Dataset
                "num_train_samples": len(train_samples),
                "num_val_samples": len(val_samples),

                # Model
                "student_parameters": count_parameters(student),

                # Runtime
                "device": str(device),
            },
            dir=str(output_dir),
        )

        wandb.define_metric(
            "epoch"
        )

        wandb.define_metric(
            "train/*",
            step_metric="epoch",
        )

        wandb.define_metric(
            "val/*",
            step_metric="epoch",
        )

        wandb.define_metric(
            "lr",
            step_metric="epoch",
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        print()
        print("=" * 72)
        print(
            f"EPOCH {epoch:03d}/{args.epochs:03d}"
        )
        print("=" * 72)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = run_epoch(
            loader=train_loader,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            feature_mean=feature_mean,
            feature_std=feature_std,
            train=True,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = run_epoch(
            loader=val_loader,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            feature_mean=feature_mean,
            feature_std=feature_std,
            train=False,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        scheduler.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(
            row
        )

        save_history(
            history,
            history_path,
        )

        print()
        print("Results")
        print("-------")

        print(
            f"Train | "
            f"loss={train_metrics['loss']:.6f} "
            f"huber={train_metrics['huber']:.6f} "
            f"mse={train_metrics['mse']:.6f} "
            f"cos={train_metrics['cosine_similarity']:.5f}"
        )

        print(
            f"Val   | "
            f"loss={val_metrics['loss']:.6f} "
            f"huber={val_metrics['huber']:.6f} "
            f"mse={val_metrics['mse']:.6f} "
            f"cos={val_metrics['cosine_similarity']:.5f}"
        )

        print(
            f"LR    | "
            f"{current_lr:.8g}"
        )


        if wandb_run is not None:

            wandb.log(
                {
                    "epoch": epoch,

                    # Training
                    "train/loss": train_metrics["loss"],
                    "train/huber": train_metrics["huber"],
                    "train/mse": train_metrics["mse"],
                    "train/cosine_similarity":
                        train_metrics["cosine_similarity"],

                    # Validation
                    "val/loss": val_metrics["loss"],
                    "val/huber": val_metrics["huber"],
                    "val/mse": val_metrics["mse"],
                    "val/cosine_similarity":
                        val_metrics["cosine_similarity"],

                    # Optimizer
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            
        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=(
                checkpoint_dir
                / "last.pt"
            ),
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val=best_val,
            args=args,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_metrics["loss"] < best_val:

            best_val = (
                val_metrics["loss"]
            )

            save_checkpoint(
                path=(
                    checkpoint_dir
                    / "best.pt"
                ),
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val=best_val,
                args=args,
            )

            print(
                f"New best validation loss: "
                f"{best_val:.6f}"
            )

    # ========================================================
    # Done
    # ========================================================

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(
        f"Best validation loss: "
        f"{best_val:.6f}"
    )
    print(
        f"Best checkpoint: "
        f"{checkpoint_dir / 'best.pt'}"
    )
    print("=" * 72)

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()