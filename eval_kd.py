#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VisualKDDataset
from models.tiny_visual_frontend import (
    TinyVisualFrontend,
    count_parameters,
)
from models.visual_frontend_resnet18 import VisualFrontend
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# ============================================================
# Arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Evaluate distilled visual frontend"
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
        help="SEANet data_list.csv.",
    )

    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )

    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "Optional precomputed teacher embedding cache. "
            "Expected .npy [T,512] with structure matching video root."
        ),
    )

    p.add_argument(
        "--frame-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional preprocessed uint8 frame cache used during KD."
        ),
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    p.add_argument(
        "--student",
        required=True,
        type=str,
        help="Student KD checkpoint.",
    )

    p.add_argument(
        "--teacher",
        type=str,
        default=None,
        help=(
            "Teacher checkpoint. Required only when --cache-dir "
            "is not specified."
        ),
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    p.add_argument(
        "--output",
        required=True,
        type=str,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Optionally evaluate only N utterances.",
    )

    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # Overrides
    # --------------------------------------------------------

    p.add_argument(
        "--source-fps",
        type=float,
        default=None,
        help="Override checkpoint source FPS.",
    )

    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override checkpoint target FPS.",
    )

    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override checkpoint frames/sample.",
    )

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------

    p.add_argument(
        "--wandb-project",
        type=str,
        default=None,
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

    return p.parse_args()


# ============================================================
# Utilities
# ============================================================

def save_cosine_distribution_plot(
    frame_cosines,
    utterance_cosines,
    output_path,
):
    frame_cosines = np.asarray(frame_cosines, dtype=np.float64)
    utterance_cosines = np.asarray(
        utterance_cosines,
        dtype=np.float64,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    datasets = [
        (
            axes[0],
            frame_cosines,
            "Frame-level cosine similarity",
            f"{len(frame_cosines):,} frame embeddings",
        ),
        (
            axes[1],
            utterance_cosines,
            "Utterance-level cosine similarity",
            f"{len(utterance_cosines):,} utterances",
        ),
    ]

    for ax, values, title, subtitle in datasets:

        mean = np.mean(values)
        median = np.median(values)
        p05 = np.percentile(values, 5)
        p95 = np.percentile(values, 95)

        # Histogram as density
        ax.hist(
            values,
            bins=80,
            density=True,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.4,
        )

        # Mean
        ax.axvline(
            mean,
            linestyle="-",
            linewidth=2,
            label=f"Mean = {mean:.4f}",
        )

        # Median
        ax.axvline(
            median,
            linestyle="--",
            linewidth=2,
            label=f"Median = {median:.4f}",
        )

        # 5–95 percentile interval
        ax.axvspan(
            p05,
            p95,
            alpha=0.12,
            label=f"P05–P95 = [{p05:.3f}, {p95:.3f}]",
        )

        ax.set_title(
            f"{title}\n{subtitle}",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")

        ax.grid(
            True,
            alpha=0.2,
            linestyle="--",
        )

        ax.legend(
            frameon=True,
            fontsize=9,
        )

        # Useful common reference
        ax.axvline(
            0.90,
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
        )

    fig.suptitle(
        "Teacher–Student Visual Embedding Agreement",
        fontsize=15,
        fontweight="bold",
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)
def distribution_stats(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cfg(cfg, key, default):

    value = cfg.get(key, default)

    if value is None:
        return default

    return value


# ============================================================
# Student
# ============================================================

def load_student(path, device):

    print()
    print("Loading student")
    print("---------------")
    print(f"Checkpoint: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Student checkpoint must be a dictionary."
        )

    cfg = checkpoint.get("args", {})

    widths = tuple(
        int(x)
        for x in str(
            get_cfg(
                cfg,
                "widths",
                "24,32,64,96",
            )
        ).split(",")
    )

    model = TinyVisualFrontend(
        embedding_dim=int(
            get_cfg(cfg, "embedding_dim", 512)
        ),
        temporal_kernel=int(
            get_cfg(cfg, "temporal_kernel", 5)
        ),
        stem_channels=int(
            get_cfg(cfg, "stem_channels", 16)
        ),
        widths=widths,
        expand=float(
            get_cfg(cfg, "expand", 2.0)
        ),
    )

    state = checkpoint.get(
        "model",
        checkpoint.get("state_dict"),
    )

    if state is None:
        raise RuntimeError(
            "Could not find 'model' or 'state_dict' "
            "inside student checkpoint."
        )

    model.load_state_dict(
        state,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print(
        f"Parameters: {count_parameters(model):,}"
    )

    return model, cfg, checkpoint


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

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(
        f"Parameters: {count_parameters(model):,}"
    )

    return model


# ============================================================
# CSV split
# ============================================================

def samples_from_split(
    data_list,
    video_root,
    cache_dir,
    split,
):

    video_root = Path(video_root)

    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else None
    )

    samples = []

    missing_video = 0
    missing_cache = 0
    duplicates = 0

    seen = set()

    with open(
        data_list,
        "r",
        newline="",
    ) as f:

        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            # SEANet CSV:
            # row[0] = split
            # row[2] = target speaker
            # row[3] = target utterance path

            if row[0].strip() != split:
                continue

            if len(row) < 4:
                continue

            speaker = row[2].strip()

            utterance = (
                row[3]
                .strip()
                .replace("\\", "/")
            )

            # row[3] is typically:
            # video_id/utterance_id
            #
            # key becomes:
            # speaker/video_id/utterance_id

            utterance = str(
                Path(utterance).with_suffix("")
            )

            key = (
                Path(speaker)
                / utterance
            ).as_posix()

            if key in seen:
                duplicates += 1
                continue

            seen.add(key)

            video_path = (
                video_root
                / f"{key}.mp4"
            )

            if not video_path.exists():
                missing_video += 1
                continue

            cache_path = None

            if cache_root is not None:

                cache_path = (
                    cache_root
                    / f"{key}.npy"
                )

                if not cache_path.exists():
                    missing_cache += 1
                    continue

            samples.append(
                {
                    "key": key,
                    "video": str(video_path),
                    "cache": (
                        str(cache_path)
                        if cache_path is not None
                        else None
                    ),
                }
            )

    print()
    print("Evaluation split")
    print("----------------")
    print(f"Split               : {split}")
    print(f"Resolved utterances : {len(samples):,}")
    print(f"Missing videos      : {missing_video:,}")

    if cache_root is not None:
        print(f"Missing teacher cache: {missing_cache:,}")

    print(f"Duplicate CSV targets: {duplicates:,}")

    return samples


# ============================================================
# DataLoader
# ============================================================

def make_loader(
    samples,
    args,
    cfg,
):

    source_fps = (
        args.source_fps
        if args.source_fps is not None
        else float(
            get_cfg(cfg, "source_fps", 25.0)
        )
    )

    fps = (
        args.fps
        if args.fps is not None
        else float(
            get_cfg(cfg, "fps", 25.0)
        )
    )

    frames = (
        args.frames
        if args.frames is not None
        else int(
            get_cfg(cfg, "frames", 50)
        )
    )

    dataset = VisualKDDataset(
        samples=samples,
        frames_per_sample=frames,
        source_fps=source_fps,
        target_fps=fps,
        normalization=get_cfg(
            cfg,
            "normalization",
            "mean_std",
        ),
        pixel_mean=float(
            get_cfg(
                cfg,
                "pixel_mean",
                0.4161,
            )
        ),
        pixel_std=float(
            get_cfg(
                cfg,
                "pixel_std",
                0.1688,
            )
        ),
        spatial=get_cfg(
            cfg,
            "spatial_preprocess",
            "center_crop",
        ),
        gray=get_cfg(
            cfg,
            "grayscale",
            "opencv",
        ),
        random_crop=False,
        frame_cache_dir=args.frame_cache_dir,
    )

    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,

        # Evaluation does not need workers surviving
        # across epochs: there is only one pass.
        "persistent_workers": False,
    }

    if args.workers > 0:

        # Deliberately conservative after the observed
        # DataLoader host-RAM OOM.
        kwargs["prefetch_factor"] = 1

    loader = DataLoader(
        dataset,
        **kwargs,
    )

    return (
        loader,
        source_fps,
        fps,
        frames,
    )


# ============================================================
# Tensor layouts
# ============================================================

def to_frontend_input(
    x,
    device,
):

    # [B,T,H,W]
    # ->
    # [T,B,1,H,W]

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

    # [B,T,512]
    # ->
    # [T,B,512]

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
# Metrics
# ============================================================

def compute_metrics(
    student,
    teacher,
):

    # [T,B,D]
    # ->
    # [T*B,D]

    sf = student.reshape(
        -1,
        student.shape[-1],
    )

    tf = teacher.reshape(
        -1,
        teacher.shape[-1],
    )

    mse = F.mse_loss(
        sf,
        tf,
    )

    mae = F.l1_loss(
        sf,
        tf,
    )

    huber = F.smooth_l1_loss(
        sf,
        tf,
    )

    cosine = (
        F.cosine_similarity(
            sf,
            tf,
            dim=-1,
        )
        .mean()
    )

    # Pearson across embedding dimensions,
    # independently for each frame.

    sc = (
        sf
        - sf.mean(
            dim=-1,
            keepdim=True,
        )
    )

    tc = (
        tf
        - tf.mean(
            dim=-1,
            keepdim=True,
        )
    )

    pearson = (
        (
            (sc * tc).sum(dim=-1)
            /
            (
                sc.norm(dim=-1)
                * tc.norm(dim=-1)
            ).clamp_min(1e-8)
        )
        .mean()
    )

    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "huber": huber.item(),
        "cosine_similarity": cosine.item(),
        "pearson": pearson.item(),
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    loader,
    student,
    teacher,
    device,
    max_batches,
):

    totals = {
        "mse": 0.0,
        "mae": 0.0,
        "huber": 0.0,
        "cosine_similarity": 0.0,
        "pearson": 0.0,
    }

    total_frames = 0
    total_utterances = 0
    total_batches = 0

    progress = tqdm(
        loader,
        desc="eval",
        dynamic_ncols=True,
    )


    frame_cosines = []
    utterance_cosines = []


    with torch.inference_mode():

        for batch_index, batch in enumerate(progress):

            if (
                max_batches is not None
                and max_batches > 0
                and batch_index >= max_batches
            ):
                break

            x, cached_target, keys = batch

            x = to_frontend_input(
                x,
                device,
            )

            # --------------------------------------------
            # Teacher
            # --------------------------------------------

            if cached_target.numel() > 0:

                target = cached_target_to_device(
                    cached_target,
                    device,
                )

            else:

                if teacher is None:
                    raise RuntimeError(
                        "No cached target and no teacher loaded."
                    )

                target = (
                    teacher(x)
                    .float()
                )

            # --------------------------------------------
            # Student
            # --------------------------------------------

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                prediction = student(x)

            prediction = prediction.float()
            target = target.float()

            if prediction.shape != target.shape:

                raise RuntimeError(
                    "\nEmbedding shape mismatch:\n"
                    f"  student: {tuple(prediction.shape)}\n"
                    f"  teacher: {tuple(target.shape)}\n"
                    f"  keys   : {list(keys)[:3]}"
                )

            metrics = compute_metrics(
                prediction,
                target,
            )

            # Weight by number of frame embeddings,
            # rather than by batch count.

            num_embeddings = (
                prediction.shape[0]
                * prediction.shape[1]
            )

            for key, value in metrics.items():

                totals[key] += (
                    value
                    * num_embeddings
                )

            total_frames += num_embeddings
            total_utterances += prediction.shape[1]
            total_batches += 1

            progress.set_postfix(
                cos=(
                    f"{totals['cosine_similarity'] / total_frames:.4f}"
                ),
                mse=(
                    f"{totals['mse'] / total_frames:.5f}"
                ),
            )


            # prediction: [T, B, 512]
            # target:     [T, B, 512]

            cos = F.cosine_similarity(
                prediction.float(),
                target.float(),
                dim=-1,
            )  # [T, B]

            # ------------------------------------------------------------
            # Frame-level distribution
            # ------------------------------------------------------------

            frame_cosines.append(
                cos.detach().cpu().reshape(-1)
            )

            # ------------------------------------------------------------
            # Utterance-level distribution
            #
            # Mean over T -> one value per utterance
            # ------------------------------------------------------------

            utt_cos = cos.mean(dim=0)  # [B]

            utterance_cosines.append(
                utt_cos.detach().cpu()
            )



    if total_frames == 0:
        raise RuntimeError(
            "Evaluation produced zero embeddings."
        )

    metrics = {
        key: value / total_frames
        for key, value in totals.items()
    }


    # Concatenate all batches into flat vectors
    frame_cosines = torch.cat(
        frame_cosines,
        dim=0,
    ).numpy()

    utterance_cosines = torch.cat(
        utterance_cosines,
        dim=0,
    ).numpy()

    frame_cosine_stats = distribution_stats(
        frame_cosines
    )

    utterance_cosine_stats = distribution_stats(
        utterance_cosines
    )

    metrics["utterances"] = total_utterances
    metrics["frame_embeddings"] = total_frames
    metrics["batches"] = total_batches
    metrics["cosine_frame_distribution"] = frame_cosine_stats
    metrics["cosine_utterance_distribution"] = utterance_cosine_stats


    return metrics, frame_cosines, utterance_cosines


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("DISTILLED VISUAL FRONTEND EVALUATION")
    print("=" * 72)
    print(f"Device              : {device}")
    print(f"Student             : {args.student}")
    print(f"Split               : {args.split}")

    if args.cache_dir is not None:
        print("Teacher targets     : precomputed cache")
        print(f"Teacher cache       : {args.cache_dir}")
    else:
        print("Teacher targets     : online teacher")
        print(f"Teacher checkpoint  : {args.teacher}")

    print("=" * 72)

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    student, cfg, checkpoint = load_student(
        args.student,
        device,
    )

    # --------------------------------------------------------
    # Resolve split
    # --------------------------------------------------------

    samples = samples_from_split(
        data_list=args.data_list,
        video_root=args.video_root,
        cache_dir=args.cache_dir,
        split=args.split,
    )

    if len(samples) == 0:
        raise RuntimeError(
            f"No utterances resolved for split={args.split}"
        )

    if (
        args.max_utterances is not None
        and args.max_utterances > 0
        and len(samples) > args.max_utterances
    ):

        rng = random.Random(args.seed)

        samples = rng.sample(
            samples,
            args.max_utterances,
        )

        print(
            f"Evaluation subset   : "
            f"{len(samples):,} utterances"
        )

    # --------------------------------------------------------
    # Teacher
    # --------------------------------------------------------

    if args.cache_dir is not None:

        teacher = None

    else:

        teacher = load_teacher(
            args.teacher,
            device,
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    (
        loader,
        source_fps,
        fps,
        frames,
    ) = make_loader(
        samples,
        args,
        cfg,
    )

    print()
    print("Evaluation configuration")
    print("------------------------")
    print(f"Source FPS          : {source_fps}")
    print(f"Target FPS          : {fps}")
    print(f"Frames/sample       : {frames}")
    print(f"Batch size          : {args.batch_size}")
    print(f"Workers             : {args.workers}")
    print(f"Batches             : {len(loader):,}")
    print()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics, frame_cosines, utterance_cosines = evaluate(
        loader=loader,
        student=student,
        teacher=teacher,
        device=device,
        max_batches=args.max_batches,
    )


    plot_path = output_dir / "cosine_distribution.png"

    save_cosine_distribution_plot(
        frame_cosines=frame_cosines,
        utterance_cosines=utterance_cosines,
        output_path=plot_path,
    )

    print(f"Saved: {plot_path}")
    
    
    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    student_params = count_parameters(
        student
    )

    metrics["student_params"] = (
        student_params
    )

    if teacher is not None:

        teacher_params = count_parameters(
            teacher
        )

        metrics["teacher_params"] = (
            teacher_params
        )

        metrics["parameter_reduction_x"] = (
            teacher_params
            / student_params
        )

    metrics["split"] = args.split
    metrics["source_fps"] = source_fps
    metrics["fps"] = fps
    metrics["frames_per_sample"] = frames
    metrics["student_checkpoint"] = str(
        Path(args.student).resolve()
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"MSE                 : "
        f"{metrics['mse']:.8f}"
    )

    print(
        f"MAE                 : "
        f"{metrics['mae']:.8f}"
    )

    print(
        f"Huber               : "
        f"{metrics['huber']:.8f}"
    )

    print(
        f"Cosine similarity   : "
        f"{metrics['cosine_similarity']:.8f}"
    )

    print(
        f"Pearson             : "
        f"{metrics['pearson']:.8f}"
    )

    print(
        f"Utterances          : "
        f"{metrics['utterances']:,}"
    )

    print(
        f"Frame embeddings    : "
        f"{metrics['frame_embeddings']:,}"
    )

    print("=" * 72)


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    metrics_path = (
        output_dir
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print()
    print(f"Saved: {metrics_path}")

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------

    if args.wandb_project is not None:

        import wandb

        run_name = (
            args.wandb_name
            if args.wandb_name is not None
            else (
                Path(args.student).parent.parent.name
                + f"_eval_{args.split}"
            )
        )

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="evaluation",
            config={
                **vars(args),
                "source_fps_actual": source_fps,
                "fps_actual": fps,
                "frames_actual": frames,
                "student_params": student_params,
            },
        )

        wandb.log(
            {
                f"eval/{key}": value
                for key, value in metrics.items()
                if isinstance(
                    value,
                    (int, float),
                )
            }
        )

        run.summary.update(metrics)

        wandb.finish()


if __name__ == "__main__":
    main()