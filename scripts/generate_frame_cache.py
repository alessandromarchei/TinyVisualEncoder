#!/usr/bin/env python3

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import blosc2
import numpy as np
from tqdm import tqdm


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import read_video


# ============================================================
# Arguments
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute lossless compressed visual frame cache."
    )

    p.add_argument(
        "--video-root",
        required=True,
        type=str,
        help="Root containing MP4 files.",
    )

    p.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output root. Directory structure mirrors --video-root.",
    )

    p.add_argument(
        "--data-list",
        type=str,
        default=None,
        help=(
            "Optional SEANet data_list.csv. If specified, all utterances "
            "appearing in the CSV, both target and interferer, are excluded "
            "from frame-cache generation."
        ),
    )

    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help=(
            "Maximum number of videos to process after filtering. "
            "Videos are randomly sampled using --seed. "
            "Default: process all available videos."
        ),
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by --max-videos sampling.",
    )

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
        "--spatial-preprocess",
        type=str,
        default="center_crop",
        choices=["center_crop", "resize"],
    )

    p.add_argument(
        "--grayscale",
        type=str,
        default="opencv",
        choices=["opencv", "average", "red", "green", "blue"],
    )

    p.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )

    p.add_argument(
        "--compression-level",
        type=int,
        default=5,
        choices=range(0, 10),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


# ============================================================
# SEANet blacklist
# ============================================================

def load_data_list_blacklist(data_list):
    """
    Read a SEANet data_list.csv and return all VoxCeleb utterance keys
    referenced by the mixtures.

    Expected columns:

        0  mixture split
        1  target split
        2  target speaker
        3  target utterance
        4  ...
        5  interferer split
        6  interferer speaker
        7  interferer utterance
        ...

    Example:

        train,train,id08108,RCODGWdfzdY/00197,0,
        train,id07077,oi72_HHDWY4/00288,...

    Produces:

        id08108/RCODGWdfzdY/00197
        id07077/oi72_HHDWY4/00288
    """

    data_list = Path(data_list)

    if not data_list.is_file():
        raise FileNotFoundError(
            f"Data list not found: {data_list}"
        )

    blacklist = set()

    target_count = 0
    interferer_count = 0
    rows = 0

    with open(data_list, "r", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            rows += 1

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            if len(row) >= 4:

                speaker = row[2].strip()
                utterance = row[3].strip()

                if speaker and utterance:

                    key = (
                        Path(speaker)
                        / utterance
                    ).as_posix()

                    blacklist.add(key)
                    target_count += 1

            # ------------------------------------------------
            # Interferer
            # ------------------------------------------------

            if len(row) >= 8:

                speaker = row[6].strip()
                utterance = row[7].strip()

                if speaker and utterance:

                    key = (
                        Path(speaker)
                        / utterance
                    ).as_posix()

                    blacklist.add(key)
                    interferer_count += 1

    print()
    print("SEANet data-list blacklist")
    print("---------------------------")
    print(f"Data list             : {data_list}")
    print(f"Rows                  : {rows:,}")
    print(f"Target references     : {target_count:,}")
    print(f"Interferer references : {interferer_count:,}")
    print(f"Unique utterances     : {len(blacklist):,}")

    return blacklist


# ============================================================
# Paths
# ============================================================

def utterance_key(video_path, video_root):
    """
    Example:

        video_root/
            id08108/
                RCODGWdfzdY/
                    00197.mp4

    ->

        id08108/RCODGWdfzdY/00197
    """

    return (
        video_path
        .relative_to(video_root)
        .with_suffix("")
        .as_posix()
    )


def output_path(video_path, video_root, output_root):
    relative = video_path.relative_to(video_root)

    return (
        output_root
        / relative.with_suffix(".b2nd")
    )


# ============================================================
# Worker
# ============================================================

def process_one(
    video_path,
    video_root,
    output_root,
    source_fps,
    target_fps,
    spatial,
    gray,
    compression_level,
    overwrite,
):
    video_path = Path(video_path)
    video_root = Path(video_root)
    output_root = Path(output_root)

    dst = output_path(
        video_path,
        video_root,
        output_root,
    )

    if dst.exists() and not overwrite:
        return (
            "skip",
            str(video_path),
            str(dst),
            0,
            0,
        )

    # --------------------------------------------------------
    # Decode + preprocessing
    # --------------------------------------------------------

    frames = read_video(
        video_path,
        source_fps=source_fps,
        target_fps=target_fps,
        spatial=spatial,
        gray=gray,
    )

    # Cache representation is ALWAYS uint8.
    # No normalization is stored.

    if frames.dtype != np.uint8:
        frames = np.clip(
            frames,
            0,
            255,
        ).astype(np.uint8)

    if frames.ndim != 3:
        raise RuntimeError(
            f"Expected [T,H,W], got "
            f"{frames.shape}: {video_path}"
        )

    if frames.shape[1:] != (112, 112):
        raise RuntimeError(
            f"Expected [T,112,112], got "
            f"{frames.shape}: {video_path}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(dst) + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    # --------------------------------------------------------
    # Blosc2
    #
    # Temporal chunks allow partial decompression when the
    # DataLoader requests a temporal window.
    # --------------------------------------------------------

    chunks = (
        min(16, len(frames)),
        112,
        112,
    )

    arr = blosc2.asarray(
        frames,
        chunks=chunks,
        urlpath=str(tmp),
        mode="w",
        cparams={
            "codec": blosc2.Codec.ZSTD,
            "clevel": compression_level,
            "filters": [
                blosc2.Filter.SHUFFLE,
            ],
        },
    )

    del arr

    tmp.replace(dst)

    raw_bytes = frames.nbytes
    compressed_bytes = dst.stat().st_size

    return (
        "write",
        str(video_path),
        str(dst),
        raw_bytes,
        compressed_bytes,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    video_root = Path(
        args.video_root
    ).resolve()

    output_root = Path(
        args.output_dir
    ).resolve()

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    if not video_root.is_dir():
        raise FileNotFoundError(
            video_root
        )

    if (
        args.max_videos is not None
        and args.max_videos <= 0
    ):
        raise ValueError(
            "--max-videos must be > 0"
        )

    if args.workers <= 0:
        raise ValueError(
            "--workers must be > 0"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Optional SEANet blacklist
    # ========================================================

    blacklist = None

    if args.data_list is not None:
        blacklist = load_data_list_blacklist(
            args.data_list
        )

    # ========================================================
    # Scan videos
    # ========================================================

    print()
    print("Scanning MP4 dataset...")

    all_videos = sorted(
        p
        for p in video_root.rglob("*.mp4")
        if p.is_file()
    )

    if not all_videos:
        raise RuntimeError(
            f"No MP4 files found under {video_root}"
        )

    total_scanned = len(all_videos)

    # ========================================================
    # Filter SEANet utterances
    # ========================================================

    if blacklist is not None:

        videos = []

        excluded_count = 0

        for video in all_videos:

            key = utterance_key(
                video,
                video_root,
            )

            if key in blacklist:
                excluded_count += 1
                continue

            videos.append(video)

    else:

        videos = all_videos
        excluded_count = 0

    available_after_filter = len(videos)

    if not videos:
        raise RuntimeError(
            "No videos remain after data-list filtering."
        )

    # ========================================================
    # Optional random subset
    # ========================================================

    if (
        args.max_videos is not None
        and args.max_videos < len(videos)
    ):

        rng = random.Random(
            args.seed
        )

        videos = rng.sample(
            videos,
            args.max_videos,
        )

        # Stable processing order after random selection.
        videos.sort()

    # ========================================================
    # Configuration
    # ========================================================

    print()
    print("=" * 72)
    print("FRAME CACHE GENERATOR")
    print("=" * 72)

    print(
        f"Video root             : "
        f"{video_root}"
    )

    print(
        f"Output root            : "
        f"{output_root}"
    )

    print()

    print(
        f"MP4 videos found       : "
        f"{total_scanned:,}"
    )

    if blacklist is not None:

        print(
            f"Data-list blacklist    : "
            f"{len(blacklist):,}"
        )

        print(
            f"Actually excluded      : "
            f"{excluded_count:,}"
        )

        print(
            f"Available after filter : "
            f"{available_after_filter:,}"
        )

    if args.max_videos is not None:

        print(
            f"Max videos requested   : "
            f"{args.max_videos:,}"
        )

        print(
            f"Sampling seed          : "
            f"{args.seed}"
        )

    print(
        f"Videos to process      : "
        f"{len(videos):,}"
    )

    print()

    print(
        f"Source FPS             : "
        f"{args.source_fps}"
    )

    print(
        f"Target FPS             : "
        f"{args.fps}"
    )

    print(
        f"Spatial                : "
        f"{args.spatial_preprocess}"
    )

    print(
        f"Grayscale              : "
        f"{args.grayscale}"
    )

    print(
        f"Format                 : "
        f"uint8 [T,112,112]"
    )

    print(
        f"Compression            : "
        f"ZSTD lossless"
    )

    print(
        f"Compression level      : "
        f"{args.compression_level}"
    )

    print(
        f"Workers                : "
        f"{args.workers}"
    )

    print("=" * 72)
    print()

    # ========================================================
    # Processing
    # ========================================================

    written = 0
    skipped = 0
    failed = 0

    raw_total = 0
    compressed_total = 0

    with ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = [
            executor.submit(
                process_one,
                video,
                video_root,
                output_root,
                args.source_fps,
                args.fps,
                args.spatial_preprocess,
                args.grayscale,
                args.compression_level,
                args.overwrite,
            )
            for video in videos
        ]

        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Caching",
            dynamic_ncols=True,
        )

        for future in progress:

            try:
                (
                    status,
                    src,
                    dst,
                    raw_bytes,
                    compressed_bytes,
                ) = future.result()

            except Exception as exc:

                failed += 1

                tqdm.write(
                    f"ERROR: {exc}"
                )

                continue

            if status == "skip":

                skipped += 1

            else:

                written += 1

                raw_total += raw_bytes
                compressed_total += compressed_bytes

            if compressed_total > 0:

                ratio = (
                    raw_total
                    / compressed_total
                )

            else:

                ratio = 0.0

            progress.set_postfix(
                written=written,
                skipped=skipped,
                failed=failed,
                ratio=f"{ratio:.2f}x",
            )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Selected          : "
        f"{len(videos):,}"
    )

    print(
        f"Written           : "
        f"{written:,}"
    )

    print(
        f"Skipped           : "
        f"{skipped:,}"
    )

    print(
        f"Failed            : "
        f"{failed:,}"
    )

    if compressed_total:

        print(
            f"Raw               : "
            f"{raw_total / 1024**3:.2f} GiB"
        )

        print(
            f"Compressed        : "
            f"{compressed_total / 1024**3:.2f} GiB"
        )

        print(
            f"Compression ratio : "
            f"{raw_total / compressed_total:.2f}x"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()