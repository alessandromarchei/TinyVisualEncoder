#!/usr/bin/env python3

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import blosc2
import numpy as np
from tqdm import tqdm

from dataset import read_video


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


def output_path(video_path, video_root, output_root):
    relative = video_path.relative_to(video_root)

    return (
        output_root
        / relative.with_suffix(".b2nd")
    )


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
        return "skip", str(video_path), str(dst), 0, 0

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
            f"Expected [T,H,W], got {frames.shape}: {video_path}"
        )

    if frames.shape[1:] != (112, 112):
        raise RuntimeError(
            f"Expected [T,112,112], got {frames.shape}: {video_path}"
        )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(str(dst) + ".tmp")

    if tmp.exists():
        tmp.unlink()

    # Chunk temporally.
    #
    # 16 frames/chunk means reading a 50-frame training window
    # only needs a handful of compressed chunks.
    chunks = (
        min(16, len(frames)),
        112,
        112,
    )

    arr = blosc2.asarray(
        frames,
        chunks=chunks,
        cparams={
            "codec": blosc2.Codec.ZSTD,
            "clevel": compression_level,
            "filters": [
                blosc2.Filter.SHUFFLE,
            ],
        },
    )

    arr.schunk.tofile(str(tmp))

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


def main():
    args = parse_args()

    video_root = Path(args.video_root).resolve()
    output_root = Path(args.output_dir).resolve()

    if not video_root.is_dir():
        raise FileNotFoundError(video_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    videos = sorted(
        p
        for p in video_root.rglob("*.mp4")
        if p.is_file()
    )

    if not videos:
        raise RuntimeError(
            f"No MP4 files found under {video_root}"
        )

    print()
    print("=" * 72)
    print("FRAME CACHE GENERATOR")
    print("=" * 72)
    print(f"Video root       : {video_root}")
    print(f"Output root      : {output_root}")
    print(f"Videos           : {len(videos):,}")
    print(f"Source FPS       : {args.source_fps}")
    print(f"Target FPS       : {args.fps}")
    print(f"Spatial          : {args.spatial_preprocess}")
    print(f"Grayscale        : {args.grayscale}")
    print(f"Format           : uint8 [T,112,112]")
    print(f"Compression      : ZSTD lossless")
    print(f"Compression level: {args.compression_level}")
    print(f"Workers          : {args.workers}")
    print("=" * 72)
    print()

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
                tqdm.write(f"ERROR: {exc}")
                continue

            if status == "skip":
                skipped += 1
            else:
                written += 1
                raw_total += raw_bytes
                compressed_total += compressed_bytes

            if compressed_total > 0:
                ratio = raw_total / compressed_total
            else:
                ratio = 0.0

            progress.set_postfix(
                written=written,
                skipped=skipped,
                failed=failed,
                ratio=f"{ratio:.2f}x",
            )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Written          : {written:,}")
    print(f"Skipped          : {skipped:,}")
    print(f"Failed           : {failed:,}")

    if compressed_total:
        print(
            f"Raw              : "
            f"{raw_total / 1024**3:.2f} GiB"
        )
        print(
            f"Compressed       : "
            f"{compressed_total / 1024**3:.2f} GiB"
        )
        print(
            f"Compression ratio: "
            f"{raw_total / compressed_total:.2f}x"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()