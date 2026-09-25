#!/usr/bin/env python3
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def temporal_indices(num_frames, source_fps, target_fps):
    if num_frames <= 0:
        return np.empty((0,), dtype=np.int64)
    if abs(target_fps - source_fps) < 1e-8:
        return np.arange(num_frames, dtype=np.int64)

    duration = num_frames / source_fps
    n = max(int(np.floor(duration * target_fps + 1e-8)), 1)
    times = np.arange(n, dtype=np.float64) / target_fps
    idx = np.rint(times * source_fps).astype(np.int64)
    return np.unique(np.clip(idx, 0, num_frames - 1))



def atomic_save_npy(path, array):
    """
    Safely create a .npy cache file.

    The temporary file avoids leaving a corrupted cache entry if
    a DataLoader worker/process is interrupted during np.save().
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(tmp, "wb") as f:
        np.save(
            f,
            array,
            allow_pickle=False,
        )

    tmp.replace(path)


def grayscale(frame, method="opencv"):
    if method == "opencv":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if method == "average":
        return frame.astype(np.float32).mean(axis=2)
    channel = {"blue": 0, "green": 1, "red": 2}.get(method)
    if channel is None:
        raise ValueError(method)
    return frame[:, :, channel]


def preprocess_frame(frame, spatial="center_crop", gray="opencv"):
    frame = grayscale(frame, gray)
    if spatial == "center_crop":
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frame = frame[56:168, 56:168]
    elif spatial == "resize":
        frame = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(spatial)
    return frame


def read_video(path, source_fps=25.0, target_fps=25.0,
               spatial="center_crop", gray="opencv"):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(preprocess_frame(frame, spatial, gray))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    frames = np.stack(frames)
    return frames[temporal_indices(len(frames), source_fps, target_fps)]


def normalize(frames, mode="mean_std", mean=0.4161, std=0.1688):
    x = torch.from_numpy(np.asarray(frames)).float()
    if mode == "zero_one":
        x = x / 255.0
    elif mode == "minus_one_one":
        x = (x / 255.0 - 0.5) / 0.5
    elif mode == "mean_std":
        x = (x / 255.0 - mean) / std
    elif mode != "raw":
        raise ValueError(mode)
    return x


def utterance_key(path, root):
    """
    Example:
        root/id00012/abc/00123.mp4
        root/id00012/abc/00123.npy

    -> id00012/abc/00123
    """
    return (
        Path(path)
        .relative_to(root)
        .with_suffix("")
        .as_posix()
    )


def samples_for_kd(
    video_root,
    data_list,
    cache_dir=None,
    val_utterances=5000,
    max_train_utterances=None,
    seed=42,
):
    video_root = Path(video_root)
    data_list = Path(data_list)
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    if not video_root.is_dir():
        raise FileNotFoundError(
            f"Video root not found: {video_root}"
        )

    if not data_list.is_file():
        raise FileNotFoundError(
            f"Data list not found: {data_list}"
        )

    # ========================================================
    # 1. SEANet val/test blacklist
    # ========================================================

    excluded = set()

    with open(data_list, "r", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row or len(row) < 4:
                continue

            if row[0].strip() not in ("val", "test"):
                continue

            speaker = row[2].strip()
            utterance = row[3].strip()

            excluded.add(
                (Path(speaker) / utterance).as_posix()
            )

    # ========================================================
    # 2. Scan MP4 dataset
    # ========================================================

    print("Scanning MP4 dataset...")

    video_map = {}

    for path in video_root.rglob("*.mp4"):
        if not path.is_file():
            continue

        key = utterance_key(
            path,
            video_root,
        )

        video_map[key] = path

    # ========================================================
    # 3. Optional teacher embedding cache
    # ========================================================

    cache_map = None

    if cache_dir is not None:

        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"Cache directory not found: {cache_dir}"
            )

        print("Scanning teacher embedding cache...")

        cache_map = {}

        for path in cache_dir.rglob("*.npy"):
            if not path.is_file():
                continue

            key = utterance_key(
                path,
                cache_dir,
            )

            cache_map[key] = path

    # ========================================================
    # 4. Determine available utterances
    # ========================================================

    if cache_map is None:
        available_keys = set(video_map)
    else:
        # We need both:
        #   MP4 -> student input
        #   NPY -> teacher target
        available_keys = (
            set(video_map)
            & set(cache_map)
        )

    # ========================================================
    # 5. Remove SEANet val/test
    # ========================================================

    removed_keys = (
        available_keys
        & excluded
    )

    allowed_keys = sorted(
        available_keys
        - excluded
    )

    if val_utterances >= len(allowed_keys):
        raise ValueError(
            f"--val-utterances={val_utterances}, "
            f"but only {len(allowed_keys)} samples are available."
        )

    # ========================================================
    # Train / validation split
    # ========================================================

    rng = random.Random(seed)

    # Work on a copy because we are going to shuffle it.
    shuffled_keys = list(allowed_keys)

    rng.shuffle(shuffled_keys)

    # --------------------------------------------------------
    # KD validation
    # --------------------------------------------------------

    val_keys = shuffled_keys[:val_utterances]

    # Everything that remains is eligible for KD training.
    train_pool = shuffled_keys[val_utterances:]

    # --------------------------------------------------------
    # Optional training subset
    # --------------------------------------------------------

    if max_train_utterances is not None:

        if max_train_utterances <= 0:
            raise ValueError(
                "--max-train-utterances must be > 0"
            )

        if max_train_utterances < len(train_pool):

            # train_pool is already randomly shuffled.
            train_keys = train_pool[:max_train_utterances]

        else:

            train_keys = train_pool

    else:

        train_keys = train_pool

    # ========================================================
    # 7. Build records
    # ========================================================

    def make_record(key):

        return {
            "key": key,
            "video": video_map[key],
            "cache": (
                cache_map[key]
                if cache_map is not None
                else None
            ),
        }

    train_samples = [
        make_record(k)
        for k in train_keys
    ]

    val_samples = [
        make_record(k)
        for k in val_keys
    ]

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 72)
    print("VISUAL KD DATASET")
    print("=" * 72)

    print(f"MP4 root                     : {video_root}")
    print(f"MP4 utterances               : {len(video_map):,}")

    if cache_map is not None:
        print(f"Teacher cache                : {cache_dir}")
        print(f"Cached utterances            : {len(cache_map):,}")
        print(
            f"MP4 + cache intersection     : "
            f"{len(available_keys):,}"
        )
        print(
            f"MP4 without cache            : "
            f"{len(set(video_map) - set(cache_map)):,}"
        )
        print(
            f"Cache without MP4            : "
            f"{len(set(cache_map) - set(video_map)):,}"
        )

    print(f"SEANet val/test blacklist    : {len(excluded):,}")
    print(f"Actually excluded            : {len(removed_keys):,}")
    print(f"Available after filtering    : {len(allowed_keys):,}")
    print()
    print(f"KD train                     : {len(train_samples):,}")
    print(f"KD validation                : {len(val_samples):,}")
    print(f"Split seed                   : {seed}")
    print("=" * 72)
    print()

    print(f"Available after filtering    : {len(allowed_keys):,}")
    print()
    print(f"KD validation                : {len(val_keys):,}")
    print(f"KD train pool                : {len(train_pool):,}")

    if max_train_utterances is not None:
        print(f"Max requested KD train       : {max_train_utterances:,}")

    print(f"KD train actually used       : {len(train_keys):,}")
    return train_samples, val_samples


def videos_for_kd(
    video_root,
    data_list,
    val_utterances=5000,
    seed=1337,
):
    """
    Build train/validation datasets for visual KD.

    Procedure:
        1. Scan ALL .mp4 files under VoxCeleb2 origin/train.
        2. Read SEANet data_list.csv.
        3. Build a blacklist containing target utterances belonging
           to SEANet val and test.
        4. Remove those utterances from VoxCeleb2.
        5. Randomly select `val_utterances` utterances from the
           remaining pool for KD validation.
        6. Use everything else for KD training.

    Expected VoxCeleb2 layout:

        video_root/
            idXXXXX/
                video_id/
                    utterance.mp4

    Relevant CSV columns:

        row[0] = SEANet split: train / val / test
        row[2] = target speaker ID
        row[3] = target utterance: video_id/utterance_id

    Returns:
        train_videos, val_videos
    """

    video_root = Path(video_root)
    data_list = Path(data_list)

    if not video_root.is_dir():
        raise FileNotFoundError(
            f"Video root not found: {video_root}"
        )

    if not data_list.is_file():
        raise FileNotFoundError(
            f"Data list not found: {data_list}"
        )

    if val_utterances < 1:
        raise ValueError(
            "--val-utterances must be >= 1"
        )

    # ========================================================
    # 1. Build blacklist from SEANet val/test
    # ========================================================

    excluded = set()

    with open(data_list, "r", newline="") as f:

        reader = csv.reader(f)

        for row in reader:

            if not row:
                continue

            if len(row) < 4:
                continue

            split = row[0].strip()

            if split not in ("val", "test"):
                continue

            speaker = row[2].strip()
            utterance = row[3].strip()

            # Example:
            #
            # id04617/hrPMKzisooU/00296

            key = (
                Path(speaker)
                / utterance
            ).as_posix()

            excluded.add(key)

    # ========================================================
    # 2. Scan ALL VoxCeleb2 origin/train
    # ========================================================

    all_videos = sorted(
        path
        for path in video_root.rglob("*.mp4")
        if path.is_file()
    )

    if not all_videos:
        raise RuntimeError(
            f"No MP4 files found under {video_root}"
        )

    # ========================================================
    # 3. Remove SEANet val/test utterances
    # ========================================================

    allowed_videos = []
    removed_videos = []

    for video_path in all_videos:

        # Example:
        #
        # video_path:
        # origin/train/id04617/hrPMKzisooU/00296.mp4
        #
        # relative:
        # id04617/hrPMKzisooU/00296

        relative = (
            video_path
            .relative_to(video_root)
            .with_suffix("")
            .as_posix()
        )

        if relative in excluded:
            removed_videos.append(video_path)
        else:
            allowed_videos.append(video_path)

    if not allowed_videos:
        raise RuntimeError(
            "No videos remain after removing "
            "SEANet val/test utterances."
        )

    # ========================================================
    # 4. Check validation size
    # ========================================================

    if val_utterances >= len(allowed_videos):
        raise ValueError(
            f"--val-utterances={val_utterances} but only "
            f"{len(allowed_videos)} utterances are available "
            f"after filtering."
        )

    # ========================================================
    # 5. Deterministic random split
    # ========================================================

    videos = allowed_videos.copy()

    rng = random.Random(seed)
    rng.shuffle(videos)

    val_videos = videos[:val_utterances]
    train_videos = videos[val_utterances:]

    # Sorting after random selection is optional, but makes
    # the stored lists/logs deterministic and easier to inspect.

    train_videos = sorted(train_videos)
    val_videos = sorted(val_videos)

    # ========================================================
    # 6. Sanity checks
    # ========================================================

    train_set = set(train_videos)
    val_set = set(val_videos)

    assert train_set.isdisjoint(val_set)

    assert (
        len(train_videos)
        + len(val_videos)
        == len(allowed_videos)
    )

    # ========================================================
    # 7. Summary
    # ========================================================

    print()
    print("=" * 70)
    print("VISUAL KD DATASET")
    print("=" * 70)

    print(f"Video root                 : {video_root}")
    print(f"SEANet data list           : {data_list}")
    print()

    print(f"All VoxCeleb2 utterances   : {len(all_videos):,}")
    print(f"SEANet val/test blacklist  : {len(excluded):,}")
    print(f"Actually removed           : {len(removed_videos):,}")
    print(f"Allowed after filtering    : {len(allowed_videos):,}")

    print()
    print(f"KD training utterances     : {len(train_videos):,}")
    print(f"KD validation utterances   : {len(val_videos):,}")
    print(f"Random split seed          : {seed}")

    print("=" * 70)
    print()

    return train_videos, val_videos

class VisualKDDataset(Dataset):

    def __init__(
        self,
        samples,
        frames_per_sample=50,
        source_fps=25.0,
        target_fps=25.0,
        normalization="mean_std",
        pixel_mean=0.4161,
        pixel_std=0.1688,
        spatial="center_crop",
        gray="opencv",
        random_crop=True,
        frame_cache_dir=None,
    ):

        self.samples = list(samples)

        self.frames_per_sample = int(
            frames_per_sample
        )

        self.source_fps = float(
            source_fps
        )

        self.target_fps = float(
            target_fps
        )

        self.normalization = normalization
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        self.spatial = spatial
        self.gray = gray

        self.random_crop = random_crop

        self.frame_cache_dir = (
            Path(frame_cache_dir)
            if frame_cache_dir is not None
            else None
        )

    def __len__(self):
        return len(self.samples)

    # ========================================================
    # Frame cache
    # ========================================================

    def _frame_cache_path(
        self,
        sample,
    ):

        if self.frame_cache_dir is None:
            return None

        # sample["key"]:
        #
        # id00012/video_id/utterance

        return (
            self.frame_cache_dir
            / f"{sample['key']}.npy"
        )

    def _load_frames(
        self,
        sample,
    ):

        video_path = sample["video"]

        cache_path = self._frame_cache_path(
            sample
        )

        # ----------------------------------------------------
        # Cache hit
        # ----------------------------------------------------

        if (
            cache_path is not None
            and cache_path.is_file()
        ):

            try:

                frames = np.load(
                    cache_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )

                if frames.ndim != 3:
                    raise ValueError(
                        f"Expected [T,H,W], got "
                        f"{frames.shape}"
                    )

                if (
                    frames.shape[1] != 112
                    or frames.shape[2] != 112
                ):
                    raise ValueError(
                        f"Expected [T,112,112], got "
                        f"{frames.shape}"
                    )

                return frames

            except Exception as exc:

                raise RuntimeError(
                    f"Invalid frame cache: "
                    f"{cache_path}"
                ) from exc

        # ----------------------------------------------------
        # Cache miss -> MP4
        # ----------------------------------------------------

        frames = read_video(
            video_path,
            source_fps=self.source_fps,
            target_fps=self.target_fps,
            spatial=self.spatial,
            gray=self.gray,
        )

        # We intentionally cache uint8:
        #
        #   grayscale
        #   spatially preprocessed
        #   temporally sampled
        #
        # but NOT normalized.

        if frames.dtype != np.uint8:

            frames = np.clip(
                frames,
                0,
                255,
            ).astype(
                np.uint8
            )

        # ----------------------------------------------------
        # Create persistent cache
        # ----------------------------------------------------

        if cache_path is not None:

            atomic_save_npy(
                cache_path,
                frames,
            )

        return frames

    # ========================================================
    # Teacher target
    # ========================================================

    def _load_teacher_target(
        self,
        sample,
    ):

        cache_path = sample["cache"]

        if cache_path is None:

            return None

        target = np.load(
            cache_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if target.ndim != 2:

            raise RuntimeError(
                f"Expected teacher cache [T,D], "
                f"got {target.shape}: "
                f"{cache_path}"
            )

        return target

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Load visual frames
        # ----------------------------------------------------

        frames = self._load_frames(
            sample
        )

        # ----------------------------------------------------
        # Teacher embeddings
        # ----------------------------------------------------

        target = self._load_teacher_target(
            sample
        )

        if (
            target is not None
            and len(target) != len(frames)
        ):

            raise RuntimeError(
                f"Temporal mismatch for "
                f"{sample['key']}: "
                f"frames={len(frames)}, "
                f"teacher={len(target)}"
            )

        # ----------------------------------------------------
        # Temporal crop
        # ----------------------------------------------------

        total_frames = len(frames)
        n = self.frames_per_sample

        if total_frames < n:

            shortage = (
                n - total_frames
            )

            # mmap arrays are read-only, therefore convert
            # the selected data to a regular ndarray here.

            frames = np.asarray(
                frames
            )

            frames = np.pad(
                frames,
                (
                    (0, shortage),
                    (0, 0),
                    (0, 0),
                ),
                mode="edge",
            )

            if target is not None:

                target = np.asarray(
                    target
                )

                target = np.pad(
                    target,
                    (
                        (0, shortage),
                        (0, 0),
                    ),
                    mode="edge",
                )

            start = 0

        else:

            if total_frames == n:

                start = 0

            elif self.random_crop:

                start = random.randint(
                    0,
                    total_frames - n,
                )

            else:

                start = (
                    total_frames - n
                ) // 2

            # IMPORTANT:
            #
            # Slice mmap BEFORE converting to ndarray.
            # We therefore touch only the selected 50 frames.

            frames = np.asarray(
                frames[
                    start:start + n
                ]
            )

            if target is not None:

                target = np.asarray(
                    target[
                        start:start + n
                    ]
                )

        # ----------------------------------------------------
        # Normalize student input
        # ----------------------------------------------------

        x = normalize(
            frames,
            mode=self.normalization,
            mean=self.pixel_mean,
            std=self.pixel_std,
        )

        # ----------------------------------------------------
        # Teacher target
        # ----------------------------------------------------

        if target is not None:

            # Copy because mmap-backed NumPy arrays may be
            # read-only.

            target = torch.from_numpy(
                np.array(
                    target,
                    dtype=np.float32,
                    copy=True,
                )
            )

        else:

            # DataLoader cannot collate None.
            #
            # Empty tensor tells train_kd.py to compute the
            # teacher online.

            target = torch.empty(
                0,
                dtype=torch.float32,
            )

        return (
            x,
            target,
            sample["key"],
        )

def collate_skip_errors(batch):
    # Kept simple intentionally: decoding failures should be fixed rather than hidden.
    xs, paths = zip(*batch)
    return torch.stack(xs, 0), list(paths)
