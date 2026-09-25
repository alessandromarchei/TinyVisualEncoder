#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VisualKDDataset, videos_from_csv
from models.tiny_visual_frontend import TinyVisualFrontend, count_parameters


def parse_args():
    p = argparse.ArgumentParser("Evaluate distilled visual frontend")
    p.add_argument("--video-root", required=True)
    p.add_argument("--data-list", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-fps", type=float, default=25.0)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


def load_teacher(path, device):
    from pretrain_networks.visual_frontend import VisualFrontend
    m = VisualFrontend()
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.student, map_location="cpu")
    cfg = ck.get("args", {})
    widths = tuple(map(int, str(cfg.get("widths", "24,32,64,96")).split(",")))

    student = TinyVisualFrontend(
        embedding_dim=int(cfg.get("embedding_dim", 512)),
        temporal_kernel=int(cfg.get("temporal_kernel", 5)),
        stem_channels=int(cfg.get("stem_channels", 16)),
        widths=widths,
        expand=float(cfg.get("expand", 2.0)),
    )
    student.load_state_dict(ck["model"], strict=True)
    student = student.to(device).eval()
    teacher = load_teacher(args.teacher, device)

    videos = videos_from_csv(args.data_list, args.video_root, args.split)
    if not videos:
        raise RuntimeError(f"No videos resolved for split={args.split}")

    ds = VisualKDDataset(
        videos, args.frames, args.source_fps, args.fps,
        normalization=cfg.get("normalization", "mean_std"),
        pixel_mean=float(cfg.get("pixel_mean", 0.4161)),
        pixel_std=float(cfg.get("pixel_std", 0.1688)),
        spatial=cfg.get("spatial_preprocess", "center_crop"),
        gray=cfg.get("grayscale", "opencv"),
        random_crop=False,
    )
    kw = dict(batch_size=args.batch_size, shuffle=False,
              num_workers=args.workers, pin_memory=True)
    if args.workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(ds, **kw)

    sums = dict(mse=0.0, mae=0.0, cosine=0.0, pearson=0.0)
    n = 0

    with torch.inference_mode():
        for bi, (x, paths) in enumerate(tqdm(loader, dynamic_ncols=True)):
            if args.max_batches and bi >= args.max_batches:
                break
            x = x.to(device, non_blocking=True).permute(1,0,2,3).unsqueeze(2)
            t = teacher(x).float()
            s = student(x).float()

            # Metrics over individual frame embeddings.
            tf = t.reshape(-1, t.shape[-1])
            sf = s.reshape(-1, s.shape[-1])

            mse = F.mse_loss(sf, tf).item()
            mae = F.l1_loss(sf, tf).item()
            cos = F.cosine_similarity(sf, tf, dim=-1).mean().item()

            tc = tf - tf.mean(dim=-1, keepdim=True)
            sc = sf - sf.mean(dim=-1, keepdim=True)
            pearson = (
                (tc * sc).sum(-1) /
                (tc.norm(dim=-1) * sc.norm(dim=-1)).clamp_min(1e-8)
            ).mean().item()

            bs = x.shape[1]
            n += bs
            sums["mse"] += mse * bs
            sums["mae"] += mae * bs
            sums["cosine"] += cos * bs
            sums["pearson"] += pearson * bs

    result = {k: v / max(n,1) for k,v in sums.items()}
    result["videos"] = n
    result["teacher_params"] = count_parameters(teacher)
    result["student_params"] = count_parameters(student)
    result["parameter_reduction_x"] = (
        result["teacher_params"] / result["student_params"]
    )

    print(json.dumps(result, indent=2))
    with open(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
