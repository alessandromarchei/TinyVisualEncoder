#!/usr/bin/env python3
import torch
import torch.nn.functional as F


def kd_loss(student, teacher, huber_weight=1.0, cosine_weight=0.5,
            feature_mean=None, feature_std=None, eps=1e-6):
    """
    student/teacher: [T,B,D]

    Optional teacher feature statistics standardize BOTH tensors in the
    teacher coordinate system before regression.
    """
    if feature_mean is not None:
        mean = feature_mean.view(1, 1, -1)
        std = feature_std.clamp_min(eps).view(1, 1, -1)
        student_n = (student - mean) / std
        teacher_n = (teacher - mean) / std
    else:
        student_n, teacher_n = student, teacher

    huber = F.smooth_l1_loss(student_n, teacher_n)
    cosine = 1.0 - F.cosine_similarity(
        student_n, teacher_n, dim=-1, eps=eps
    ).mean()
    total = huber_weight * huber + cosine_weight * cosine

    with torch.no_grad():
        mse = F.mse_loss(student_n, teacher_n)
        cos_sim = F.cosine_similarity(
            student_n, teacher_n, dim=-1, eps=eps
        ).mean()

    return total, {
        "huber": huber.detach(),
        "mse": mse.detach(),
        "cosine_similarity": cos_sim.detach(),
    }
