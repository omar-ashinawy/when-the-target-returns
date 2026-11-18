"""SRR@N: did the tracker get the returning target back and hold it?

An event counts as recovered at deadline N when IoU >= 0.5 holds on 5 consecutive frames and the run
starts within the first N visible frames after the return (the reappearance frame is the first).
Intervals are 95% percentile intervals from a bootstrap that resamples whole sequences.
"""
import numpy as np


def iou(box, gt):
    """IoU of two (x1, y1, x2, y2) boxes; a missing prediction (None) has IoU 0."""
    if box is None:
        return 0.0
    a, b = np.asarray(box, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    wh = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0.0)
    inter = float(wh[0] * wh[1])
    union = (float(np.prod(np.maximum(a[2:] - a[:2], 0.0))) + float(np.prod(np.maximum(b[2:] - b[:2], 0.0)))
             - inter)
    return inter / union if union > 0.0 else 0.0


def srr(frames, ious, deadline=30, hold=5, threshold=0.5):
    """frames: indices of the visible frames from the reappearance on, in order; ious: IoU on each.

    Returns True or False, or None when fewer than deadline + hold - 1 visible frames were observed.
    """
    frames, ious = np.asarray(frames, dtype=np.int64), np.asarray(ious, dtype=np.float64)
    if len(frames) < deadline + hold - 1:
        return None
    for start in range(deadline):
        run, overlap = frames[start:start + hold], ious[start:start + hold]
        if len(run) == hold and np.all(overlap >= threshold) and np.all(np.diff(run) == 1):
            return True
    return False


def interval(values, sequences, resamples=2000, seed=0):
    """Mean of per-event values (0/1 outcomes for a rate, paired differences for a gain) and its 95%
    interval from a bootstrap that draws whole sequences with replacement."""
    labels = {label: i for i, label in enumerate(sorted(set(sequences)))}
    codes = np.array([labels[s] for s in sequences])
    sums = np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=len(labels))
    counts = np.bincount(codes, weights=np.ones(len(codes)), minlength=len(labels))
    draws = np.random.default_rng(seed).integers(0, len(labels), size=(resamples, len(labels)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(sums.sum() / counts.sum()), (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
