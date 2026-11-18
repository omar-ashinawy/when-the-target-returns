# When the Target Returns

Code for *When the Target Returns: Frozen SAM Trackers Need the Right Box More Than a Cleaner Memory*
(MAVU workshop, WACV 2027).

- `fork.py`: the four-way fork at a target's reappearance for frozen SAM 3 and SAM 2.1. The copies
  keep propagating, clear the memory written while the target was away, prompt the correct box, or
  do both.
- `srr.py`: SRR@30 (IoU ≥ 0.5 held on 5 consecutive frames, starting within 30 visible frames of the
  return) and the sequence-level bootstrap interval.

## Requirements

PyTorch with CUDA (we used 2.8), [SAM 3](https://github.com/facebookresearch/sam3) for `SAM3`, and
[SAM 2](https://github.com/facebookresearch/sam2) with `sam2.1_hiera_large.pt` for `SAM21`.

## Use

```python
from fork import SAM3, fork
from srr import iou, srr, interval

tracker = SAM3()  # or SAM21("sam2.1_hiera_large.pt")

# frames[0] is where tracking starts from init_box; the target is gone on frames 1..19,
# is back on frame 20 with box_20, and every copy is followed to frame 53.
copies = fork(tracker, frames, init_box, [(1, 20, 53, box_20)])[0]

for name, rows in copies.items():
    seen = [(t, box) for t, box, _ in rows if visible[t]]
    print(name, srr([t for t, _ in seen], [iou(box, gt[t]) for t, box in seen]))
```

`fork` returns, for each event, every copy's box and presence score on the frames from the
reappearance on. In the paper, LaSOT-ext and UAV events start one frame before each disappearance,
and each VOT-LT sequence is tracked from its first frame with all its events passed in one call.
`interval(outcomes, sequence_ids)` gives a rate or a paired difference with its 95% interval.
