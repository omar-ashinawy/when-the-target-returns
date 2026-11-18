"""The four-way fork at a target's reappearance, for frozen SAM 3 and SAM 2.1.

At the frame where the target comes back, the tracker's state is copied into four copies that
differ only on that frame:

  native   keep propagating
  memory   clear the memory written while the target was away (keep the initial prompt and the
           last 5 frames before the disappearance that the tracker trusted), then propagate
  box      prompt the correct box
  both     clear the memory and prompt the box

After that frame every copy propagates normally. A copy's box is the tight box of its mask.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

COPIES = ("native", "memory", "box", "both")
KEEP_TRUSTED = 5
STORE = ("maskmem_features", "maskmem_pos_enc", "pred_masks", "pred_masks_high_res", "obj_ptr",
         "object_score_logits")


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def mask_box01(mask):
    """Tight box of mask > 0 as a normalized [1, 4] cxcywh tensor, or None for an empty mask."""
    mask = torch.as_tensor(mask).detach()
    mask = mask.reshape(mask.shape[-2:])
    pixels = (mask > 0).nonzero(as_tuple=False)
    if pixels.numel() == 0:
        return None
    height, width = mask.shape
    y1, x1 = pixels.amin(dim=0).float()
    y2, x2 = pixels.amax(dim=0).float() + 1.0
    return torch.stack([(x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height),
                        (x2 - x1) / width, (y2 - y1) / height]).reshape(1, 4)


def presence(output):
    return float(torch.sigmoid(output["object_score_logits"].detach().float()).reshape(-1)[0].cpu())


class Tracker:
    """Frame-by-frame driver of a frozen SAM video model with explicit control of its memory."""

    def reset(self, num_frames):
        self.memory = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self.num_frames = num_frames

    def _run(self, image, t, points):
        cond = points is not None
        x = self.preprocess(image)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, feats, pos, sizes = self.model._prepare_backbone_features(self.model.forward_image(x))
            out = self.model.track_step(frame_idx=t, is_init_cond_frame=cond, current_vision_feats=feats,
                                        current_vision_pos_embeds=pos, feat_sizes=sizes,
                                        point_inputs=points, mask_inputs=None, output_dict=self.memory,
                                        num_frames=self.num_frames, run_mem_encoder=True,
                                        **self.extra(x))
        stored = {k: out[k] for k in STORE if k in out}
        self.memory["cond_frame_outputs" if cond else "non_cond_frame_outputs"][t] = stored
        recent = self.memory["non_cond_frame_outputs"]
        for old in sorted(recent)[:-self.keep]:
            del recent[old]
        return stored

    def start(self, image, box):
        self.anchor = self._run(image, 0, self.points(box, image))
        return self.anchor

    def step(self, image, t):
        return self._run(image, t, None)

    def prompt(self, image, box, t):
        return self._run(image, t, self.points(box, image))

    def clear_memory(self, trusted):
        self.memory = {"cond_frame_outputs": {0: self.anchor}, "non_cond_frame_outputs": dict(trusted)}

    def snapshot(self):
        return {k: dict(v) for k, v in self.memory.items()}

    def restore(self, snapshot):
        self.memory = {k: dict(v) for k, v in snapshot.items()}


class SAM3(Tracker):
    image_size = 1008

    def __init__(self, checkpoint=None, device="cuda"):
        from sam3.model_builder import build_tracker, download_ckpt_from_hf
        model = build_tracker(apply_temporal_disambiguation=False, with_backbone=True)
        state = torch.load(checkpoint or download_ckpt_from_hf("sam3"), map_location="cpu",
                           weights_only=True)
        state = state.get("model", state)
        # The video checkpoint keeps the tracker under tracker.* and the shared backbone under
        # detector.backbone.*.
        weights = {}
        for key, value in model.state_dict().items():
            for source in (("detector." if key.startswith("backbone.") else "tracker.") + key, key):
                if source in state and state[source].shape == value.shape:
                    weights[key] = state[source]
                    break
        model.load_state_dict(weights, strict=False)
        model.bf16_context.__exit__(None, None, None)  # the predictor enters a global autocast
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = device
        self.keep = max(model.num_maskmem, model.max_obj_ptrs_in_encoder) + 8

    def reset(self, num_frames):
        super().reset(10 ** 9)

    def trusts(self, p):
        return p > 0.7

    def preprocess(self, image):
        x = torch.from_numpy(np.array(np.asarray(image), dtype=np.float32))
        if x.max() > 1.0 + 1e-6:
            x = x / 255.0
        x = F.interpolate(x.permute(2, 0, 1).contiguous()[None], size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        return ((x - 0.5) / 0.5).to(self.device)

    def extra(self, x):
        return {"image": x}

    def points(self, box, image):
        scale = torch.tensor([self.image_size / image.width, self.image_size / image.height] * 2)
        x1, y1, x2, y2 = (torch.as_tensor(box, dtype=torch.float32) * scale).tolist()
        return {"point_coords": torch.tensor([[[x1, y1], [x2, y2]]], device=self.device),
                "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=self.device)}

    def box(self, output, image):
        box01 = mask_box01(output["pred_masks_high_res"])
        if box01 is None:
            return None
        cx, cy, w, h = box01[0].float().unbind(-1)
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        scale = torch.tensor([image.width, image.height] * 2, dtype=torch.float32, device=xyxy.device)
        x1, y1, x2, y2 = (xyxy * scale).unbind(-1)
        clipped = torch.stack([x1.clamp(0, image.width), y1.clamp(0, image.height),
                               x2.clamp(0, image.width), y2.clamp(0, image.height)])
        return clipped.cpu().numpy().astype(float).tolist()


class SAM21(Tracker):
    def __init__(self, checkpoint, config="configs/sam2.1/sam2.1_hiera_l.yaml", device="cuda"):
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.utils.transforms import SAM2Transforms
        self.model = build_sam2_video_predictor(config, str(checkpoint), device=device, mode="eval")
        self.model.eval().requires_grad_(False)
        self.image_size = int(self.model.image_size)
        self.transform = SAM2Transforms(resolution=self.image_size, mask_threshold=0.0)
        self.device = device
        self.keep = max(self.model.num_maskmem, self.model.max_obj_ptrs_in_encoder) + 8

    def trusts(self, p):
        return p >= 0.5

    def preprocess(self, image):
        return self.transform(image).unsqueeze(0).to(device=self.device)

    def extra(self, x):
        return {}

    def points(self, box, image):
        corners = torch.as_tensor(box, dtype=torch.float32, device=self.device).reshape(1, 2, 2).clone()
        corners[..., 0] *= float(self.image_size) / float(image.width)
        corners[..., 1] *= float(self.image_size) / float(image.height)
        return {"point_coords": corners,
                "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=self.device)}

    def box(self, output, image):
        box01 = mask_box01(output["pred_masks_high_res"])
        if box01 is None:
            return None
        cx, cy, w, h = box01[0].float().cpu().numpy().astype(np.float64)
        x1 = min(max((cx - 0.5 * w) * image.width, 0.0), float(image.width))
        y1 = min(max((cy - 0.5 * h) * image.height, 0.0), float(image.height))
        x2 = min(max((cx + 0.5 * w) * image.width, x1), float(image.width))
        y2 = min(max((cy + 0.5 * h) * image.height, y1), float(image.height))
        return [float(x1), float(y1), float(x2), float(y2)]


def fork(tracker, frames, box, events, copies=COPIES):
    """Track ``frames`` from ``box`` on frames[0] and fork the four copies at every reappearance.

    frames   image paths; tracking starts on frames[0] from ``box`` (x1, y1, x2, y2, pixels)
    events   (disappear, reappear, end, reappear_box) per event, as indices into ``frames``:
             the target is absent from ``disappear`` to ``reappear - 1``, ``reappear_box`` is its
             box on ``reappear``, and every copy is followed to ``end``
    Returns one dict per event mapping each copy to [(frame, box or None, presence), ...] for the
    frames reappear..end.
    """
    tracker.reset(len(frames))
    forks_at = {}
    for i, event in enumerate(events):
        forks_at.setdefault(event[1], []).append(i)
    trusted, frozen, native = {}, {}, {}
    results = [{} for _ in events]
    for t in range(max(event[2] for event in events) + 1):
        for i, event in enumerate(events):
            if event[0] == t:
                frozen[i] = dict(trusted)
        if t in forks_at:
            snapshot = tracker.snapshot()
            for i in forks_at[t]:
                _, _, end, reappear_box = events[i]
                for copy in copies:
                    if copy == "native":
                        continue
                    tracker.restore(snapshot)
                    rows = []
                    for g in range(t, end + 1):
                        image = load_rgb(frames[g])
                        if g > t:
                            output = tracker.step(image, g)
                        else:
                            if copy in ("memory", "both"):
                                tracker.clear_memory(frozen[i])
                            output = (tracker.step(image, g) if copy == "memory"
                                      else tracker.prompt(image, reappear_box, g))
                        rows.append((g, tracker.box(output, image), presence(output)))
                    results[i][copy] = rows
            tracker.restore(snapshot)
        image = load_rgb(frames[t])
        output = tracker.start(image, box) if t == 0 else tracker.step(image, t)
        native[t] = (t, tracker.box(output, image), presence(output))
        if t > 0 and tracker.trusts(presence(output)):
            trusted[t] = output
            for old in sorted(trusted)[:-KEEP_TRUSTED]:
                del trusted[old]
    for i, (_, reappear, end, _) in enumerate(events):
        if "native" in copies:
            results[i]["native"] = [native[g] for g in range(reappear, end + 1)]
    return results
