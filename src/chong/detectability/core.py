"""Read-only KAMP data access and empirical, display-domain probe synthesis.

No material identity, physical diameter, attenuation law, or production FPR is
inferred. A probe is a fixed signed residual extracted from a labelled target.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
EXPECTED_MANIFEST = "1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Dataset:
    """Use the existing kx path helpers; never open a test image or label."""
    def __init__(self):
        spec = importlib.util.spec_from_file_location("chong_kx", REPO / "src/chong/harness/common/kx.py")
        self.kx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.kx)
        data = REPO / "runs/chong/local/work/data_view"
        if not data.exists():
            raise RuntimeError("Run src/chong/detectability/prepare.ps1 first (read-only source data view).")
        for key, name in (("MANIFEST", "manifest.csv"), ("MANIFEST_SHA", "manifest.sha256"),
                          ("DATA_PASS", "PASS"), ("SPLIT_INFO", "split_info.json")):
            setattr(self.kx, key, data / name)
        self.kx.DATA = data
        if digest(self.kx.MANIFEST) != EXPECTED_MANIFEST:
            raise ValueError("Unexpected data version; original data will not be modified.")
        self.splits = self.kx.load_split(allow_test=False)
        frame = self.kx.load_manifest()
        self.rows = {r.image_id: r for r in frame[frame.split.isin(["train", "val"])].itertuples(index=False)}
        self.images, self.labels = self.kx.path_maps(self.rows)

    def image(self, iid):
        return np.asarray(Image.open(self.images[iid]).convert("L")).copy()

    def boxes(self, iid):
        return self.kx.read_yolo_labels(self.labels[iid])


def product_mask(im):
    """Estimated foreground only. This is not a measured package boundary."""
    rim = np.concatenate([im[:8].ravel(), im[-8:].ravel(), im[:, :8].ravel(), im[:, -8:].ravel()])
    level = float(np.median(rim))
    noise = float(np.median(np.abs(rim.astype(float) - level)))
    binary = (im < level - max(12.0, 4 * noise)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        raise ValueError("No product foreground found")
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = map(int, stats[index])
    if area < im.size * 0.08 or area > im.size * 0.85:
        raise ValueError("Unreliable product foreground")
    mask = (labels == index).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask.astype(bool), (x, y, w, h)


def extract_probe(im, box):
    """Estimate a compact local target residual, not a rectangular donor patch.

    The high-pass image selects a dark component near the annotation center.
    Harmonic interpolation fills its support for a local background estimate.
    The fixed residual is reused without scaling/rotation/contrast changes.
    Review the recovered background and residual before accepting a template.
    """
    h, w = im.shape
    cx, cy, bw, bh = box
    x, y = int(round(cx * w)), int(round(cy * h))
    r = max(10, int(np.ceil(max(bw*w, bh*h))) + 3)
    if x-r < 0 or y-r < 0 or x+r >= w or y+r >= h:
        raise ValueError("Donor too close to image boundary")
    patch = im[y-r:y+r+1, x-r:x+r+1].copy()
    hp = cv2.medianBlur(patch, 7).astype(float) - patch.astype(float)
    yy, xx = np.indices(patch.shape)
    # Search inside the central portion of the annotation, away from carrier edges.
    search = (np.abs(xx-r) <= max(2, bw*w*0.3)) & (np.abs(yy-r) <= max(2, bh*h*0.3))
    py, px = np.unravel_index(np.argmax(np.where(search, hp, -np.inf)), hp.shape)
    peak = float(hp[py, px])
    if peak < 6:
        raise ValueError("Insufficient separable target signal")
    binary = ((hp >= max(3, peak*0.22)) &
              (np.abs(xx-r) <= bw*w/2+1) & (np.abs(yy-r) <= bh*h/2+1)).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary)
    seed = int(labels[py, px])
    if seed == 0:
        raise ValueError("Missing target component")
    component = (labels == seed).astype(np.uint8)
    if component.sum() < 2:
        raise ValueError("Single-pixel candidate rejected")
    support = cv2.dilate(component, np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.where(support)
    if xs.max()-xs.min()+1 > bw*w+5 or ys.max()-ys.min()+1 > bh*h+5:
        raise ValueError("Target component is not compact")
    background = fill_support(patch, support)
    residual = np.where(support, patch.astype(float)-background, 0).astype(np.float32)
    x0, x1, y0, y1 = int(xs.min())-1, int(xs.max())+2, int(ys.min())-1, int(ys.max())+2
    signal = residual[y0:y1, x0:x1].copy()
    return {
        "signal": signal, "patch": patch, "background": background,
        "support": support, "peak": peak,
        "center_offset": [float(px-x0), float(py-y0)],
        "crop": [x0, y0, x1, y1], "donor_center": [x, y],
        "annotation_size": [float(bw*w), float(bh*h)],
        "dark_mass": float(-np.minimum(signal, 0).sum()),
        "bright_mass": float(np.maximum(signal, 0).sum()),
    }


def fill_support(patch, support):
    background = patch.astype(float).copy()
    background[support] = np.median(background[~support])
    for _ in range(250):
        smooth = cv2.filter2D(background, -1, np.array([[0,.25,0],[.25,0,.25],[0,.25,0.]]))
        delta = np.max(np.abs(smooth[support]-background[support]))
        background[support] = smooth[support]
        if delta < 0.002:
            break
    return background


def extract_sham(im, probe, boxes):
    """Same residual operation/mask on a nearby unlabelled source patch.

    This is an artifact control, not a certified clean production image.
    Pick by background similarity, without looking at detector predictions.
    """
    support = probe["support"]
    r = support.shape[0] // 2
    x, y = probe["donor_center"]
    fg, _ = product_mask(im)
    choices = []
    for dx, dy in ((2*r+4, 0), (-2*r-4, 0), (0, 2*r+4), (0, -2*r-4)):
        xx, yy = x+dx, y+dy
        if xx-r < 0 or yy-r < 0 or xx+r >= im.shape[1] or yy+r >= im.shape[0]:
            continue
        if intersects_labels(xx, yy, 2*r+1, 2*r+1, boxes, im.shape, margin=2):
            continue
        if not fg[yy-r:yy+r+1, xx-r:xx+r+1][support].all():
            continue
        patch = im[yy-r:yy+r+1, xx-r:xx+r+1]
        delta = np.abs(patch.astype(float)-probe["background"])[~support].mean()
        choices.append((float(delta), xx, yy, patch))
    if not choices:
        raise ValueError("No nearby labelled-target-free sham patch")
    _, xx, yy, patch = min(choices, key=lambda v: v[0])
    background = fill_support(patch, support)
    residual = np.where(support, patch.astype(float)-background, 0).astype(np.float32)
    x0, y0, x1, y1 = probe["crop"]
    return residual[y0:y1, x0:x1].copy(), [xx, yy], patch.copy()


def intersects_labels(x, y, pw, ph, boxes, shape, margin=7):
    h, w = shape
    for cx, cy, bw, bh in boxes:
        if abs(x-cx*w) < pw/2 + bw*w/2 + margin and abs(y-cy*h) < ph/2 + bh*h/2 + margin:
            return True
    return False


def inject(im, signal, x, y):
    """Add the exact residual; refuse clipping, wrapping, or size changes."""
    ph, pw = signal.shape
    x0, y0 = int(x)-pw//2, int(y)-ph//2
    if x0 < 0 or y0 < 0 or x0+pw > im.shape[1] or y0+ph > im.shape[0]:
        raise ValueError("Probe outside image")
    out = im.copy()
    values = im[y0:y0+ph, x0:x0+pw].astype(float) + signal
    if values.min() < 0 or values.max() > 255:
        raise ValueError("Insertion would clip the fixed signal")
    out[y0:y0+ph, x0:x0+pw] = np.rint(values).astype(np.uint8)
    return out, (x0, y0, x0+pw, y0+ph)


def hit_at(predictions, box):
    """Local center matching: unrelated original detections never count."""
    x0, y0, x1, y1 = box
    return max((p[4] for p in predictions if x0 <= (p[0]+p[2])/2 <= x1
                and y0 <= (p[1]+p[3])/2 <= y1), default=0.0)


def png(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).save(path)
