"""Cheap, conservative product ROI; no learned detector or extra inference."""
import numpy as np

from core import product_mask


def crop_box(image, padding_fraction=.12):
    h, w = image.shape
    try:
        _, (x, y, rw, rh) = product_mask(image)
        pad = max(12, int(np.ceil(max(rw, rh)*padding_fraction)))
        return max(0, x-pad), max(0, y-pad), min(w, x+rw+pad), min(h, y+rh+pad)
    except ValueError:
        return 0, 0, w, h


def predict_roi(model, image, setting, predict, box=None):
    box = crop_box(image) if box is None else box
    x0, y0, x1, y1 = box
    predictions, speed = predict(model, image[y0:y1, x0:x1], setting)
    return [[x0+a, y0+b, x0+c, y0+d, confidence] for a,b,c,d,confidence in predictions], speed
