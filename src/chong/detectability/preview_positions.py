"""Show measured-image coordinates used by the synthetic position experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from core import Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.probes / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    ds = Dataset()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4), layout="constrained")
    for ax, m in zip(axes, ["1", "2", "3"]):
        example = next(r for r in rows if r["machine"] == m)
        iid = example["background_id"]
        im = ds.image(iid)
        h, w = im.shape
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        for cx, cy, bw, bh in ds.boxes(iid):
            ax.add_patch(Rectangle(((cx-bw/2)*w, (cy-bh/2)*h), bw*w, bh*h,
                                   edgecolor="#16bbff", facecolor="none", linewidth=1.5))
        points = {(r["x"], r["y"]) for r in rows if r["background_id"] == iid}
        xx, yy = zip(*sorted(points))
        ax.scatter(xx, yy, marker="+", s=70, color="#ff9b27", linewidths=1.4)
        ax.set_title(f"Machine {m}: {len(points)} tested locations\n{w} x {h} native pixels")
        ax.set_axis_off()
    fig.suptitle("One fixed probe is inserted at a time\nOrange: tested positions | Blue: existing targets (retained, excluded from new-hit score)", fontsize=13)
    output = a.probes / "images/placement_overview.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
