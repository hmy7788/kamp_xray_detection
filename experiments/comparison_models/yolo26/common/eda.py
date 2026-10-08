"""확정 데이터 진단. 숫자를 reports/eda_stats.json 과 표로 낸다. 분석가(profiler)는 이 출력만 근거로 reports/eda.md 를 쓴다.

사용: python common/eda.py
내용: 장비·월·출처별 사진 수, 묶음 수와 크기, 해상도, 라벨 상자 수·크기(장비별), 라벨링 진행률, 이웃 사진 상관(표본).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def q(v, ps=(0.1, 0.5, 0.9)):
    v = np.asarray(v, dtype=float)
    return [round(float(np.quantile(v, p)), 1) for p in ps] if len(v) else []


def main() -> None:
    df = kx.load_manifest()
    df["resolution"] = df["width"].astype(str) + "x" + df["height"].astype(str)
    df["has_label"] = [(kx.DATA / p).exists() for p in df.label_path]
    S: dict = {"n_images": int(len(df)), "manifest_sha256": kx.manifest_sha()}

    S["by_machine"] = df.groupby("machine").size().to_dict()
    S["by_month"] = df.groupby("month").size().to_dict()
    S["by_source"] = df.groupby("source").size().to_dict()
    S["by_machine_month"] = {f"m{m}_{mo}": int(n) for (m, mo), n in df.groupby(["machine", "month"]).size().items()}
    S["resolution"] = {f"m{m} {r}": int(n) for (m, r), n in df.groupby(["machine", "resolution"]).size().items()}
    b = df.groupby("burst_id").size()
    S["bursts"] = {"n": int(len(b)), "size_dist": {str(k): int(v) for k, v in b.value_counts().sort_index().items()}}

    # 라벨
    lab = df[df.has_label]
    S["labeling"] = {"official": int((lab.source == "official").sum()), "team": int((lab.source == "team").sum()),
                     "team_total": int((df.source == "team").sum()),
                     "by_labeler": {m: int(((lab.source == "team") & (lab.labeler == m)).sum()) for m in kx.MEMBERS},
                     "need_by_labeler": {m: int(((df.source == "team") & (df.labeler == m)).sum()) for m in kx.MEMBERS}}
    rows = []
    for r in lab.itertuples(index=False):
        for cx, cy, w, h in kx.read_yolo_labels(kx.DATA / r.label_path):
            rows.append(dict(image_id=r.image_id, machine=r.machine, source=r.source, cx=cx, cy=cy,
                             w_px=w * r.width, h_px=h * r.height))
    boxes = pd.DataFrame(rows)
    if len(boxes):
        per_img = boxes.groupby("image_id").size()
        S["boxes"] = {"n": int(len(boxes)),
                      "per_image": {str(k): int(v) for k, v in per_img.value_counts().sort_index().items()},
                      "side_px_p10_50_90": q(np.sqrt(boxes.w_px * boxes.h_px)),
                      "side_px_by_machine": {str(m): q(np.sqrt(g.w_px * g.h_px)) for m, g in boxes.groupby("machine")},
                      "side_px_by_source": {str(s): q(np.sqrt(g.w_px * g.h_px)) for s, g in boxes.groupby("source")},
                      "cx_p10_50_90": q(boxes.cx), "cy_p10_50_90": q(boxes.cy),
                      "edge_within_0.05": int(((boxes.cx < .05) | (boxes.cx > .95) | (boxes.cy < .05) | (boxes.cy > .95)).sum())}
        S["images_without_box"] = int((lab.image_id.map(per_img).fillna(0) == 0).sum())

    # 이웃 사진 상관 (묶음 안 연속 2장, 표본 60묶음)
    rng = np.random.default_rng(0)
    corr = []
    for bid in rng.choice(b[b >= 2].index, size=min(60, int((b >= 2).sum())), replace=False):
        g = df[df.burst_id == bid].sort_values("ts").head(2)
        a1 = np.asarray(Image.open(kx.DATA / g.iloc[0].image_path).convert("L"), float).ravel()
        a2 = np.asarray(Image.open(kx.DATA / g.iloc[1].image_path).convert("L"), float).ravel()
        if a1.size == a2.size:
            corr.append(float(np.corrcoef(a1, a2)[0, 1]))
    S["neighbor_corr_p10_50_90"] = q(corr, (0.1, 0.5, 0.9)) if corr else []
    # 밝기 (장비별, 표본 30장)
    br = {}
    for m, g in df.groupby("machine"):
        samp = g.sample(min(30, len(g)), random_state=0)
        vals = [np.asarray(Image.open(kx.DATA / p).convert("L"), float).mean() for p in samp.image_path]
        br[str(m)] = round(float(np.mean(vals)), 1)
    S["mean_brightness_by_machine"] = br

    kx.REPORTS.mkdir(exist_ok=True)
    kx.write_json(kx.REPORTS / "eda_stats.json", S)
    for k, v in S.items():
        print(f"{k}: {v}")
    print(f"저장: {kx.REPORTS / 'eda_stats.json'}")


if __name__ == "__main__":
    main()
