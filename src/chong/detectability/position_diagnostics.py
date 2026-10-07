"""Describe paired location changes without assigning unobserved physical causes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from core import Dataset, json_write


def descriptors(ds, samples):
    cache, out = {}, {}
    for s in samples:
        iid = s["background_id"]
        if iid not in cache:
            cache[iid] = ds.image(iid)
        im = cache[iid]
        x, y = s["x"], s["y"]
        patch = im[max(0, y-12):y+13, max(0, x-12):x+13].astype(float)
        median = float(np.median(patch))
        variation = float(np.median(np.abs(patch-median)))
        dx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)/8
        dy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)/8
        out[s["id"]] = {"median_brightness": median, "local_variation_MAD": variation,
                         "local_gradient_p90": float(np.percentile(np.hypot(dx, dy), 90))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--train-pilot", type=Path, required=True)
    a = ap.parse_args()
    record = json.loads((a.input / "record.json").read_text(encoding="utf-8"))
    ds = Dataset()
    train_samples = [json.loads(x) for x in (a.train_pilot / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(ds.rows[s["background_id"]].split != "train" for s in train_samples):
        raise ValueError("Descriptor cut points must come from train pilot only")
    train_desc = descriptors(ds, train_samples)
    fields = ["median_brightness", "local_variation_MAD", "local_gradient_p90"]
    cuts = {k: [float(v) for v in np.quantile([d[k] for d in train_desc.values()], [1/3, 2/3])] for k in fields}
    baseline = json.loads((a.input / "predictions_probes_gpu1024_fp32.json").read_text(encoding="utf-8"))
    desc = descriptors(ds, baseline)
    fixed_ids = [r["id"] for r in baseline]
    threshold = record["config"]["thresholds"][0]
    success = lambda r: r["after_conf"] >= threshold and r["before_conf"] < threshold and r["sham_conf"] < threshold
    base_hits = {r["id"]: success(r) for r in baseline}
    results = []
    for setting in record["config"]["settings"]:
        if not setting["map"]:
            continue
        rows = json.loads((a.input / f"predictions_probes_{setting['id']}.json").read_text(encoding="utf-8"))
        if [r["id"] for r in rows] != fixed_ids:
            raise ValueError("Settings did not use identical probe placements")
        groups = defaultdict(list)
        lost = gained = 0
        for row in rows:
            hit = success(row)
            lost += base_hits[row["id"]] and not hit
            gained += not base_hits[row["id"]] and hit
            for field in fields:
                category = int(np.searchsorted(cuts[field], desc[row["id"]][field], side="right"))
                groups[(field, category)].append((hit, row["background_burst"]))
        conditions = [{"field": f, "bin": ["low", "middle", "high"][b], "n": len(items),
                       "new_hit": sum(int(v[0]) for v in items), "bursts": len({v[1] for v in items}),
                       "fraction": sum(int(v[0]) for v in items)/len(items)} for (f, b), items in sorted(groups.items())]
        results.append({"setting": setting["id"], "lost_vs_1024": int(lost), "gained_vs_1024": int(gained), "conditions": conditions})
    json_write(a.input / "position_diagnostics.json", {"threshold": threshold, "train_cut_points": cuts,
               "results": results, "note": "Descriptive associations with observed image brightness/variation/gradient; no causal powder-density or package-thickness claim. Repeated probes share backgrounds and templates."})
    lines = ["# 위치 이동 실험의 관찰 조건", "", "같은 위치를 짝지어 기본 1024에서 성공했으나 다른 설정에서 놓친 경우를 셉니다. 배경 밝기·질감·경사도의 구간 기준은 train 합성 검수 영상에서 고정했습니다. 분말 두께나 충전 불량 등 물리 원인을 측정한 것은 아닙니다.", "", "| 설정 | 1024 성공 → 누락 | 1024 누락 → 성공 |", "|---|---:|---:|"]
    for r in results:
        lines.append(f'| {r["setting"]} | {r["lost_vs_1024"]} | {r["gained_vs_1024"]} |')
    lines.extend(["", "조건별 검출 수, 분모, 배경 촬영 묶음 수는 position_diagnostics.json에 있습니다. 합성 위치 수를 독립 관측 수로 보지 않습니다."])
    (a.input / "position_diagnostics.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(a.input / "position_diagnostics.json")


if __name__ == "__main__":
    main()
