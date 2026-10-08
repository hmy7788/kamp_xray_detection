"""실험 폴더를 채점해 metrics.json 과 errors/*.csv 를 만든다 (docs/contracts/evaluation.md).

사용: python common/evaluate.py runs/<exp_id> [--target-recall 0.99]        검증(val) 예측 preds_val.csv 채점
      python common/evaluate.py runs/<exp_id> --final                       테스트 예측 preds_test.csv 채점 (reproduce.py 만 쓴다)

시드가 여럿이면 시드별로 지표를 내고 평균·표준편차를 적는다. 임계값은 시드 전부의 예측에서 상자 F1 이 최대인 값 하나.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402
import metrics as M  # noqa: E402


def load_preds(csv: Path) -> dict[int, dict[str, list]]:
    """seed -> {image_id: [(cx,cy,w,h,conf)]}. 빈 행은 빈 리스트."""
    df = pd.read_csv(csv, dtype={"image_id": str})
    out: dict[int, dict[str, list]] = {}
    for r in df.itertuples(index=False):
        d = out.setdefault(int(r.seed), {})
        lst = d.setdefault(r.image_id, [])
        if pd.notna(r.conf):
            lst.append((float(r.cx), float(r.cy), float(r.w), float(r.h), float(r.conf)))
    return out


def load_gt(ids) -> dict[str, list]:
    _, labs = kx.path_maps(ids)
    return {i: kx.read_yolo_labels(labs[i]) for i in ids}


def stat(vals):
    v = [float(x) for x in vals]
    return {"mean": float(np.mean(v)) if v else 0.0, "std": float(np.std(v)) if v else 0.0, "seeds": v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    rd = Path(a.run_dir)
    cfg = kx.read_yaml(rd / "config.yaml")

    if a.final:
        split = kx.load_split(allow_test=True)
        ids, csv, part = split["test"], rd / "preds_test.csv", "test"
    else:
        split = kx.load_split()
        ids, csv, part = split["val"], rd / "preds_val.csv", "val"
    if not csv.exists():
        print(f"FAIL: {csv} 없음")
        return 1
    gt = load_gt(ids)
    preds = {s: {i: p.get(i, []) for i in ids} for s, p in load_preds(csv).items()}
    seeds = sorted(preds)

    # 임계값: 시드 전부의 예측에서 상자 F1 최대, 0.05 단위
    thr = float(np.mean([M.best_threshold(gt, preds[s]) for s in seeds]))
    thr = float(np.round(thr * 20) / 20)

    per_seed = collections.defaultdict(list)   # metric -> [seed values]
    for s in seeds:
        p = preds[s]
        per_seed["map50"].append(M.map50(gt, p))
        per_seed["map50_95"].append(M.map50_95(gt, p))
        tp, fp, fn, _, _ = M.match(gt, p, thr, 0.5)
        pr, rc, f1 = M.prf(tp, fp, fn)
        per_seed["box_precision"].append(pr); per_seed["box_recall"].append(rc); per_seed["box_f1"].append(f1)
        tp, fp, fn, _, _ = M.match(gt, p, thr, 0.5, mode="center")
        pr, rc, _ = M.prf(tp, fp, fn)
        per_seed["center_hit_precision"].append(pr); per_seed["center_hit_recall"].append(rc)
        itp, ifp, ifn, itn = M.image_level(gt, p, thr)
        per_seed["img_f1"].append(M.prf(itp, ifp, ifn)[2]); per_seed["img_recall"].append(M.prf(itp, ifp, ifn)[1])
    overall = {m: stat(v) for m, v in per_seed.items()}

    # 조건별 재현율 (시드별 맞춤 여부 평균). 오류 목록은 첫 시드 기준
    cond = pd.read_csv(kx.CONDITIONS_CSV, dtype={"image_id": str}) if kx.CONDITIONS_CSV.exists() else None
    spec = kx.read_yaml(kx.COMMON / "conditions.yaml")
    by_condition = {}
    missed_rows, fp_rows = [], []
    if cond is not None:
        cond = cond[cond.image_id.isin(gt)]
        hit_iou = collections.defaultdict(list)
        hit_ctr = collections.defaultdict(list)
        for s in seeds:
            p = preds[s]
            _, _, _, m_iou, fps = M.match(gt, p, thr, 0.5)
            _, _, _, m_ctr, _ = M.match(gt, p, thr, 0.5, mode="center")
            for r in cond.itertuples(index=False):
                key = (r.image_id, int(r.box_idx))
                hit_iou[key].append(key in m_iou)
                hit_ctr[key].append(key in m_ctr)
            if s == seeds[0]:
                missed_rows = [r._asdict() for r in cond.itertuples(index=False) if (r.image_id, int(r.box_idx)) not in m_iou]
                fp_rows = [dict(image_id=i, cx=p_[0], cy=p_[1], w=p_[2], h=p_[3], conf=p_[4]) for i, p_ in fps]
        cond = cond.assign(hit_iou=[np.mean(hit_iou[(r.image_id, int(r.box_idx))]) for r in cond.itertuples(index=False)],
                           hit_ctr=[np.mean(hit_ctr[(r.image_id, int(r.box_idx))]) for r in cond.itertuples(index=False)])
        for axis in spec["axes"]:
            if axis not in cond.columns:
                continue
            by_condition[axis] = {}
            for b, g in cond.groupby(axis):
                by_condition[axis][str(b)] = {"box_recall": float(g.hit_iou.mean()),
                                              "center_hit_recall": float(g.hit_ctr.mean()), "n": int(len(g))}

    sweep, safety = M.threshold_sweep(gt, preds[seeds[0]], a.target_recall)

    out = {
        "exp_id": cfg["exp_id"], "part": part,
        "threshold": thr, "n_seeds": len(seeds), "n_images": len(gt),
        "n_boxes": sum(len(v) for v in gt.values()),
        "overall": overall, "by_condition": by_condition,
        "threshold_sweep": sweep, "safety": safety,
        "evaluated_at": kx.now_iso(),
    }
    name = "metrics_test.json" if a.final else "metrics.json"
    kx.write_json(rd / name, out)
    (rd / "errors").mkdir(exist_ok=True)
    pd.DataFrame(missed_rows).to_csv(rd / "errors" / "missed.csv", index=False, lineterminator="\n")
    pd.DataFrame(fp_rows).to_csv(rd / "errors" / "false_positives.csv", index=False, lineterminator="\n")
    o = overall
    print(f"{cfg['exp_id']} [{part}]: map50 {o['map50']['mean']:.3f}±{o['map50']['std']:.3f} | "
          f"box_recall {o['box_recall']['mean']:.3f} | center_hit_recall {o['center_hit_recall']['mean']:.3f} | "
          f"img_f1 {o['img_f1']['mean']:.3f} | thr {thr} | safety {safety}")
    print(f"저장: {rd / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
