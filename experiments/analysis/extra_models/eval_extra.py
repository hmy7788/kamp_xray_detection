"""6개 모델(우리 3개 + 팀원 3개)을 같은 기준으로 평가한다: 실제 val·test, 합성 점(data_synth/test).

  python src/minyeop/extra_models/predict_extra.py --out runs/minyeop/09_extra_models_v1     # 먼저 팀원 모델 예측
  python src/minyeop/extra_models/eval_extra.py --out runs/minyeop/09_extra_models_v1

- 임계값: 모델마다 val 에서 F1 최대인 값(IoU 0.5, 동률이면 가운데). 평가 v2 용은 중심 거리 5px 기준으로 val 에서 다시 고른다.
- 실제 test: P/R/F1(IoU 0.5), AP50, mAP50-95, 공식 라벨만, v2(R=5)와 R=2 의 F1. 합성 점: eval_synth.py 와 같은 검출률(중심 5px)·대비 구간·호기 구간.
- 우리 3개 모델의 예측은 runs/minyeop/{01,02,04}_*/preds_{val,test}.json, 합성 점은 08_synth_eval_v1/summary.json 에서 가져온다(같은 임계값).
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn"))
import metrics  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_synth", ROOT / "src" / "minyeop" / "synth_eval" / "eval_synth.py")
ES = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ES)

OURS = {"Faster R-CNN R50-FPN": "runs/minyeop/01_frcnn_r50fpn_min640_v1", "YOLOv3-tiny": "runs/minyeop/02_yolov3tiny_img640_v1",
        "Faster R-CNN MobileNetV3-FPN": "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1"}
EXTRA = {"YOLO26n": "yolo26n", "RT-DETR-l": "rtdetr_l", "D-FINE-N": "dfine_n"}
IOUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    a = ap.parse_args()
    out = ROOT / a.out
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    preds = {}
    for m, run in OURS.items():
        preds[m] = {s: load(ROOT / run / f"preds_{s}.json") for s in ("val", "test")}
    for m, d in EXTRA.items():
        preds[m] = {s: load(out / d / f"preds_{s}.json") for s in ("val", "test", "synth")}

    # --- 실제 val·test
    real = {}
    for m, p in preds.items():
        b1 = metrics.best_f1(p["val"])
        thr = float(b1["thr"]) if "thr" in b1 else None
        b2 = metrics.best_f1(p["val"], center_r=5.0)
        thr2 = float(b2["thr"]) if "thr" in b2 else None
        if thr is None:   # best_f1 의 반환 키 이름이 다르면 임계값을 직접 찾는다
            grid = np.round(np.arange(0.05, 0.99, 0.01), 2)
            rows = [(t, metrics.prf_at(p["val"], t)["F1"]) for t in grid]
            top = max(f for _, f in rows)
            c = [t for t, f in rows if f >= top - 1e-12]
            thr = float(c[len(c) // 2])
            rows2 = [(t, metrics.prf_at(p["val"], t, center_r=5.0)["F1"]) for t in grid]
            top2 = max(f for _, f in rows2)
            c2 = [t for t, f in rows2 if f >= top2 - 1e-12]
            thr2 = float(c2[len(c2) // 2])
        te = p["test"]
        off = [r for r in te if man[r["name"]]["source"] == "official"]
        pt = metrics.prf_at(te, thr)
        real[m] = {"임계값(val, IoU0.5)": thr, "val F1": metrics.prf_at(p["val"], thr)["F1"],
                   "test P": pt["P"], "test R": pt["R"], "test F1": pt["F1"], "TP/FP/FN": [pt["TP"], pt["FP"], pt["FN"]],
                   "공식 라벨만 F1": metrics.prf_at(off, thr)["F1"],
                   "AP50": metrics.ap(te), "mAP50-95": float(np.mean([metrics.ap(te, i) for i in IOUS])),
                   "임계값(val, v2 R=5)": thr2, "test F1 v2 R=5": metrics.prf_at(te, thr2, center_r=5.0)["F1"],
                   "test F1 v2 R=2": metrics.prf_at(te, thr2, center_r=2.0)["F1"]}
        r = real[m]
        print(f"{m:30} thr {thr:.2f} | test F1 {r['test F1']:.3f} (P {r['test P']:.3f} R {r['test R']:.3f}, TP/FP/FN {r['TP/FP/FN']}) 공식 {r['공식 라벨만 F1']:.3f} | AP50 {r['AP50']:.3f} mAP50-95 {r['mAP50-95']:.3f} | v2 thr {thr2:.2f} R5 {r['test F1 v2 R=5']:.3f} R2 {r['test F1 v2 R=2']:.3f}", flush=True)

    # --- 합성 점
    droot = ROOT / "data_synth" / "test"
    rows = list(csv.DictReader(open(droot / "dots.csv", encoding="utf-8")))
    cnt = {}
    for r in rows:
        r["idx"] = cnt.get(r["image"], 0)
        cnt[r["image"]] = r["idx"] + 1
        r["x"], r["y"], r["s"], r["f"], r["contrast"] = int(r["x"]), int(r["y"]), float(r["s"]), float(r["f"]), float(r["contrast"])
    dots = {}
    for r in rows:
        dots.setdefault(r["image"], []).append((r["x"], r["y"]))
    n_img = len(dots)
    old = json.loads((ROOT / "runs/minyeop/08_synth_eval_v1/summary.json").read_text(encoding="utf-8"))["결과"]
    synth = {m: old[m] for m in OURS}
    bin_names = ["<6", "6~10", "10~14", "14~20", "≥20"]
    for m in EXTRA:
        thr = real[m]["임계값(val, IoU0.5)"]
        by_name = {r["name"]: r for r in preds[m]["synth"]}
        res = {}
        for r_px in (5.0, 2.0):
            hit, fa = ES.evaluate(dots, by_name, thr, r_px)
            res[f"R{r_px:g}"] = {"전체": ES.rate(rows, hit, m), "오경보 검출 수": fa, "FPPI": fa / n_img}
            if r_px == 5.0:
                res["R5"]["진하기 s"] = {str(s): ES.rate([r for r in rows if r["s"] == s], hit, m) for s in sorted({r["s"] for r in rows})}
                res["R5"]["크기 f"] = {str(f): ES.rate([r for r in rows if r["f"] == f], hit, m) for f in sorted({r["f"] for r in rows})}
                res["R5"]["호기"] = {k: ES.rate([r for r in rows if r["machine"] == k], hit, m) for k in "123"}
                res["R5"]["대비"] = {n: ES.rate([r for r in rows if lo <= r["contrast"] < hi], hit, m) for n, (lo, hi) in zip(bin_names, ES.BINS)}
                res["R5"]["호기별 s=1.0·f=1.0"] = {k: ES.rate([r for r in rows if r["machine"] == k and r["s"] == 1.0 and r["f"] == 1.0], hit, m) for k in "123"}
                res["R5"]["호기별 대비 14 이상"] = {k: ES.rate([r for r in rows if r["machine"] == k and r["contrast"] >= 14], hit, m) for k in "123"}
        synth[m] = res
    # 우리 3개 모델도 같은 형식으로 호기별 s=1.0 / 대비 14 이상 추가 (per_dot.csv 사용)
    pd_rows = list(csv.DictReader(open(ROOT / "runs/minyeop/08_synth_eval_v1/per_dot.csv", encoding="utf-8")))
    for m in OURS:
        sub = [r for r in pd_rows if r["모델"] == m]
        def rt(cond):
            s = [r for r in sub if cond(r)]
            k = sum(int(r["검출(R=5)"]) for r in s)
            lo, hi = ES.wilson(k, len(s))
            return {"점 수": len(s), "검출": k, "검출률": k / len(s), "구간": [lo, hi]}
        synth[m]["R5"]["호기별 s=1.0·f=1.0"] = {k: rt(lambda r, k=k: r["호기"] == k and r["s"] == "1.0" and r["f"] == "1.0") for k in "123"}
        synth[m]["R5"]["호기별 대비 14 이상"] = {k: rt(lambda r, k=k: r["호기"] == k and float(r["대비"]) >= 14) for k in "123"}
    print(f"\n== 합성 점 (이미지 {n_img}장, 점 {len(rows)}개, 중심 5px) ==")
    for m in list(OURS) + list(EXTRA):
        t = synth[m]["R5"]
        print(f"{m:30} 전체 {t['전체']['검출률']*100:5.1f}% (R=2 {synth[m]['R2']['전체']['검출률']*100:5.1f}%) 오경보 {synth[m]['R5']['오경보 검출 수']:>3}개 | s: " +
              " ".join(f"{v['검출률']*100:5.1f}" for v in t["진하기 s"].values()) + " | 대비: " + " ".join(f"{v['검출률']*100:5.1f}" for v in t["대비"].values()) +
              " | s=1·f=1 호기별: " + " ".join(f"{t['호기별 s=1.0·f=1.0'][k]['검출률']*100:5.1f}" for k in "123") + " | 대비≥14 호기별: " + " ".join(f"{t['호기별 대비 14 이상'][k]['검출률']*100:5.1f}" for k in "123"), flush=True)
    (out / "summary.json").write_text(json.dumps({"실제 val·test": real, "합성 점": synth, "설정": {"합성 이미지": n_img, "합성 점": len(rows),
                                                  "dataset_sha256": (droot / "dataset.sha256").read_text().strip()}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장:", out / "summary.json")


if __name__ == "__main__":
    main()
