"""scripts/synth_eval.py 가 만든 합성 점 평가 데이터(data_synth/<split>)를 세 모델로 평가한다.

  python scripts/synth_eval.py --split test --out data_synth/test       # 먼저 데이터 생성
  python src/minyeop/synth_eval/eval_synth.py --out runs/minyeop/08_synth_eval_v1

지표 (모델마다 val 에서 정한 임계값으로 고정)
  - 점 단위 검출률 = 합성 점 중 중심 R px 이내에 임계값 이상 검출이 있는 비율 (R=5 기본, R=2 도 함께). 95% 구간은 Wilson.
  - 오경보 = 어느 합성 점에도 R px 이내가 아닌 검출. 실제 점은 지웠으므로 정답 밖 검출은 모두 오경보다(FPPI = 이미지당 평균 수).
  - 진하기 s, 크기 f, 대비(둘레 밝기 - 점 밝기), 호기별로 나눠 본다. 대비 구간은 [0,6) [6,10) [10,14) [14,20) [20,~).
주의: 합성 점은 실제 점이 아니다(투과율 조각을 곱한 것). 이전 분석에서 실제와 약 10%p 차이, 사각 흔적이 있었다. 모델 비교의 상대적 해석에 쓴다.
검출 결과(preds)는 --work-dir 이 아니라 실험 폴더에 저장하지 않는다(정답 좌표가 들어 있음). dots.csv 기준으로 계산한 결과만 저장한다.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
FR, YO, MB = (ROOT / "src" / "minyeop" / p for p in ("faster_rcnn", "yolov3_tiny", "faster_rcnn_mobilenet"))
sys.path.insert(0, str(MB))
sys.path.insert(0, str(FR))

RUNS = {"Faster R-CNN R50-FPN": (ROOT / "runs/minyeop/01_frcnn_r50fpn_min640_v1", 0.95),
        "YOLOv3-tiny": (ROOT / "runs/minyeop/02_yolov3tiny_img640_v1", 0.06),
        "Faster R-CNN MobileNetV3-FPN": (ROOT / "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1", 0.93)}
BINS = [(-99, 6), (6, 10), (10, 14), (14, 20), (20, 999)]
BIN_NAMES = ["<6", "6~10", "10~14", "14~20", "≥20"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(dots, recs, thr, r):
    """점마다 검출 여부(bool 목록)와 오경보 수. dots: 이미지별 [(x, y)], recs: {이름: rec}."""
    hit, fa = {}, 0
    for name, pts in dots.items():
        dets = [d for d in recs[name]["dets"] if d[4] >= thr]
        cs = [((d[0] + d[2]) / 2, (d[1] + d[3]) / 2) for d in dets]
        for i, (x, y) in enumerate(pts):
            hit[(name, i)] = any(np.hypot(cx - x, cy - y) <= r for cx, cy in cs)
        fa += sum(1 for cx, cy in cs if not any(np.hypot(cx - x, cy - y) <= r for x, y in pts))
    return hit, fa


def rate(rows, hit, model_key):
    k = sum(1 for r in rows if hit[(r["image"], r["idx"])])
    lo, hi = wilson(k, len(rows))
    return {"점 수": len(rows), "검출": k, "검출률": k / len(rows) if rows else float("nan"), "구간": [lo, hi]}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_synth", help="data_synth/<split>/ 의 부모 폴더")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="runs/minyeop/08_synth_eval_v1")
    ap.add_argument("--r", type=float, default=5.0)
    a = ap.parse_args()
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"이미 존재하는 실험 폴더입니다(덮어쓰기 방지): {a.out}")
    out.mkdir(parents=True)
    droot = ROOT / a.data / a.split
    rows = list(csv.DictReader(open(droot / "dots.csv", encoding="utf-8")))
    seen = {}
    for r in rows:
        r["idx"] = seen.setdefault(r["image"], [])
        r["idx"] = len(r["idx"]); seen[r["image"]].append(1)
        r["x"], r["y"], r["s"], r["f"], r["contrast"] = int(r["x"]), int(r["y"]), float(r["s"]), float(r["f"]), float(r["contrast"])
    dots = {}
    for r in rows:
        dots.setdefault(r["image"], []).append((r["x"], r["y"]))
    n_img = len(dots)
    print(f"{a.split}: 합성 이미지 {n_img}장, 점 {len(rows)}개 (data_synth/{a.split}/dataset.sha256 = {(droot / 'dataset.sha256').read_text().strip()[:12]}…)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fr_r50 = load_module("fr_r50", FR / "predict.py")
    fr_mob = load_module("fr_mob", FR / "predict.py")
    from mobilenet_model import build_model as mob_build  # noqa: E402
    fr_mob.build_model = mob_build
    yo = load_module("yolo_predict", YO / "predict.py")
    from utils import torch_utils  # noqa: E402
    recs = {}
    recs["Faster R-CNN R50-FPN"] = fr_r50.predict_split(RUNS["Faster R-CNN R50-FPN"][0], a.split, "best", device, 4, data_dir=str(ROOT / a.data))[0]
    recs["Faster R-CNN MobileNetV3-FPN"] = fr_mob.predict_split(RUNS["Faster R-CNN MobileNetV3-FPN"][0], a.split, "best", device, 4, data_dir=str(ROOT / a.data))[0]
    ycfg = json.loads((RUNS["YOLOv3-tiny"][0] / "config.json").read_text(encoding="utf-8"))
    recs["YOLOv3-tiny"] = yo.predict_split(RUNS["YOLOv3-tiny"][0], a.split, "best", torch_utils.select_device("0"), ycfg["img_size"], data_root=str(ROOT / a.data))[0]

    summary = {"설정": {"split": a.split, "R(px)": a.r, "합성 이미지": n_img, "합성 점": len(rows), "dataset_sha256": (droot / "dataset.sha256").read_text().strip(), "임계값": {m: t for m, (_, t) in RUNS.items()}}, "결과": {}}
    per_dot = []
    for m, (run, thr) in RUNS.items():
        by_name = {r["name"]: r for r in recs[m]}
        res = {}
        for r_px in (a.r, 2.0):
            hit, fa = evaluate(dots, by_name, thr, r_px)
            tag = f"R{r_px:g}"
            res[tag] = {"전체": rate(rows, hit, m), "오경보 검출 수": fa, "FPPI": fa / n_img}
            if r_px == a.r:
                res[tag]["진하기 s"] = {str(s): rate([r for r in rows if r["s"] == s], hit, m) for s in sorted({r["s"] for r in rows})}
                res[tag]["크기 f"] = {str(f): rate([r for r in rows if r["f"] == f], hit, m) for f in sorted({r["f"] for r in rows})}
                res[tag]["호기"] = {k: rate([r for r in rows if r["machine"] == k], hit, m) for k in ("1", "2", "3")}
                res[tag]["대비"] = {n: rate([r for r in rows if lo <= r["contrast"] < hi], hit, m) for n, (lo, hi) in zip(BIN_NAMES, BINS)}
                res[tag]["진하기 x 크기"] = {f"s{s}_f{f}": rate([r for r in rows if r["s"] == s and r["f"] == f], hit, m) for s in sorted({r["s"] for r in rows}) for f in sorted({r["f"] for r in rows})}
                for r in rows:
                    per_dot.append([m, r["image"], r["machine"], r["x"], r["y"], r["s"], r["f"], r["contrast"], int(hit[(r["image"], r["idx"])])])
        summary["결과"][m] = res
        t = res[f"R{a.r:g}"]
        print(f"\n== {m} (임계값 {thr}) ==")
        print(f"  전체 검출률 R={a.r:g}: {t['전체']['검출']}/{t['전체']['점 수']} = {t['전체']['검출률']*100:.1f}% (구간 {t['전체']['구간'][0]*100:.1f}~{t['전체']['구간'][1]*100:.1f}) | R=2: {res['R2']['전체']['검출률']*100:.1f}% | 오경보 {t['오경보 검출 수']}개 (FPPI {t['FPPI']:.3f})")
        print("  진하기 s :", "  ".join(f"{k}: {v['검출률']*100:5.1f}%" for k, v in t["진하기 s"].items()))
        print("  크기 f   :", "  ".join(f"{k}: {v['검출률']*100:5.1f}%" for k, v in t["크기 f"].items()))
        print("  호기     :", "  ".join(f"{k}: {v['검출률']*100:5.1f}%" for k, v in t["호기"].items()))
        print("  대비     :", "  ".join(f"{k}({v['점 수']}): {v['검출률']*100:5.1f}%" for k, v in t["대비"].items()), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "per_dot.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["모델", "이미지", "호기", "x", "y", "s", "f", "대비", f"검출(R={a.r:g})"])
        w.writerows(per_dot)
    print("\n저장:", out)


if __name__ == "__main__":
    main()
