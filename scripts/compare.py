"""6단계: 모델 비교표 생성과 재현 확인.

    python scripts/compare.py                        # outputs/ 의 결과로 비교표 → outputs/model_comparison.md, .csv
    python scripts/compare.py --root results         # 제출 시점 결과(results/)로 비교표
    python scripts/compare.py --check                # outputs/ (재현 실행) 와 results/ (제출 결과) 의 test 지표 비교

- 최종 모델·베이스라인: <root>/<model>/eval_report_test.json, speed.json (이 저장소의 공통 파이프라인 결과)
- 비교 모델 4종 (Faster R-CNN 2종, YOLO26n, RT-DETR-l): results/comparison_models/six_models/summary.json
  (같은 분할·같은 채점 규칙으로 팀이 채점한 기록. 코드는 experiments/comparison_models/ 참고)
"""
import _bootstrap  # noqa: F401

import argparse
import csv
from pathlib import Path

from kamp_xray.common import OUTPUTS, RESULTS, load_json

MAIN = [("yolov3_tiny", "YOLOv3-tiny (베이스라인)"), ("dfine_n", "D-FINE-N (최종 모델)")]
OTHERS = [("Faster R-CNN R50-FPN", "Faster R-CNN (ResNet-50 FPN)"), ("Faster R-CNN MobileNetV3-FPN", "Faster R-CNN (MobileNetV3 FPN)"),
          ("YOLO26n", "YOLO26n"), ("RT-DETR-l", "RT-DETR-l")]
KEYS = ["map50", "map50_95", "precision", "recall", "f1", "threshold", "fps_cpu4"]


def main_rows(root):
    rows = []
    for m, name in MAIN:
        rep = root / m / "eval_report_test.json"
        if not rep.exists():
            continue
        r = load_json(rep)
        o = r["overall"]
        sp = root / m / "speed.json"
        fps = load_json(sp)["results"].get("cpu4", {}).get("fps") if sp.exists() else None
        rows.append({"model": name, "map50": o["ap50"], "map50_95": o["map50_95"], "precision": o["precision"],
                     "recall": o["recall"], "f1": o["f1"], "threshold": round(r["threshold"], 4), "fps_cpu4": fps,
                     "tp_fp_fn": f"{o['tp']}/{o['fp']}/{o['fn']}", "source": f"{root.name}/{m}/eval_report_test.json"})
    return rows


def other_rows():
    p = RESULTS / "comparison_models" / "six_models" / "summary.json"
    if not p.exists():
        return []
    real = list(load_json(p).values())[0]
    rows = []
    for key, name in OTHERS:
        r = real.get(key)
        if not r:
            continue
        g = lambda *names: next((r[n] for n in names if n in r), None)
        rows.append({"model": name, "map50": round(g("AP50"), 4), "map50_95": round(g("mAP50-95"), 4),
                     "precision": round(g("test P"), 4), "recall": round(g("test R"), 4), "f1": round(g("test F1"), 4),
                     "threshold": g("임계값(val, IoU0.5)"), "fps_cpu4": None, "tp_fp_fn": "/".join(map(str, g("TP/FP/FN"))),
                     "source": "results/comparison_models/six_models/summary.json"})
    return rows


def write_table(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    cols = ["model", *KEYS, "tp_fp_fn", "source"]
    with open(out / "model_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: "-" if v is None else (f"{v:.4f}" if isinstance(v, float) and v < 1.0001 else str(v))
    lines = ["| 모델 | mAP50 | mAP50-95 | Precision | Recall | F1 | 임계값(val) | FPS (CPU 4스레드) | TP/FP/FN |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | " + " | ".join(fmt(r[k]) for k in KEYS) + f" | {r['tp_fp_fn']} |")
    lines += ["", "- test 396장, 정답 상자 663개. 임계값은 모델별로 val F1 최대값으로 정해 test에 그대로 적용.",
              "- 출처: " + "; ".join(sorted({r['source'] for r in rows}))]
    (out / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\n  →", out / "model_comparison.md")


def check():
    """재현 실행(outputs/)과 제출 결과(results/)의 test 지표 차이."""
    ok = True
    for m, name in MAIN:
        a, b = OUTPUTS / m / "eval_report_test.json", RESULTS / m / "eval_report_test.json"
        if not (a.exists() and b.exists()):
            print(f"{name}: 비교할 파일이 없습니다 ({a if not a.exists() else b})")
            ok = False
            continue
        ra, rb = load_json(a), load_json(b)
        print(f"{name}: 임계값 재현 {ra['threshold']:.4f} / 제출 {rb['threshold']:.4f}")
        for k in ("ap50", "map50_95", "precision", "recall", "f1", "tp", "fp", "fn"):
            va, vb = ra["overall"][k], rb["overall"][k]
            same = abs(va - vb) <= (0 if isinstance(va, int) else 1e-4)
            ok &= same
            print(f"    {k:10} 재현 {va:<8} 제출 {vb:<8} {'일치' if same else '차이'}")
    print("모두 일치" if ok else "차이가 있습니다 (직접 학습한 가중치라면 GPU 비결정성으로 조금 다를 수 있음)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", choices=["outputs", "results"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    root = OUTPUTS if a.root == "outputs" else RESULTS
    write_table(main_rows(root) + other_rows(), root)


if __name__ == "__main__":
    main()
