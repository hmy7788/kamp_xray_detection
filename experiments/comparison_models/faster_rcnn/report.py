"""검출 결과 기록(preds_*.json)으로 docs/experiments.md 의 보고 표 한 줄을 계산한다. 모델과 무관하다.

  python src/minyeop/faster_rcnn/report.py --preds-val runs/<이름>/<실험>/preds_val.json --preds-test runs/<이름>/<실험>/preds_test.json
  python src/minyeop/faster_rcnn/report.py --name 01_frcnn_r50fpn_min640_v1            # runs/minyeop/<name>/ 의 두 파일을 사용

- 신뢰도 임계값은 **val 에서 F1 이 최대인 값**으로 정해 test 에 그대로 적용한다 (--threshold 로 직접 지정 가능).
- 기록 형식은 metrics.py 첫머리의 설명을 보세요. 추론 시간은 기록에 없으므로 --infer-ms 로 직접 넘긴다.
- 부트스트랩: 이미지를 중복 허용으로 다시 뽑아(기본 1,000회) 95% 구간을 구한다.
  연속 촬영본은 서로 닮아 이미지를 독립으로 보면 구간이 실제보다 좁게 나온다 (낙관적).
출력은 마크다운 표 줄이고, runs/.../report_<split>.json 으로도 저장한다 (--no-save 로 끔).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from dataset import DATA, ROOT  # noqa: E402


def per_image(recs, thr, iou_thr=0.5, center_r=None):
    """이미지별 (n_gt, TP, FP, FN) 과 AP 계산용 (신뢰도, TP 여부) 배열."""
    cnt, aps = [], []
    for r in recs:
        t, _, nm = metrics.match_image(r, thr, iou_thr, center_r)
        tp = int(t.sum())
        cnt.append((len(r["gt"]), tp, len(t) - tp, len(r["gt"]) - nm))
        t2, c2, _ = metrics.match_image(r, 0.001, iou_thr, center_r)
        aps.append((c2, t2))
    return np.array(cnt), aps


def ap_from(aps, idx, n_gt):
    if n_gt == 0:
        return 0.0
    conf = np.concatenate([aps[i][0] for i in idx]); tp = np.concatenate([aps[i][1] for i in idx])
    if not len(conf):
        return 0.0
    o = np.argsort(-conf); tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    k = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[k + 1] - mrec[k]) * mpre[k + 1]))


def summarize(cnt, aps, idx):
    idx = np.asarray(idx)
    g, tp, fp, fn = cnt[idx].sum(0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / g if g else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    has = cnt[idx][:, 0] > 0
    full = float((cnt[idx][has][:, 3] == 0).mean()) if has.any() else float("nan")  # 결함을 전부 찾은 이미지 비율
    return {"P": p, "R": r, "F1": f1, "TP": int(tp), "FP": int(fp), "FN": int(fn), "n_img": int(len(idx)), "n_box": int(g),
            "full_det": full, "FPPI": fp / len(idx) if len(idx) else 0.0, "AP50": ap_from(aps, idx, g)}


def bootstrap(cnt, aps, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    n = len(cnt); out = {"F1": [], "AP50": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = summarize(cnt, aps, idx)
        out["F1"].append(s["F1"]); out["AP50"].append(s["AP50"])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--preds-val"); ap.add_argument("--preds-test")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--infer-ms", type=float, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--center-r", type=float, default=None, help="평가 v2: IoU 대신 중심 거리(px) 기준으로 매칭 (임계값도 이 기준으로 val 에서 다시 정함)")
    ap.add_argument("--tag", default="", help="저장 파일 이름에 붙일 표시 (예: _v2 -> report_v2_test.json)")
    a = ap.parse_args()

    if a.name:
        run = ROOT / "runs" / a.owner / a.name
        a.preds_val = a.preds_val or str(run / "preds_val.json"); a.preds_test = a.preds_test or str(run / "preds_test.json")
    else:
        run = None
    val = json.loads(Path(a.preds_val).read_text(encoding="utf-8"))
    thr = a.threshold if a.threshold is not None else metrics.best_f1(val, center_r=a.center_r)["thr"]
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8"))}

    print(f"임계값 {thr} ({'지정값' if a.threshold is not None else 'val 에서 F1 최대'})\n")
    print("| 평가셋 | 구분 | 이미지 | 박스 | AP@0.5 (95% 구간) | P | R | F1 (95% 구간) | TP / FP / FN | 완전 검출률 | FPPI |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for split, path in (("val", a.preds_val), ("test", a.preds_test)):
        if not path or not Path(path).exists():
            continue
        recs = json.loads(Path(path).read_text(encoding="utf-8"))
        cnt, aps = per_image(recs, thr, center_r=a.center_r)
        groups = {"전체": np.arange(len(recs))}
        for src, label in (("official", "공식 라벨"), ("team", "팀 라벨")):
            ix = np.array([i for i, r in enumerate(recs) if man[r["name"]]["source"] == src])
            if len(ix):
                groups[label] = ix
        saved = {"threshold": thr, "split": split, "matching": f"center<={a.center_r}px" if a.center_r is not None else "IoU>=0.5"}
        for label, ix in groups.items():
            s = summarize(cnt, aps, ix)
            ci = bootstrap(cnt[ix], [aps[i] for i in ix], a.n_boot) if label == "전체" else None
            apc = f" ({ci['AP50'][0]:.3f}~{ci['AP50'][1]:.3f})" if ci else ""
            f1c = f" ({ci['F1'][0]:.3f}~{ci['F1'][1]:.3f})" if ci else ""
            print(f"| {split} | {label} | {s['n_img']} | {s['n_box']} | {s['AP50']:.3f}{apc} | {s['P']:.3f} | {s['R']:.3f} | "
                  f"{s['F1']:.3f}{f1c} | {s['TP']} / {s['FP']} / {s['FN']} | {s['full_det']:.3f} | {s['FPPI']:.3f} |")
            saved[label] = {**s, **({"ci95": ci} if ci else {})}
        if run and not a.no_save:
            (run / f"report{a.tag}_{split}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    if a.infer_ms is not None:
        print(f"\n추론 시간 {a.infer_ms} ms/장")


if __name__ == "__main__":
    main()
