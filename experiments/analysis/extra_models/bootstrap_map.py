"""val mAP50-95 의 부트스트랩 신뢰구간과, R50 과의 쌍체 차이 구간. 최종 모델 선정에서 '동률인가'를 객관적으로 판정하는 용도.

  python src/minyeop/extra_models/bootstrap_map.py --out runs/minyeop/09_extra_models_v1 [--b 400]

방법: val 이미지를 같은 번호로 복원추출(쌍체)해 모든 모델의 mAP50-95 를 다시 계산한다. 이미지 안의 매칭은 다른 이미지와 독립이라 이미지별로
IoU 0.5~0.95 의 (신뢰도, TP 여부)를 한 번만 만들어 두고 재추출 때는 이어 붙여 AP 만 다시 계산한다(metrics.ap 와 같은 계산).
전체 val 369장과 공식 라벨 이미지만(59장) 두 가지를 낸다. 이미지가 적어 구간이 넓다. 모델 선정은 val 로만 한다(test 는 쓰지 않는다).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn"))
import metrics  # noqa: E402

P = {"R50": "runs/minyeop/01_frcnn_r50fpn_min640_v1/preds_val.json", "YOLOv3-tiny": "runs/minyeop/02_yolov3tiny_img640_v1/preds_val.json",
     "MobileNetV3": "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1/preds_val.json", "YOLO26n": "runs/minyeop/09_extra_models_v1/yolo26n/preds_val.json",
     "RT-DETR-l": "runs/minyeop/09_extra_models_v1/rtdetr_l/preds_val.json", "D-FINE-N": "runs/minyeop/09_extra_models_v1/dfine_n/preds_val.json"}
IOUS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def ap_from(conf, tp, n_gt):
    if not len(conf) or not n_gt:
        return 0.0
    o = np.argsort(-conf, kind="stable")
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def per_image(recs):
    """이미지별·IoU별 (신뢰도, TP) 와 정답 수."""
    out = []
    for r in recs:
        row = []
        for i in IOUS:
            t, c, _ = metrics.match_image(r, 0.001, float(i))
            row.append((c, t))
        out.append((row, len(r["gt"])))
    return out


def boot_map(pi, idx):
    n_gt = sum(pi[j][1] for j in idx)
    aps = []
    for k in range(len(IOUS)):
        conf = np.concatenate([pi[j][0][k][0] for j in idx])
        tp = np.concatenate([pi[j][0][k][1] for j in idx])
        aps.append(ap_from(conf, tp, n_gt))
    return float(np.mean(aps))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    ap.add_argument("--b", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    recs = {m: json.loads((ROOT / p).read_text(encoding="utf-8")) for m, p in P.items()}
    names = [r["name"] for r in recs["R50"]]
    assert all([r["name"] for r in recs[m]] == names for m in recs), "이미지 순서가 모델마다 다릅니다"
    pi = {m: per_image(r) for m, r in recs.items()}
    subsets = {"전체 val": list(range(len(names))), "공식 라벨 이미지만": [i for i, n in enumerate(names) if man[n]["source"] == "official"]}
    res = {"설정": {"부트스트랩 횟수": a.b, "시드": a.seed, "방법": "val 이미지 복원추출(모델 간 쌍체), mAP50-95"}, "결과": {}}
    for sname, ids in subsets.items():
        rng = np.random.default_rng(a.seed)
        samples = [rng.integers(0, len(ids), len(ids)) for _ in range(a.b)]
        pt = {m: boot_map(pi[m], ids) for m in P}
        boots = {m: np.array([boot_map(pi[m], [ids[j] for j in s]) for s in samples]) for m in P}
        res["결과"][sname] = {"이미지": len(ids), "모델": {}}
        print(f"\n== {sname} ({len(ids)}장, B={a.b}) ==")
        for m in P:
            lo, hi = np.percentile(boots[m], [2.5, 97.5])
            d = boots[m] - boots["R50"]
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            res["결과"][sname]["모델"][m] = {"mAP50-95": pt[m], "구간": [float(lo), float(hi)], "R50 대비 차이": pt[m] - pt["R50"], "차이 구간": [float(dlo), float(dhi)],
                                          "R50보다 높은 비율": float((d > 0).mean())}
            top = max(P, key=lambda k: pt[k])
            dt = boots[m] - boots[top]
            tlo, thi = np.percentile(dt, [2.5, 97.5])
            res["결과"][sname]["모델"][m]["선두 대비"] = {"선두": top, "차이": pt[m] - pt[top], "구간": [float(tlo), float(thi)], "동률(구간이 0 포함)": bool(tlo <= 0 <= thi) or m == top}
            tag = "" if m == "R50" else f" | R50 대비 {pt[m] - pt['R50']:+.3f} (구간 {dlo:+.3f}~{dhi:+.3f}, 높을 확률 {(d > 0).mean() * 100:.0f}%)" + ("" if m == top else f" | 선두({top}) 대비 {pt[m] - pt[top]:+.3f} (구간 {tlo:+.3f}~{thi:+.3f}) {'동률' if tlo <= 0 <= thi else '유의하게 낮음'}")
            print(f"{m:12} {pt[m]:.3f} (구간 {lo:.3f}~{hi:.3f}){tag}", flush=True)
    (ROOT / a.out / "bootstrap_val_map.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장:", ROOT / a.out / "bootstrap_val_map.json")


if __name__ == "__main__":
    main()
