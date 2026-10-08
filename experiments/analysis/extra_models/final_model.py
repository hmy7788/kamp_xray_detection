"""최종 모델(D-FINE-N)의 임계값별 운영 분석: 실제 test 신뢰도 분포, 합성 점 검출률, 가짜 정상 오경보를 임계값별로 낸다.

  python src/minyeop/extra_models/final_model.py --work-dir <임시 폴더> --out runs/minyeop/09_extra_models_v1

3단 판정(불합격 / 재검사 / 합격)의 경계를 정할 근거 수치를 만든다.
  (a) 실제 test: 결함 있는 369장과 빈 라벨 27장의 이미지 최고 신뢰도 분포
  (b) 합성 점(작대기 안, 시드 42): 대비 9~18(옅은 점)의 검출률과 이미지당 오경보 검출 수
  (c) 가짜 정상(평균 보간 6, NS+노이즈 6): 오경보 이미지 비율
주의: (b)(c)는 합성·가짜 정상이라 참고용이다. 임계값은 val 에서 정한 값(0.66)이 기본이고 이 분석으로 바꾸지 않는다(분석용).
"""
import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "dot_removal"))
THRS = [0.3, 0.5, 0.66, 0.8, 0.9]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ctr(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def mx(r):
    return max([d[4] for d in r["dets"]], default=0.0)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    a = ap.parse_args()
    out = ROOT / a.out
    px = load_module("predict_extra", ROOT / "src" / "minyeop" / "extra_models" / "predict_extra.py")
    rd = load_module("remove_dot", ROOT / "src" / "minyeop" / "dot_removal" / "remove_dot.py")
    val_thr = json.loads((out / "summary.json").read_text(encoding="utf-8"))["실제 val·test"]["D-FINE-N"]["임계값(val, IoU0.5)"]
    res = {"val 임계값": val_thr, "임계값 목록": THRS}

    # (a) 실제 test
    te = json.loads((out / "dfine_n" / "preds_test.json").read_text(encoding="utf-8"))
    defect = np.array([mx(r) for r in te if r["gt"]])
    empty = np.array([mx(r) for r in te if not r["gt"]])
    bands = {"결함 있음(369장)": defect, "빈 라벨(27장)": empty}
    res["실제 test 이미지 최고 신뢰도"] = {k: {"장수": len(v), "분위(0,5,25,50,75,100%)": [float(x) for x in np.percentile(v, [0, 5, 25, 50, 75, 100])],
                                        "임계값별 이상 장수": {str(t): int((v >= t).sum()) for t in THRS}} for k, v in bands.items()}
    print(f"(a) 실제 test 이미지 최고 신뢰도 (D-FINE-N, val 임계값 {val_thr})")
    for k, v in bands.items():
        print(f"  {k}: 최소 {v.min():.3f}, 5% {np.percentile(v, 5):.3f}, 중앙값 {np.median(v):.3f} | " + " ".join(f">={t}: {int((v >= t).sum())}장" for t in THRS))

    # (b) 합성 점
    dots = list(csv.DictReader(open(ROOT / "data_synth" / "test" / "dots.csv", encoding="utf-8")))
    by_img = {}
    for r in dots:
        by_img.setdefault(r["image"], []).append((int(r["x"]), int(r["y"]), float(r["contrast"])))
    sy = {r["name"]: r for r in json.loads((out / "dfine_n" / "preds_synth.json").read_text(encoding="utf-8"))}
    res["합성 점"] = {}
    print("\n(b) 합성 점 (R=5px)")
    for t in THRS:
        hit = tot = hit_f = tot_f = fa = 0
        for n, pts in by_img.items():
            cs = [ctr(d) for d in sy[n]["dets"] if d[4] >= t]
            for x, y, c in pts:
                ok = any(np.hypot(cx - x, cy - y) <= 5 for cx, cy in cs)
                tot += 1
                hit += ok
                if 9 <= c < 18:
                    tot_f += 1
                    hit_f += ok
            fa += sum(1 for cx, cy in cs if not any(np.hypot(cx - x, cy - y) <= 5 for x, y, _ in pts))
        res["합성 점"][str(t)] = {"전체 검출률": hit / tot, "옅은 점(대비 9~18) 검출률": hit_f / tot_f, "옅은 점 수": tot_f, "오경보 검출 수": fa, "이미지당 오경보": fa / len(by_img)}
        print(f"  thr {t}: 전체 {hit / tot * 100:5.1f}% | 옅은 점(대비 9~18, n={tot_f}) {hit_f / tot_f * 100:5.1f}% | 오경보 {fa}개 (이미지당 {fa / len(by_img):.3f})")

    # (c) 가짜 정상
    gt = {r["name"]: r["gt"] for r in te}
    names = sorted(gt)
    with_dot = [n for n in names if gt[n]]
    res["가짜 정상"] = {}
    spec = px.MODELS["dfine_n"]
    for method, half, noise in (("mean", 6, 0.0), ("ns", 6, 0.75)):
        tg = f"{method}_half{half}"
        work = Path(a.work_dir) / "final" / tg
        if work.exists():
            shutil.rmtree(work)
        (work / "images").mkdir(parents=True)
        rng = np.random.default_rng(0)
        for n in names:
            src = ROOT / "data/test/images" / n
            if gt[n]:
                im, *_ = rd.remove_dots(src, gt[n], half, rng, noise, method)
                Image.fromarray(im, "L").save(work / "images" / n)
            else:
                shutil.copy(src, work / "images" / n)
        paths = sorted((work / "images").glob("*.png"))
        dets = px.run_dfine(spec, px.load_images(paths), "cuda:0", 8)
        mxs = {p.name: max([d[4] for d in dd], default=0.0) for p, dd in zip(paths, dets)}
        v = np.array([mxs[n] for n in with_dot])
        res["가짜 정상"][tg] = {str(t): {"오경보 이미지": int((v >= t).sum()), "비율": float((v >= t).mean())} for t in THRS}
        print(f"\n(c) 가짜 정상 {tg}: " + " | ".join(f"thr {t}: {int((v >= t).sum())}/{len(v)} ({(v >= t).mean() * 100:.1f}%)" for t in THRS), flush=True)
    (out / "final_model_dfine.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장:", out / "final_model_dfine.json")


if __name__ == "__main__":
    main()
