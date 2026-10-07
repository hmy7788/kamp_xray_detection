"""가짜 정상 이미지에서의 이미지 단위 오경보 분석.

데이터에는 정상 제품 이미지가 사실상 없어서(빈 라벨 105장은 제품이 반만 찍힌 특수 사진), test 이미지의 결함 점을 보간으로 지운 이미지를
"가짜 정상"으로 보고 세 모델이 헛경보를 얼마나 내는지 센다. 점만 지우고 제품·배경은 그대로라서 같은 사진의 결함 있음/없음 쌍이 생긴다.

  python src/minyeop/fake_normal/false_alarm.py --work-dir <임시 폴더> --out runs/minyeop/05_fake_normal_v1

지표 (모델마다 val 에서 정한 임계값으로 고정, test 는 한 번 읽음)
  - 이미지 단위 오경보율 = 가짜 정상 이미지 중 임계값 이상 검출이 하나라도 있는 이미지 비율 (특이도 = 1 - 오경보율)
  - FPPI = 가짜 정상 이미지당 평균 검출 수
  - 오경보를 '지운 자리(예전 점 중심 5px 이내)'와 '그 밖의 자리'로 나눈다 (지운 자리 반응은 보간 자국이나 잔여일 수 있음)
  - 이미지 점수(최고 검출 신뢰도)로 만든 ROC-AUC: 원본(결함 있음) 대 가짜 정상(결함 없음). 같은 사진의 쌍이다.
주의: 오경보에는 보간으로 지운 흔적이 만든 것이 섞일 수 있다. 그래서 방식(평균 보간 / Navier-Stokes+노이즈)과 지우는 크기를 바꿔 함께 본다.
원본 data/ 는 바꾸지 않는다. 지운 이미지는 --work-dir 에만 만들고 Git 에 올리지 않는다.
"""
import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FR, YO, MB, DR = (ROOT / "src" / "minyeop" / p for p in ("faster_rcnn", "yolov3_tiny", "faster_rcnn_mobilenet", "dot_removal"))
sys.path.insert(0, str(MB))
sys.path.insert(0, str(FR))
sys.path.insert(0, str(DR))
import metrics  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auc(pos, neg):
    """Mann-Whitney 방식 ROC-AUC (동점은 0.5)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", default="runs/minyeop/05_fake_normal_v1")
    ap.add_argument("--r", type=float, default=5.0, help="오경보를 '지운 자리'로 볼 중심 거리(px)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    rd = load_module("remove_dot", DR / "remove_dot.py")          # 점 제거 함수 재사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fr_r50 = load_module("fr_r50", FR / "predict.py")
    fr_mob = load_module("fr_mob", FR / "predict.py")
    from mobilenet_model import build_model as mob_build  # noqa: E402
    fr_mob.build_model = mob_build
    yo = load_module("yolo_predict", YO / "predict.py")
    from utils import torch_utils  # noqa: E402

    runs = {"Faster R-CNN R50-FPN": (ROOT / "runs/minyeop/01_frcnn_r50fpn_min640_v1", 0.95),
            "YOLOv3-tiny": (ROOT / "runs/minyeop/02_yolov3tiny_img640_v1", 0.06),
            "Faster R-CNN MobileNetV3-FPN": (ROOT / "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1", 0.93)}
    orig = {m: {r["name"]: r for r in json.loads((run / "preds_test.json").read_text(encoding="utf-8"))} for m, (run, _) in runs.items()}
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8"))}
    names = sorted(orig["YOLOv3-tiny"])
    gt = {n: orig["YOLOv3-tiny"][n]["gt"] for n in names}
    with_dot = [n for n in names if gt[n]]
    empty = [n for n in names if not gt[n]]
    print(f"test {len(names)}장: 결함 있음 {len(with_dot)}장(가짜 정상으로 변환), 빈 라벨 {len(empty)}장(제품이 반만 찍힌 사진)")

    settings = [("mean", 6, 0.0), ("ns", 6, 0.75), ("mean", 8, 0.0), ("ns", 8, 0.75)]
    summary = {"설정": {"test 이미지": len(names), "결함 있음": len(with_dot), "빈 라벨": len(empty), "지운 자리 반경(px)": a.r}, "결과": {}}
    per_image_rows = []
    for method, half, noise in settings:
        tag = f"{method}_half{half}"
        work = Path(a.work_dir) / "fakenormal" / tag
        if work.exists():
            shutil.rmtree(work)
        (work / "test" / "images").mkdir(parents=True)
        shutil.copytree(ROOT / "data/test/labels", work / "test" / "labels")
        rng = np.random.default_rng(a.seed)
        spots = {}
        for n in names:
            src = ROOT / "data/test/images" / n
            if gt[n]:
                im, sp, _, _ = rd.remove_dots(src, gt[n], half, rng, noise, method)
                Image.fromarray(im, "L").save(work / "test" / "images" / n)
                spots[n] = sp
            else:
                shutil.copy(src, work / "test" / "images" / n)
        new = {}
        new["Faster R-CNN R50-FPN"] = fr_r50.predict_split(runs["Faster R-CNN R50-FPN"][0], "test", "best", device, 4, data_dir=str(work))[0]
        new["Faster R-CNN MobileNetV3-FPN"] = fr_mob.predict_split(runs["Faster R-CNN MobileNetV3-FPN"][0], "test", "best", device, 4, data_dir=str(work))[0]
        ycfg = json.loads((runs["YOLOv3-tiny"][0] / "config.json").read_text(encoding="utf-8"))
        new["YOLOv3-tiny"] = yo.predict_split(runs["YOLOv3-tiny"][0], "test", "best", torch_utils.select_device("0"), ycfg["img_size"], data_root=str(work))[0]

        summary["결과"][tag] = {}
        for m, (run, thr) in runs.items():
            recs = {r["name"]: r for r in new[m]}
            mx = lambda r: max([d[4] for d in r["dets"]], default=0.0)
            dets_ge = lambda r: [d for d in r["dets"] if d[4] >= thr]
            fake_max = np.array([mx(recs[n]) for n in with_dot])
            orig_max = np.array([mx(orig[m][n]) for n in with_dot])
            fa_img = int((fake_max >= thr).sum())
            n_det = near = far = 0
            for n in with_dot:
                for d in dets_ge(recs[n]):
                    n_det += 1
                    cx, cy = center(d)
                    if any(np.hypot(cx - x, cy - y) <= a.r for x, y in spots[n]):
                        near += 1
                    else:
                        far += 1
            e_fa = sum(1 for n in empty if mx(recs[n]) >= thr)
            lo, hi = wilson(fa_img, len(with_dot))
            summary["결과"][tag][m] = {
                "임계값": thr, "가짜 정상 이미지": len(with_dot), "오경보 이미지": fa_img, "오경보율": fa_img / len(with_dot), "오경보율 95% 구간": [lo, hi],
                "특이도": 1 - fa_img / len(with_dot), "FPPI": n_det / len(with_dot), "오경보 검출 수": n_det, "그중 지운 자리": near, "그중 그 밖의 자리": far,
                "원본(결함 있음) 이미지 검출률": float((orig_max >= thr).mean()), "이미지 점수 ROC-AUC(원본 대 가짜 정상)": auc(orig_max, fake_max),
                "빈 라벨 이미지 오경보": f"{e_fa}/{len(empty)}",
            }
            for n, om, fm in zip(with_dot, orig_max, fake_max):
                per_image_rows.append([tag, m, n, man[n]["machine"], man[n]["source"], round(float(om), 4), round(float(fm), 4)])
            s = summary["결과"][tag][m]
            print(f"[{tag:12}] {m:30} 오경보 이미지 {fa_img:>3}/{len(with_dot)} ({s['오경보율']*100:4.1f}%, 구간 {lo*100:.1f}~{hi*100:.1f}) | FPPI {s['FPPI']:.3f} "
                  f"(지운 자리 {near} / 그 밖 {far}) | AUC {s['이미지 점수 ROC-AUC(원본 대 가짜 정상)']:.4f} | 빈 라벨 오경보 {e_fa}/{len(empty)}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "per_image.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["설정", "모델", "이미지", "호기", "라벨출처", "원본 최고신뢰도", "가짜정상 최고신뢰도"])
        w.writerows(per_image_rows)
    print("저장:", out)


if __name__ == "__main__":
    main()
