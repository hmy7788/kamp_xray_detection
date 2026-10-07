"""6개 모델의 점 제거(03)와 가짜 정상 오경보(05)를 기존과 같은 설정으로 다시 돌린다 (팀원 모델 포함).

  python src/minyeop/extra_models/robust_extra.py --work-dir <임시 폴더> --out runs/minyeop/09_extra_models_v1

점 제거와 가짜 정상은 같은 이미지다(같은 함수 remove_dots, 시드 0, 같은 순서). 그래서 설정마다 지운 test 이미지를 한 번만 만들어 두 분석을 함께 계산한다.
설정(기존과 같음): 평균 보간 half 4/6/8, 평균 보간+노이즈 half 6, Navier-Stokes+노이즈(0.75) half 4/5/6/8. 가짜 정상(05)은 그중 mean 6/8, ns 6/8.
지표는 임계값을 val 값으로 고정(09 summary.json 의 '실제 val·test').
  점 제거: 정답 자리(중심 5px)에서 검출이 남는 수 / 정답 수, 전체 검출 수.
  가짜 정상: 결함 있던 이미지 369장 중 임계값 이상 검출이 하나라도 있는 이미지 수(오경보율), FPPI, 지운 자리/그 밖 구분, 빈 라벨 27장 오경보, 원본 대 가짜 정상 AUC.
주의: 지운 이미지는 진짜 정상이 아니다(보간 흔적, 노이즈 패턴이 단서가 될 수 있음). 오경보는 범위로 해석한다(docs/analysis.md 2절).
지운 이미지는 --work-dir 에만 만들고 Git 에 올리지 않는다.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
FR, YO, MB, DR, EX = (ROOT / "src" / "minyeop" / p for p in ("faster_rcnn", "yolov3_tiny", "faster_rcnn_mobilenet", "dot_removal", "extra_models"))
sys.path.insert(0, str(MB))
sys.path.insert(0, str(FR))
sys.path.insert(0, str(DR))

OURS = {"Faster R-CNN R50-FPN": "runs/minyeop/01_frcnn_r50fpn_min640_v1", "YOLOv3-tiny": "runs/minyeop/02_yolov3tiny_img640_v1",
        "Faster R-CNN MobileNetV3-FPN": "runs/minyeop/04_frcnn_mobv3_min640_anc16_v1"}
EXTRA = {"YOLO26n": "yolo26n", "RT-DETR-l": "rtdetr_l", "D-FINE-N": "dfine_n"}
SETTINGS = [("mean", 4, 0.0), ("mean", 6, 0.0), ("mean", 8, 0.0), ("mean", 6, 0.75), ("ns", 4, 0.75), ("ns", 5, 0.75), ("ns", 6, 0.75), ("ns", 8, 0.75)]
FAKE_TAGS = {("mean", 6, 0.0), ("ns", 6, 0.75), ("mean", 8, 0.0), ("ns", 8, 0.75)}


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
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def ctr(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def tag(method, half, noise):
    return f"{method}{'noise' if (method == 'mean' and noise > 0) else ''}_half{half}"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    ap.add_argument("--r", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / a.out
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rd = load_module("remove_dot", DR / "remove_dot.py")
    px = load_module("predict_extra", EX / "predict_extra.py")
    fr_r50 = load_module("fr_r50", FR / "predict.py")
    fr_mob = load_module("fr_mob", FR / "predict.py")
    from mobilenet_model import build_model as mob_build  # noqa: E402
    fr_mob.build_model = mob_build
    yo = load_module("yolo_predict", YO / "predict.py")
    from utils import torch_utils  # noqa: E402
    ycfg = json.loads((ROOT / OURS["YOLOv3-tiny"] / "config.json").read_text(encoding="utf-8"))

    real = json.loads((out / "summary.json").read_text(encoding="utf-8"))["실제 val·test"]
    thr = {m: real[m]["임계값(val, IoU0.5)"] for m in list(OURS) + list(EXTRA)}
    orig = {m: {r["name"]: r for r in json.loads((ROOT / run / "preds_test.json").read_text(encoding="utf-8"))} for m, run in OURS.items()}
    for m, d in EXTRA.items():
        orig[m] = {r["name"]: r for r in json.loads((out / d / "preds_test.json").read_text(encoding="utf-8"))}
    names = sorted(orig["YOLOv3-tiny"])
    gt = {n: orig["YOLOv3-tiny"][n]["gt"] for n in names}
    with_dot = [n for n in names if gt[n]]
    empty = [n for n in names if not gt[n]]
    print(f"test {len(names)}장: 결함 있음 {len(with_dot)}장, 빈 라벨 {len(empty)}장", flush=True)

    res = {"설정": {"같은 자리 판정 반경(px)": a.r, "지운 점 수": sum(len(gt[n]) for n in with_dot), "결함 있던 이미지": len(with_dot), "빈 라벨 이미지": len(empty), "임계값": thr},
           "점 제거": {}, "가짜 정상": {}}
    for method, half, noise in SETTINGS:
        tg = tag(method, half, noise)
        work = Path(a.work_dir) / "robust" / tg
        if work.exists():
            shutil.rmtree(work)
        (work / "test" / "images").mkdir(parents=True)
        shutil.copytree(ROOT / "data/test/labels", work / "test" / "labels")
        rng = np.random.default_rng(a.seed)
        spots, c_before, c_after = {}, [], []
        for n in names:
            src = ROOT / "data/test/images" / n
            if gt[n]:
                im, sp, cb, ca = rd.remove_dots(src, gt[n], half, rng, noise, method)
                Image.fromarray(im, "L").save(work / "test" / "images" / n)
                spots[n] = sp
                c_before += cb
                c_after += ca
            else:
                shutil.copy(src, work / "test" / "images" / n)
        new = {}
        new["Faster R-CNN R50-FPN"] = {r["name"]: r for r in fr_r50.predict_split(ROOT / OURS["Faster R-CNN R50-FPN"], "test", "best", device, 4, data_dir=str(work))[0]}
        new["Faster R-CNN MobileNetV3-FPN"] = {r["name"]: r for r in fr_mob.predict_split(ROOT / OURS["Faster R-CNN MobileNetV3-FPN"], "test", "best", device, 4, data_dir=str(work))[0]}
        new["YOLOv3-tiny"] = {r["name"]: r for r in yo.predict_split(ROOT / OURS["YOLOv3-tiny"], "test", "best", torch_utils.select_device("0"), ycfg["img_size"], data_root=str(work))[0]}
        paths = sorted((work / "test" / "images").glob("*.png"))
        imgs = px.load_images(paths)
        for m, d in EXTRA.items():
            spec = px.MODELS[d]
            dets = (px.run_ultra if spec["kind"] == "ultra" else px.run_dfine)(spec, imgs, "cuda:0" if torch.cuda.is_available() else "cpu", 8)
            new[m] = {p.name: {"name": p.name, "dets": dd} for p, dd in zip(paths, dets)}
        print(f"\n===== {tg} (지운 점 {len(c_before)}개, 대비 중앙값 {np.median(c_before):.1f} -> {np.median(c_after):.1f}) =====", flush=True)
        res["점 제거"][tg] = {"대비 중앙값(전→후)": [float(np.median(c_before)), float(np.median(c_after))], "모델": {}}
        do_fake = (method, half, noise) in FAKE_TAGS
        if do_fake:
            res["가짜 정상"][tg] = {}
        for m in thr:
            t = thr[m]
            # 점 제거: 정답 자리 검출
            hits = n_det = n_gt = 0
            for n in names:
                dets = [d for d in new[m][n]["dets"] if d[4] >= t]
                n_det += len(dets)
                for g in gt[n]:
                    n_gt += 1
                    gx, gy = ctr(g)
                    hits += int(any(np.hypot(ctr(d)[0] - gx, ctr(d)[1] - gy) <= a.r for d in dets))
            o_hits = 0
            for n in names:
                dets = [d for d in orig[m][n]["dets"] if d[4] >= t]
                for g in gt[n]:
                    gx, gy = ctr(g)
                    o_hits += int(any(np.hypot(ctr(d)[0] - gx, ctr(d)[1] - gy) <= a.r for d in dets))
            res["점 제거"][tg]["모델"][m] = {"원본 정답 자리 검출": o_hits, "지운 뒤 정답 자리 검출": hits, "정답 수": n_gt, "지운 뒤 전체 검출 수": n_det}
            line = f"  {m:30} 점 제거: 원본 {o_hits}/{n_gt} -> 지운 뒤 {hits}/{n_gt} (전체 검출 {n_det})"
            if do_fake:
                mx = lambda r: max([d[4] for d in r["dets"]], default=0.0)
                fake_max = np.array([mx(new[m][n]) for n in with_dot])
                orig_max = np.array([mx(orig[m][n]) for n in with_dot])
                fa = int((fake_max >= t).sum())
                n_d = near = 0
                for n in with_dot:
                    for d in new[m][n]["dets"]:
                        if d[4] >= t:
                            n_d += 1
                            near += int(any(np.hypot(ctr(d)[0] - x, ctr(d)[1] - y) <= a.r for x, y in spots[n]))
                e_fa = sum(1 for n in empty if mx(new[m][n]) >= t)
                lo, hi = wilson(fa, len(with_dot))
                res["가짜 정상"][tg][m] = {"오경보 이미지": fa, "오경보율": fa / len(with_dot), "구간": [lo, hi], "FPPI": n_d / len(with_dot), "오경보 검출 수": n_d,
                                         "그중 지운 자리": near, "그 밖": n_d - near, "AUC(원본 대 가짜 정상)": auc(orig_max, fake_max), "빈 라벨 오경보": f"{e_fa}/{len(empty)}"}
                line += f" | 가짜 정상 오경보 {fa}/{len(with_dot)} ({fa / len(with_dot) * 100:.1f}%, 지운 자리 {near}/그 밖 {n_d - near}) AUC {res['가짜 정상'][tg][m]['AUC(원본 대 가짜 정상)']:.3f} 빈 라벨 {e_fa}/{len(empty)}"
            print(line, flush=True)
    (out / "robust_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장:", out / "robust_summary.json")


if __name__ == "__main__":
    main()
