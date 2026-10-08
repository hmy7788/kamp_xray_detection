"""D-FINE 예측을 RT-DETR과 같은 형식(preds_<split>.json)으로 저장한다. 채점은 ../rtdetr/evaluate.py가 한다.

    python src/yeonchang/dfine/predict.py --exp 03_dfinen_img640_e50 --split val
    python src/yeonchang/rtdetr/evaluate.py --exp 03_dfinen_img640_e50 --split val --reuse-preds    # 임계값 결정
    python src/yeonchang/dfine/predict.py --exp 03_dfinen_img640_e50 --split test
    python src/yeonchang/rtdetr/evaluate.py --exp 03_dfinen_img640_e50 --split test --reuse-preds   # 최종 1회

예측은 신뢰도 0.001 이상 전부(상위 300개). ms_per_image는 전처리+추론+후처리, batch 1, 예열 후 측정.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rtdetr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # rtdetr/train.py가 아니라 이 폴더의 train.py
from common import DATA, RUNS, check_data_version, read_manifest  # noqa: E402
from train import imread, predict_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    exp_dir = RUNS / args.exp
    cfg = json.loads((exp_dir / "config.json").read_text(encoding="utf-8"))
    if "finished" not in cfg:
        sys.exit("학습이 끝나지 않은 실험입니다. 최종 결과로 쓰지 않습니다 (_incomplete_로 이름을 바꾸세요).")
    if args.split == "test":
        if not (exp_dir / "eval_report_val.json").exists():
            sys.exit("먼저 val로 임계값을 정하세요.")
        if (exp_dir / "eval_report_test.json").exists():
            sys.exit("test는 이미 한 번 채점했습니다. 다시 보지 않습니다.")
    check_data_version()

    from transformers import AutoImageProcessor, DFineForObjectDetection
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    wdir = exp_dir / "weights" / "best"
    processor = AutoImageProcessor.from_pretrained(wdir)
    model = DFineForObjectDetection.from_pretrained(wdir).to(device).eval()
    size = cfg["hyp"]["imgsz"]

    manifest = read_manifest()
    ids = sorted(i for i, r in manifest.items() if r["split"] == args.split)
    imgs = [imread(DATA / manifest[i]["image_path"]) for i in ids]
    preds_list = predict_images(model, processor, imgs, size, device)

    # 속도: batch 1, 예열 10장 후 전체
    predict_images(model, processor, imgs[:10], size, device, batch=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    predict_images(model, processor, imgs, size, device, batch=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000 / len(imgs)

    out = exp_dir / f"preds_{args.split}.json"
    out.write_text(json.dumps({"preds": dict(zip(ids, preds_list)), "ms_per_image": ms}), encoding="utf-8")
    print(f"saved {out.name}: {len(ids)} images, {ms:.1f} ms/img (batch 1, FP16 autocast)")


if __name__ == "__main__":
    main()
