"""팀원 모델(YOLO26n, RT-DETR-l, D-FINE-N)의 가중치로 val·test·합성 데이터를 추론해, 우리 모델과 같은 형식의 검출 결과 JSON 을 만든다.

  python src/minyeop/extra_models/predict_extra.py --out runs/minyeop/09_extra_models_v1 [--models yolo26n,rtdetr_l,dfine_n] [--sets val,test,synth]

가중치는 weights/pts/ 에 둔다(Git 제외). 각 모델은 만든 사람의 설정(입력 크기, 신뢰도 하한, 최대 검출 수)을 따른다.
  yolo26n   : 입력 1024, conf 0.001, max_det 50   (runs/chong/01_yolo26n_img1024 의 harness_config)
  rtdetr_l  : 입력 640,  conf 0.001, max_det 300  (src/yeonchang/rtdetr/evaluate.py)
  dfine_n   : 입력 640x640 직접 리사이즈, 정규화 없음(preprocessor_config.json), 점수 0.001 이상 전부 (Hugging Face DFineForObjectDetection)
출력 형식: [{"name", "w", "h", "gt": [[x1,y1,x2,y2]], "dets": [[x1,y1,x2,y2,score]]}] (src/minyeop/faster_rcnn/metrics.py 의 입력과 같음)
preds_*.json 에는 정답 좌표가 들어 있어 Git 에 올리지 않는다. 임계값은 val 결과로 정하는 것은 이 스크립트가 아니라 평가 쪽(eval_extra.py)이 한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
PTS = ROOT / "weights" / "pts"
MODELS = {
    "yolo26n": {"path": PTS / "yolo26" / "yolo26n_10241024_40epoch_mAP50_0982" / "best.pt", "kind": "ultra", "cls": "YOLO", "imgsz": 1024, "max_det": 50},
    "rtdetr_l": {"path": PTS / "RT-DETR-l" / "best.pt", "kind": "ultra", "cls": "RTDETR", "imgsz": 640, "max_det": 300},
    "dfine_n": {"path": PTS / "D-FINE-N" / "best", "kind": "dfine", "imgsz": 640},
}
SETS = {"val": ROOT / "data" / "val", "test": ROOT / "data" / "test", "synth": ROOT / "data_synth" / "test"}


def read_gt(label_path, w, h):
    out = []
    if label_path.exists():
        for ln in label_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                _, cx, cy, bw, bh = (float(v) for v in ln.split()[:5])
                out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def load_images(paths):
    return [np.array(Image.open(p).convert("L")) for p in paths]


def run_ultra(spec, imgs, device, batch):
    import ultralytics
    model = getattr(ultralytics, spec["cls"])(str(spec["path"]))
    dets = []
    for i in range(0, len(imgs), batch):
        chunk = [np.stack([im] * 3, -1) for im in imgs[i:i + batch]]            # 회색조 -> 3채널
        for r in model.predict(chunk, imgsz=spec["imgsz"], conf=0.001, max_det=spec["max_det"], device=device, verbose=False):
            b = r.boxes
            dets.append([[*map(float, xy), float(c)] for xy, c in zip(b.xyxy.tolist(), b.conf.tolist())])
    return dets


def run_dfine(spec, imgs, device, batch):
    import torch.nn.functional as F
    from transformers import DFineForObjectDetection
    model = DFineForObjectDetection.from_pretrained(str(spec["path"])).to(device).eval()
    s = spec["imgsz"]
    dets = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            chunk = imgs[i:i + batch]
            x = torch.stack([F.interpolate(torch.from_numpy(im)[None, None].float().div(255).to(device), size=(s, s), mode="bilinear", align_corners=False)[0].expand(3, -1, -1) for im in chunk])
            out = model(pixel_values=x)
            prob = out.logits.sigmoid()                                           # (B, Q, 클래스 수)
            score, _ = prob.max(-1)
            cx, cy, bw, bh = out.pred_boxes.unbind(-1)                            # 0~1 비율 (cx, cy, w, h)
            for j, im in enumerate(chunk):
                h, w = im.shape
                keep = score[j] >= 0.001
                bx = torch.stack([(cx[j] - bw[j] / 2) * w, (cy[j] - bh[j] / 2) * h, (cx[j] + bw[j] / 2) * w, (cy[j] + bh[j] / 2) * h, score[j]], -1)[keep]
                dets.append(bx.cpu().tolist())
    return dets


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    ap.add_argument("--models", default="yolo26n,rtdetr_l,dfine_n")
    ap.add_argument("--sets", default="val,test,synth")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="이미지 수 제한(동작 확인용)")
    a = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = ROOT / a.out
    for m in a.models.split(","):
        spec = MODELS[m]
        (out / m).mkdir(parents=True, exist_ok=True)
        for s in a.sets.split(","):
            root = SETS[s]
            paths = sorted((root / "images").glob("*.png"))[:a.limit]
            imgs = load_images(paths)
            t0 = time.time()
            dets = (run_ultra if spec["kind"] == "ultra" else run_dfine)(spec, imgs, device, a.batch)
            sec = time.time() - t0
            recs = []
            for p, im, d in zip(paths, imgs, dets):
                h, w = im.shape
                recs.append({"name": p.name, "w": w, "h": h, "gt": read_gt(root / "labels" / (p.stem + ".txt"), w, h), "dets": d})
            (out / m / f"preds_{s}.json").write_text(json.dumps(recs), encoding="utf-8")
            n_det = sum(len(r["dets"]) for r in recs)
            print(f"{m:9} {s:5} {len(recs)}장, 검출 후보 {n_det}개 (이미지당 {n_det / max(len(recs), 1):.1f}), {sec:.0f}초(모델 로드 포함)", flush=True)


if __name__ == "__main__":
    main()
