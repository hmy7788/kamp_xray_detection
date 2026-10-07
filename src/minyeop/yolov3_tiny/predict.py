"""학습된 YOLOv3-tiny 로 val/test 를 추론해 공통 검출 결과 기록(preds_<split>.json)을 저장하고 지표를 출력한다.

  python src/minyeop/yolov3_tiny/predict.py --name 02_yolov3tiny_img640_v1 --split val  --ckpt best
  python src/minyeop/yolov3_tiny/predict.py --name 02_yolov3tiny_img640_v1 --split test --ckpt best   # 최종 후보만 한 번

- 기록 형식과 임계값 규칙은 Faster R-CNN 과 같다 (src/minyeop/faster_rcnn/metrics.py, docs/README.md).
  임계값은 같은 체크포인트의 val 결과에서 F1 이 최대인 값으로 정해 test 에 그대로 적용한다.
- 신뢰도 0.001 이상 검출을 모두 저장한다 (NMS IoU 0.6, 클래스 1개). 정답 좌표가 들어 있어 Git 에는 올라가지 않는다.
- 추론 시간은 이미지 1장씩(배치 1), 전처리·NMS 포함이다. Faster R-CNN(배치 4, AMP)과 조건이 다르니 직접 비교하지 말 것.
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
YOLO = ROOT / "src" / "yolov3"
sys.path.insert(0, str(YOLO))
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn"))
import metrics  # noqa: E402  (모델 공통 지표 함수)
from models import Darknet  # noqa: E402
from utils import torch_utils  # noqa: E402
from utils.datasets import LoadImages  # noqa: E402
from utils.utils import non_max_suppression, scale_coords  # noqa: E402

CONF_FLOOR = 0.001


def predict_split(run, split, ckpt_name, device, img_size, data_root=None):
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model = Darknet(str(YOLO / cfg["cfg"]), img_size)
    ck = torch.load(run / "weights" / f"{ckpt_name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    paths = sorted(((Path(data_root) if data_root else ROOT / "data") / split / "images").glob("*.png"))  # data_root: 분석용으로 다른 이미지 폴더를 읽을 때
    out, total = [], 0.0
    with torch.no_grad():
        for p in paths:
            with contextlib.redirect_stdout(io.StringIO()):  # LoadImages 가 이미지 경로를 출력하는 것을 막는다
                _, img, im0, _ = next(iter(LoadImages(str(p), img_size=img_size)))
            x = torch.from_numpy(img).to(device).float() / 255.0
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            pred = model(x.unsqueeze(0))[0]
            det = non_max_suppression(pred, CONF_FLOOR, 0.6, multi_label=False)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            total += time.time() - t
            dets = []
            if det is not None:
                det[:, :4] = scale_coords(x.shape[1:], det[:, :4], im0.shape).round()
                dets = [[float(v) for v in d[:5]] for d in det.cpu().numpy()]  # x1 y1 x2 y2 conf
            h, w = im0.shape[:2]
            lab = p.parent.parent / "labels" / (p.stem + ".txt")
            gt = []
            for line in lab.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _, cx, cy, bw, bh = map(float, line.split()[:5])
                    gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            out.append({"name": p.name, "w": w, "h": h, "gt": gt, "dets": dets})
    return out, 1000 * total / max(len(paths), 1), ck.get("epoch")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--threshold", type=float, default=None, help="직접 지정할 임계값 (기본: val 에서 F1 최대인 값)")
    a = ap.parse_args()

    run = ROOT / "runs" / a.owner / a.name
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    device = torch_utils.select_device(a.device)
    img = cfg["img_size"]

    # 임계값은 val 에서 정한다 (체크포인트가 바뀌면 val 을 다시 계산해야 하므로 val 은 항상 새로 만든다)
    if a.split == "val":
        vrecs, vms, ep = predict_split(run, "val", a.ckpt, device, img)
        (run / "preds_val.json").write_text(json.dumps(vrecs), encoding="utf-8")
    else:
        vp = run / "preds_val.json"
        if not vp.exists():
            sys.exit("preds_val.json 이 없습니다. 먼저 같은 --ckpt 로 --split val 을 실행하세요.")
        vrecs, vms = json.loads(vp.read_text(encoding="utf-8")), None
    bf = metrics.best_f1(vrecs)
    thr = a.threshold if a.threshold is not None else bf["thr"]

    if a.split == "val":
        recs, ms = vrecs, vms
    else:
        recs, ms, ep = predict_split(run, "test", a.ckpt, device, img)
        (run / "preds_test.json").write_text(json.dumps(recs), encoding="utf-8")

    r = metrics.prf_at(recs, thr)
    print(f"\n=== {a.owner}/{a.name} | {a.ckpt}.pt (epoch {ep}) | {a.split} {len(recs)}장 ===")
    print(f"임계값 {thr} ({'지정값' if a.threshold is not None else 'val 에서 F1 최대'})")
    print(f"AP@0.5 {metrics.ap(recs):.3f} | P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} | TP {r['TP']} FP {r['FP']} FN {r['FN']}")
    if ms is not None:
        print(f"추론 시간 {ms:.1f} ms/장 (배치 1, {device})")


if __name__ == "__main__":
    main()
