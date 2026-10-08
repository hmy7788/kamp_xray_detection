"""6개 모델의 이미지당 추론 시간을 한 스크립트에서 같은 조건으로 잰다.

  python src/minyeop/extra_models/speed_all.py --out runs/minyeop/09_extra_models_v1 [--passes 2]

조건(기존 속도 측정과 같음): 배치 1, fp32, 이미지 읽기·리사이즈·정규화 같은 전처리는 시간에서 제외(미리 만들어 장치에 올려 둠),
시간 = 모델 연산 + 후처리(NMS, 상위 검출 선택 등), GPU 는 동기화 후 측정, 워밍업 제외, test 이미지에서 고르게 고른 장 수의 중앙값.
GPU 200장, CPU 30장(스레드 1·4·16). 같은 프로세스에서 6개 모델을 모두 재고, 노트북 변동을 보이기 위해 --passes 번 반복(순서를 뒤집음)한다.
다른 프로세스(학습, 추론)가 돌고 있으면 값이 오염되므로 측정 중에는 다른 작업을 하지 않는다.
입력: 우리 3개 모델은 이미지를 640(짧은 변 기준 torchvision 내부 리사이즈 / YOLOv3-tiny 는 letterbox 640)으로, 팀원 모델은 각자의 입력 크기
(YOLO26n 1024 letterbox, RT-DETR-l 640x640, D-FINE-N 640x640)로 전처리한다(기존 평가와 같은 입력).
"""
import argparse
import contextlib
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "yolov3"))
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn_mobilenet"))
sys.path.insert(0, str(ROOT / "src" / "minyeop" / "faster_rcnn"))
from model import build_model as build_r50  # noqa: E402
from mobilenet_model import build_model as build_mob  # noqa: E402
from models import Darknet  # noqa: E402
from utils.datasets import LoadImages  # noqa: E402
from utils.utils import non_max_suppression  # noqa: E402

PTS = ROOT / "weights" / "pts"
N_GPU, N_CPU, WARM_GPU, WARM_CPU = 200, 30, 20, 3
THREADS = [1, 4, 16]
ORDER = ["Faster R-CNN R50-FPN", "YOLOv3-tiny", "Faster R-CNN MobileNetV3-FPN", "YOLO26n", "RT-DETR-l", "D-FINE-N"]


def test_images(n):
    imgs = sorted((ROOT / "data" / "test" / "images").glob("*.png"))
    return [imgs[i] for i in np.linspace(0, len(imgs) - 1, n).astype(int)]


class Runner:
    """모델 하나: 입력 전처리(시간 제외)와 시간 잴 함수."""

    def __init__(self, name):
        self.name = name

    def params(self):
        return 0.0


class TV(Runner):
    def __init__(self, name):
        super().__init__(name)
        if name == "Faster R-CNN R50-FPN":
            self.m = build_r50(640, 1000, None, pretrained=False)
            run = "01_frcnn_r50fpn_min640_v1"
        else:
            self.m = build_mob(640, 1000, [16, 32, 64, 128, 256], pretrained=False)
            run = "04_frcnn_mobv3_min640_anc16_v1"
        self.m.load_state_dict(torch.load(ROOT / "runs" / "minyeop" / run / "weights" / "best.pt", map_location="cpu", weights_only=False)["model"])
        self.m.eval()

    def prep(self, p, dev):
        return torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).permute(2, 0, 1).float().div(255).to(dev)

    def to(self, dev):
        self.m.to(dev)

    def f(self, x, dev):
        return self.m([x])

    def params(self):
        return sum(p.numel() for p in self.m.parameters()) / 1e6


class Yolov3(Runner):
    def __init__(self, name):
        super().__init__(name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.m = Darknet(str(ROOT / "src/yolov3/yolov3-tiny.cfg"), 640)
        self.m.load_state_dict(torch.load(ROOT / "runs/minyeop/02_yolov3tiny_img640_v1/weights/best.pt", map_location="cpu", weights_only=False)["model"])
        self.m.eval()

    def prep(self, p, dev):
        with contextlib.redirect_stdout(io.StringIO()):
            _, img, _, _ = next(iter(LoadImages(str(p), img_size=640)))
        return torch.from_numpy(img).float().div(255).to(dev)

    def to(self, dev):
        self.m.to(dev)

    def f(self, x, dev):
        pred = self.m(x.unsqueeze(0))[0]
        return non_max_suppression(pred, 0.001, 0.6, multi_label=False)

    def params(self):
        return sum(p.numel() for p in self.m.parameters()) / 1e6


class Ultra(Runner):
    def __init__(self, name, path, cls, imgsz, scale_fill, max_det):
        super().__init__(name)
        import ultralytics
        self.m = getattr(ultralytics, cls)(str(path))
        self.imgsz, self.scale_fill, self.max_det = imgsz, scale_fill, max_det
        from ultralytics.data.augment import LetterBox
        self.lb = LetterBox((imgsz, imgsz), auto=not scale_fill, scale_fill=scale_fill, stride=32)

    def prep(self, p, dev):
        im = np.stack([np.asarray(Image.open(p).convert("L"))] * 3, -1)
        im = self.lb(image=im)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float().div(255).unsqueeze(0).to(dev)

    def to(self, dev):
        pass

    def f(self, x, dev):
        return self.m.predict(x, imgsz=self.imgsz, conf=0.001, max_det=self.max_det, device=str(dev), verbose=False)

    def params(self):
        return sum(p.numel() for p in self.m.model.parameters()) / 1e6


class Dfine(Runner):
    def __init__(self, name):
        super().__init__(name)
        from transformers import DFineForObjectDetection
        self.m = DFineForObjectDetection.from_pretrained(str(PTS / "D-FINE-N" / "best")).eval()

    def prep(self, p, dev):
        import torch.nn.functional as F
        g = torch.from_numpy(np.asarray(Image.open(p).convert("L")).copy())[None, None].float().div(255)
        return F.interpolate(g, size=(640, 640), mode="bilinear", align_corners=False).expand(1, 3, -1, -1).contiguous().to(dev)

    def to(self, dev):
        self.m.to(dev)

    def f(self, x, dev):
        out = self.m(pixel_values=x)
        sc = out.logits.sigmoid().max(-1)[0]
        return out.pred_boxes[sc >= 0.001]

    def params(self):
        return sum(p.numel() for p in self.m.parameters()) / 1e6


def timeit(r, xs, dev, warm):
    ts = []
    with torch.inference_mode():
        for i in range(warm):
            r.f(xs[i % len(xs)], dev)
        for x in xs:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            r.f(x, dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"중앙값": statistics.median(ts), "평균": statistics.mean(ts), "p95": ts[max(int(len(ts) * 0.95) - 1, 0)], "n": len(ts)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    runners = {"Faster R-CNN R50-FPN": TV("Faster R-CNN R50-FPN"), "YOLOv3-tiny": Yolov3("YOLOv3-tiny"), "Faster R-CNN MobileNetV3-FPN": TV("Faster R-CNN MobileNetV3-FPN"),
               "YOLO26n": Ultra("YOLO26n", PTS / "yolo26" / "yolo26n_10241024_40epoch_mAP50_0982" / "best.pt", "YOLO", 1024, False, 50),
               "RT-DETR-l": Ultra("RT-DETR-l", PTS / "RT-DETR-l" / "best.pt", "RTDETR", 640, True, 300), "D-FINE-N": Dfine("D-FINE-N")}
    params = {k: round(r.params(), 1) for k, r in runners.items()}
    print("파라미터(M):", params, "| GPU:", torch.cuda.get_device_name(0), flush=True)
    res = {"설정": {"GPU 이미지": N_GPU, "CPU 이미지": N_CPU, "워밍업": [WARM_GPU, WARM_CPU], "CPU 스레드": THREADS, "패스": a.passes, "정밀도": "fp32", "배치": 1}, "파라미터(M)": params, "측정": []}
    gpu_imgs, cpu_imgs = test_images(N_GPU), test_images(N_CPU)
    for ps in range(a.passes):
        order = ORDER if ps % 2 == 0 else ORDER[::-1]
        one = {}
        for dev_name in ("cuda", "cpu"):
            dev = torch.device(dev_name)
            for th in ([None] if dev_name == "cuda" else THREADS):
                if th:
                    torch.set_num_threads(th)
                label = "GPU" if th is None else f"CPU {th}스레드"
                for name in order:
                    r = runners[name]
                    r.to(dev)
                    xs = [r.prep(p, dev) for p in (gpu_imgs if dev_name == "cuda" else cpu_imgs)]
                    stat = timeit(r, xs, dev, WARM_GPU if dev_name == "cuda" else WARM_CPU)
                    one.setdefault(label, {})[name] = stat
                    print(f"[패스 {ps + 1}] {label:10} {name:30} 중앙값 {stat['중앙값']:8.1f} ms | 평균 {stat['평균']:8.1f} | p95 {stat['p95']:8.1f}", flush=True)
                    if dev_name == "cuda":
                        r.to(torch.device("cpu"))
        res["측정"].append(one)
    # 패스별 중앙값의 범위
    summ = {}
    for label in res["측정"][0]:
        summ[label] = {}
        for name in ORDER:
            v = [p[label][name]["중앙값"] for p in res["측정"]]
            summ[label][name] = {"최소": min(v), "최대": max(v), "패스별": v}
    res["요약"] = summ
    out = ROOT / a.out / "speed_all.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 요약(패스별 중앙값의 최소~최대, ms) ==")
    for label, d in summ.items():
        print(label, {k[:12]: f"{v['최소']:.0f}~{v['최대']:.0f}" for k, v in d.items()})
    print("저장:", out)


if __name__ == "__main__":
    main()
