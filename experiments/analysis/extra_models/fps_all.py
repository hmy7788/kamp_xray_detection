"""6개 모델의 FPS를 4가지 조건으로 잰다: GPU FP32, GPU FP16, CPU 4스레드, CPU 1스레드.

  python src/minyeop/extra_models/fps_all.py --out runs/minyeop/09_extra_models_v1 [--passes 3]

조건(speed_all.py 와 같음): 배치 1, 이미지 읽기·리사이즈·정규화 등 전처리는 시간에서 제외, 시간 = 모델 연산 + 후처리(NMS, 상위 검출 선택),
GPU 는 동기화 후 측정, 워밍업 제외, test 이미지에서 고르게 고른 장 수(GPU 200장, CPU 30장).
FPS = 1000 / 평균 지연(ms). 같은 프로세스에서 6개 모델을 모두 재고 --passes 번 반복(순서를 번갈아 뒤집음)한다.
FP16: 가중치와 입력을 half 로 바꿔 쓴다(autocast 가 아님). 우리 3개 모델은 모델을 복사해 half 로 바꾸고(FP32 모델은 그대로 둠), 후처리 입력은 float 로 되돌린다.
YOLO26n·RT-DETR-l 은 ultralytics 의 half=True, D-FINE-N 은 모델을 half 로 불러온다. FP16 이 출력을 망가뜨리지 않았는지 FP32 와 비교하는 점검(30장)을 함께 낸다
(최고 신뢰도 차이, 임계값 이상 이미지 판정 일치율). FP16 에서의 정확도(F1 등)는 이 스크립트로 재지 않는다.
다른 프로세스가 돌고 있으면 값이 오염되므로 측정 중에는 다른 작업을 하지 않는다.
"""
import argparse
import contextlib
import copy
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
N_CHECK = 30
ORDER = ["Faster R-CNN R50-FPN", "YOLOv3-tiny", "Faster R-CNN MobileNetV3-FPN", "YOLO26n", "RT-DETR-l", "D-FINE-N"]
MODES = ["GPU FP32", "GPU FP16", "CPU 4스레드", "CPU 1스레드"]


def power_status():
    """Windows 전원 상태: AC 연결 여부와 배터리 잔량. 배터리에서는 GPU·CPU 성능이 제한되어 속도가 크게 달라진다."""
    if sys.platform != "win32":
        return {"AC": None, "배터리(%)": None}
    import ctypes

    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = SPS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return {"AC": {0: False, 1: True}.get(s.ACLineStatus), "배터리(%)": int(s.BatteryLifePercent) if s.BatteryLifePercent != 255 else None}


def test_images(n):
    imgs = sorted((ROOT / "data" / "test" / "images").glob("*.png"))
    return [imgs[i] for i in np.linspace(0, len(imgs) - 1, n).astype(int)]


class TV:
    def __init__(self, name):
        self.name = name
        if name == "Faster R-CNN R50-FPN":
            m = build_r50(640, 1000, None, pretrained=False)
            run = "01_frcnn_r50fpn_min640_v1"
        else:
            m = build_mob(640, 1000, [16, 32, 64, 128, 256], pretrained=False)
            run = "04_frcnn_mobv3_min640_anc16_v1"
        m.load_state_dict(torch.load(ROOT / "runs" / "minyeop" / run / "weights" / "best.pt", map_location="cpu", weights_only=False)["model"])
        self.base = m.eval()

    def build(self, half):
        self.half = half
        self.m = copy.deepcopy(self.base).half() if half else self.base

    def prep(self, p, dev):
        x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy()).permute(2, 0, 1).float().div(255)
        return (x.half() if self.half else x).to(dev)

    def f(self, x, dev):
        return self.m([x])

    def scores(self, x, dev):
        return self.m([x])[0]["scores"].float().cpu().numpy()


class Yolov3:
    def __init__(self, name):
        self.name = name
        with contextlib.redirect_stdout(io.StringIO()):
            m = Darknet(str(ROOT / "src/yolov3/yolov3-tiny.cfg"), 640)
        m.load_state_dict(torch.load(ROOT / "runs/minyeop/02_yolov3tiny_img640_v1/weights/best.pt", map_location="cpu", weights_only=False)["model"])
        self.base = m.eval()

    def build(self, half):
        self.half = half
        self.m = copy.deepcopy(self.base).half() if half else self.base

    def prep(self, p, dev):
        with contextlib.redirect_stdout(io.StringIO()):
            _, img, _, _ = next(iter(LoadImages(str(p), img_size=640)))
        x = torch.from_numpy(img).float().div(255)
        return (x.half() if self.half else x).to(dev)

    def f(self, x, dev):
        pred = self.m(x.unsqueeze(0))[0]
        return non_max_suppression(pred.float(), 0.001, 0.6, multi_label=False)

    def scores(self, x, dev):
        out = self.f(x, dev)[0]
        return out[:, 4].float().cpu().numpy() if out is not None else np.zeros(0)


class Ultra:
    def __init__(self, name, path, cls, imgsz, scale_fill, max_det):
        self.name, self.path, self.cls = name, path, cls
        self.imgsz, self.max_det = imgsz, max_det
        from ultralytics.data.augment import LetterBox
        self.lb = LetterBox((imgsz, imgsz), auto=not scale_fill, scale_fill=scale_fill, stride=32)

    def build(self, half):
        import ultralytics
        self.half = half
        self.m = getattr(ultralytics, self.cls)(str(self.path))   # 정밀도마다 새로 불러와 서로 영향을 주지 않게 한다

    def prep(self, p, dev):
        im = np.stack([np.asarray(Image.open(p).convert("L"))] * 3, -1)
        im = self.lb(image=im)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float().div(255).unsqueeze(0).to(dev)

    def f(self, x, dev):
        return self.m.predict(x, imgsz=self.imgsz, conf=0.001, max_det=self.max_det, device=str(dev), half=self.half and dev.type == "cuda", verbose=False)

    def scores(self, x, dev):
        return self.f(x, dev)[0].boxes.conf.float().cpu().numpy()


class Dfine:
    def __init__(self, name):
        self.name = name

    def build(self, half):
        from transformers import DFineForObjectDetection
        self.half = half
        m = DFineForObjectDetection.from_pretrained(str(PTS / "D-FINE-N" / "best")).eval()
        self.m = m.half() if half else m

    def prep(self, p, dev):
        import torch.nn.functional as F
        g = torch.from_numpy(np.asarray(Image.open(p).convert("L")).copy())[None, None].float().div(255)
        x = F.interpolate(g, size=(640, 640), mode="bilinear", align_corners=False).expand(1, 3, -1, -1).contiguous()
        return (x.half() if self.half else x).to(dev)

    def f(self, x, dev):
        out = self.m(pixel_values=x)
        sc = out.logits.sigmoid().max(-1)[0]
        return out.pred_boxes[sc >= 0.001]

    def scores(self, x, dev):
        out = self.m(pixel_values=x)
        return out.logits.sigmoid().max(-1)[0].float().flatten().cpu().numpy()


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
    mean = statistics.mean(ts)
    return {"평균(ms)": mean, "중앙값(ms)": statistics.median(ts), "p95(ms)": ts[max(int(len(ts) * 0.95) - 1, 0)], "FPS": 1000 / mean, "n": len(ts)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--allow-battery", action="store_true", help="충전기가 연결되지 않아도 측정한다(값이 낮고 변동이 크다)")
    a = ap.parse_args()
    pw = power_status()
    if pw["AC"] is False and not a.allow_battery:
        sys.exit(f"충전기가 연결되어 있지 않다(배터리 {pw['배터리(%)']}%). 배터리에서는 성능이 제한되어 값이 낮고 변동이 크다. 충전기를 연결하고 전원 모드를 '최고 성능'으로 한 뒤 다시 실행하거나 --allow-battery 를 붙여라.")
    runners = {"Faster R-CNN R50-FPN": TV("Faster R-CNN R50-FPN"), "YOLOv3-tiny": Yolov3("YOLOv3-tiny"), "Faster R-CNN MobileNetV3-FPN": TV("Faster R-CNN MobileNetV3-FPN"),
               "YOLO26n": Ultra("YOLO26n", PTS / "yolo26" / "yolo26n_10241024_40epoch_mAP50_0982" / "best.pt", "YOLO", 1024, False, 50),
               "RT-DETR-l": Ultra("RT-DETR-l", PTS / "RT-DETR-l" / "best.pt", "RTDETR", 640, True, 300), "D-FINE-N": Dfine("D-FINE-N")}
    thr = {m: v["임계값(val, IoU0.5)"] for m, v in json.loads((ROOT / "runs/minyeop/09_extra_models_v1/summary.json").read_text(encoding="utf-8"))["실제 val·test"].items()}
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    res = {"설정": {"전원": pw, "GPU 이미지": N_GPU, "CPU 이미지": N_CPU, "워밍업": [WARM_GPU, WARM_CPU], "패스": a.passes, "배치": 1, "FPS": "1000 / 평균 지연(ms)", "FP16": "가중치·입력 half (autocast 아님)"},
           "측정": [], "FP16 출력 점검": {}}
    gpu_imgs, cpu_imgs, chk_imgs = test_images(N_GPU), test_images(N_CPU), test_images(N_CHECK)

    # FP16 출력 점검 (FP32 와 비교): 최고 신뢰도 차이, 임계값 이상 이미지 판정 일치율
    dev = torch.device("cuda")
    for name in ORDER:
        r = runners[name]
        r.build(False)
        if hasattr(r.m, "to"):
            r.m.to(dev)
        x32 = [r.prep(p, dev) for p in chk_imgs]
        with torch.inference_mode():
            s32 = [r.scores(x, dev) for x in x32]
        r.build(True)
        if hasattr(r.m, "to"):
            r.m.to(dev)
        x16 = [r.prep(p, dev) for p in chk_imgs]
        with torch.inference_mode():
            s16 = [r.scores(x, dev) for x in x16]
        t = thr[name]
        top32 = np.array([s.max() if len(s) else 0.0 for s in s32])
        top16 = np.array([s.max() if len(s) else 0.0 for s in s16])
        same = float(np.mean([(a_ >= t) == (b_ >= t) for a_, b_ in zip(top32, top16)]))
        nan = bool(np.isnan(top16).any())
        res["FP16 출력 점검"][name] = {"최고 신뢰도 차이(최대)": float(np.abs(top32 - top16).max()), "최고 신뢰도 차이(평균)": float(np.abs(top32 - top16).mean()),
                                  "임계값 이상 판정 일치율": same, "FP32에서 임계값 이상 이미지": int((top32 >= t).sum()), "FP16에서 임계값 이상 이미지": int((top16 >= t).sum()), "NaN": nan}
        print(f"[FP16 점검] {name:30} 최고 신뢰도 차이 최대 {np.abs(top32 - top16).max():.4f}, 판정 일치 {same * 100:.0f}% (FP32 {int((top32 >= t).sum())}장 / FP16 {int((top16 >= t).sum())}장 / {N_CHECK}장){' NaN!' if nan else ''}", flush=True)
        if hasattr(r.m, "to"):
            r.m.to("cpu")
        torch.cuda.empty_cache()

    for ps in range(a.passes):
        order = ORDER if ps % 2 == 0 else ORDER[::-1]
        one = {}
        for mode in MODES:
            half = mode == "GPU FP16"
            dev = torch.device("cuda" if mode.startswith("GPU") else "cpu")
            if mode.startswith("CPU"):
                torch.set_num_threads(4 if "4" in mode else 1)
            for name in order:
                r = runners[name]
                r.build(half)
                if hasattr(r.m, "to"):
                    r.m.to(dev)
                xs = [r.prep(p, dev) for p in (gpu_imgs if dev.type == "cuda" else cpu_imgs)]
                st = timeit(r, xs, dev, WARM_GPU if dev.type == "cuda" else WARM_CPU)
                one.setdefault(mode, {})[name] = st
                print(f"[패스 {ps + 1}] {mode:11} {name:30} FPS {st['FPS']:7.1f} | 평균 {st['평균(ms)']:8.1f} ms | 중앙값 {st['중앙값(ms)']:8.1f} | p95 {st['p95(ms)']:8.1f}", flush=True)
                if hasattr(r.m, "to"):
                    r.m.to("cpu")
                torch.cuda.empty_cache()
        res["측정"].append(one)
    summ = {}
    for mode in MODES:
        summ[mode] = {}
        for name in ORDER:
            fps = [p[mode][name]["FPS"] for p in res["측정"]]
            summ[mode][name] = {"FPS 중앙값": statistics.median(fps), "FPS 최소": min(fps), "FPS 최대": max(fps), "패스별 FPS": fps,
                                "지연 중앙값(ms)": statistics.median([p[mode][name]["평균(ms)"] for p in res["측정"]])}
    res["요약"] = summ
    out = ROOT / a.out / "fps_all.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== FPS 요약 (패스 중앙값, 최소~최대) ==")
    for mode in MODES:
        print(mode, {k[:10]: f"{v['FPS 중앙값']:.1f} ({v['FPS 최소']:.1f}~{v['FPS 최대']:.1f})" for k, v in summ[mode].items()})
    print("저장:", out)


if __name__ == "__main__":
    main()
