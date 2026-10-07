"""팀원 모델(YOLO26n, RT-DETR-l, D-FINE-N)의 추론 속도와 파라미터 수. 우리 모델의 측정(docs/analysis.md 4절)과 같은 조건: 배치 1, fp32, test 이미지.

  python src/minyeop/extra_models/speed_extra.py --out runs/minyeop/09_extra_models_v1

시간 = 이미지를 읽은 뒤의 전처리(리사이즈·정규화) + 모델 연산 + 후처리. 파일 읽기는 제외한다. GPU 는 동기화 후 측정, CPU 는 스레드 수를 고정한다.
앞 10장은 워밍업이라 제외하고, 이후 N장의 중앙값과 평균을 낸다. 노트북 CPU 는 실행마다 20~40% 변동하므로 한 번의 측정으로 단정하지 않는다(해당 값은 참고용).
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("predict_extra", ROOT / "src" / "minyeop" / "extra_models" / "predict_extra.py")
px = importlib.util.module_from_spec(spec)
spec.loader.exec_module(px)


def make_runner(name, device):
    s = px.MODELS[name]
    if s["kind"] == "ultra":
        import ultralytics
        model = getattr(ultralytics, s["cls"])(str(s["path"]))
        n_par = sum(p.numel() for p in model.model.parameters())

        def run(im):
            model.predict(np.stack([im] * 3, -1), imgsz=s["imgsz"], conf=0.001, max_det=s["max_det"], device=device, verbose=False)
        return run, n_par
    import torch.nn.functional as F
    from transformers import DFineForObjectDetection
    model = DFineForObjectDetection.from_pretrained(str(s["path"])).to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def run(im):
        x = F.interpolate(torch.from_numpy(im)[None, None].float().div(255).to(device), size=(s["imgsz"],) * 2, mode="bilinear", align_corners=False).expand(1, 3, -1, -1)
        out = model(pixel_values=x)
        sc = out.logits.sigmoid().max(-1)[0]
        _ = (out.pred_boxes[sc >= 0.001]).cpu()
    return run, n_par


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/minyeop/09_extra_models_v1")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    paths = sorted((ROOT / "data" / "test" / "images").glob("*.png"))[::6][:a.n + 10]
    imgs = px.load_images(paths)
    res = {"설정": {"이미지": len(imgs) - 10, "워밍업": 10, "배치": 1, "정밀도": "fp32", "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, "모델": {}}
    for name in ("yolo26n", "rtdetr_l", "dfine_n"):
        res["모델"][name] = {}
        for label, device, threads in (("GPU", "cuda:0", None), ("CPU 1스레드", "cpu", 1), ("CPU 4스레드", "cpu", 4)):
            if threads:
                torch.set_num_threads(threads)
            run, n_par = make_runner(name, device)
            res["모델"][name]["파라미터(M)"] = n_par / 1e6
            ts = []
            for i, im in enumerate(imgs):
                if device != "cpu":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                run(im)
                if device != "cpu":
                    torch.cuda.synchronize()
                if i >= 10:
                    ts.append((time.perf_counter() - t0) * 1000)
            res["모델"][name][label] = {"중앙값(ms)": float(np.median(ts)), "평균(ms)": float(np.mean(ts))}
            print(f"{name:9} {label:11} 중앙값 {np.median(ts):7.1f} ms, 평균 {np.mean(ts):7.1f} ms | 파라미터 {n_par / 1e6:.1f}M", flush=True)
    (ROOT / a.out / "speed_extra.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", ROOT / a.out / "speed_extra.json")


if __name__ == "__main__":
    main()
