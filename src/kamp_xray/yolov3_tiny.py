"""베이스라인 YOLOv3-tiny 학습·추론. 모델 코드는 third_party/yolov3 (ultralytics YOLOv3 2020년 버전 수정본)를 쓴다.

원작성: 허민엽 (원 실험 이름 02_yolov3tiny_img640_v1). 제출용으로 경로와 설정 파일만 정리했고,
학습 명령과 추론(letterbox 640, NMS IoU 0.6, 신뢰도 0.001 이상 전부)은 제출 가중치를 만든 코드와 같다.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import (DATA, ROOT, YOLOV3_DIR, check_data_version, keep_awake, make_records, read_manifest,
                     sanitize_paths, select_device, split_ids)


def _import_yolov3():
    if str(YOLOV3_DIR) not in sys.path:
        sys.path.insert(0, str(YOLOV3_DIR))
    from models import Darknet  # noqa: E402
    from utils.datasets import LoadImages  # noqa: E402
    from utils.utils import non_max_suppression, scale_coords  # noqa: E402
    return Darknet, LoadImages, non_max_suppression, scale_coords


def train(hyp, out_dir, device="0", epochs=None):
    """third_party/yolov3/train.py 를 out_dir 안에서 실행한다 (산출물: weights/best.pt, last.pt, results.txt, train.log).
    best.pt 는 val 의 0.99*mAP@0.5 + 0.01*R 로 고른다. test 는 쓰지 않는다."""
    out_dir = Path(out_dir)
    check_data_version()
    (out_dir / "weights").mkdir(parents=True, exist_ok=False)
    rel = lambda p: Path(os.path.relpath(p, out_dir)).as_posix()      # 개인 PC 경로가 기록되지 않게 상대 경로 사용
    (out_dir / "kamp.data").write_text(
        f"classes=1\ntrain={rel(DATA / 'train/images')}\nvalid={rel(DATA / 'val/images')}\n"
        f"names={rel(YOLOV3_DIR / 'classes.names')}\n", encoding="utf-8")
    pretrained = ROOT / hyp["pretrained"]
    if not pretrained.exists():
        raise SystemExit(f"사전학습 가중치가 없습니다: {pretrained}\n  python scripts/download_pretrained.py 를 먼저 실행하세요.")
    n_epochs = epochs or hyp["epochs"]
    img = str(hyp["img_size"])
    dev = "cpu" if select_device(device).type == "cpu" else str(device)
    cmd = [sys.executable, str(YOLOV3_DIR / "train.py"), "--cfg", rel(YOLOV3_DIR / hyp["cfg"]), "--data", "kamp.data",
           "--weights", rel(pretrained), "--epochs", str(n_epochs), "--batch-size", str(hyp["batch_size"]),
           "--img-size", img, img, img, "--device", dev]
    config = {"model": "YOLOv3-tiny (ultralytics yolov3 2020, COCO 사전학습)", "hyp": {**hyp, "epochs": n_epochs},
              "command": " ".join(["train.py", *cmd[2:]]), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    env = {**os.environ, "PYTHONUTF8": "1", "YOLO_SAVE_DIR": "."}
    t0 = time.time()
    keep_awake(True)
    try:
        with open(out_dir / "train.log", "w", encoding="utf-8") as log:
            rc = subprocess.Popen(cmd, cwd=out_dir, env=env, stdout=log, stderr=subprocess.STDOUT).wait()
    finally:
        keep_awake(False)
    config.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_minutes": round((time.time() - t0) / 60, 1),
                   "return_code": rc})
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    sanitize_paths(out_dir / "train.log")
    if rc != 0:
        raise SystemExit(f"YOLOv3-tiny 학습 실패 (코드 {rc}). 로그: {out_dir / 'train.log'}")
    best, last = out_dir / "weights" / "best.pt", out_dir / "weights" / "last.pt"
    if not best.exists() and last.exists():
        # 원본 train.py 는 마지막 epoch 에는 best.pt 를 쓰지 않는다 (epoch 수가 아주 적은 동작 확인 실행에서만 생김)
        import shutil
        shutil.copy2(last, best)
        print("참고: best.pt 가 없어 last.pt 를 복사했습니다 (짧은 동작 확인 실행).")
    return best


def load_model(weights, device, cfg="yolov3-tiny.cfg", img_size=640):
    import torch
    Darknet, *_ = _import_yolov3()
    model = Darknet(str(YOLOV3_DIR / cfg), img_size)
    ck = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def predict(weights, split, device="0", img_size=640, nms_iou=0.6, cfg="yolov3-tiny.cfg"):
    """split 전체를 추론해 공통 예측 기록 목록을 돌려준다 (letterbox img_size, 신뢰도 0.001 이상 전부)."""
    import torch
    _, LoadImages, non_max_suppression, scale_coords = _import_yolov3()
    dev = select_device(device)
    model = load_model(weights, dev, cfg, img_size)
    manifest = read_manifest()
    ids = split_ids(manifest, split)
    dets_list = []
    with torch.no_grad():
        for i in ids:
            p = DATA / manifest[i]["image_path"]
            with contextlib.redirect_stdout(io.StringIO()):          # LoadImages 가 경로를 출력하는 것을 막는다
                _, img, im0, _ = next(iter(LoadImages(str(p), img_size=img_size)))
            x = torch.from_numpy(img).to(dev).float() / 255.0
            pred = model(x.unsqueeze(0))[0]
            det = non_max_suppression(pred, 0.001, nms_iou, multi_label=False)[0]
            dets = []
            if det is not None:
                det[:, :4] = scale_coords(x.shape[1:], det[:, :4], im0.shape).round()
                dets = [[float(v) for v in d[:5]] for d in det.cpu().numpy()]
            dets_list.append(dets)
    return make_records(manifest, ids, dets_list)
