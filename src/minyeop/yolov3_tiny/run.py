"""YOLOv3-tiny 학습 실행기. 공유 코드(src/yolov3)는 건드리지 않고, 실험 폴더 runs/minyeop/<name>/ 에서 학습을 돌린다.

  python src/minyeop/yolov3_tiny/run.py --name 02_yolov3tiny_img640_v1 --epochs 100 --batch-size 16

- 실험 폴더 안에서 실행하므로 산출물(weights/, results.txt, *_batch*.jpg)이 모두 그 폴더에 생긴다.
- 데이터 설정(kamp.data)은 실험 폴더에 **상대 경로**로 만든다 (개인 PC 경로가 Git 에 올라가지 않게).
- 학습 로그는 train.log 에 저장되고, 진행은 watch.py 로 본다. test 는 쓰지 않는다 (valid = val).
- best.pt 는 train.py 가 val 의 0.99*mAP@0.5 + 0.01*R 로 고른다 (Faster R-CNN 과 같이 val 기준).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
YOLO = ROOT / "src" / "yolov3"


def keep_awake(on):
    """학습 중 Windows 절전을 막는다 (시스템 설정은 바꾸지 않음)."""
    if sys.platform != "win32":
        return
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="실험 이름 (결과: runs/<owner>/<name>/)")
    ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--cfg", default="yolov3-tiny.cfg")
    ap.add_argument("--weights", default="weights/pretrained/yolov3-tiny.pt", help="COCO 사전학습 가중치 (저장소 루트 기준)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=640, help="학습·검증 입력 크기 (고정)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="train.py 에 그대로 넘길 추가 인자")
    a = ap.parse_args()

    run = ROOT / "runs" / a.owner / a.name
    if run.exists():
        sys.exit(f"이미 존재하는 실험 폴더입니다(덮어쓰기 방지): runs/{a.owner}/{a.name}")
    (run / "weights").mkdir(parents=True)

    rel = lambda p: Path(os.path.relpath(p, run)).as_posix()  # 실험 폴더 기준 상대 경로
    (run / "kamp.data").write_text(
        f"classes=1\ntrain={rel(ROOT / 'data/train/images')}\nvalid={rel(ROOT / 'data/val/images')}\n"
        f"names={rel(YOLO / 'classes.names')}\n", encoding="utf-8")
    cmd = [sys.executable, str(YOLO / "train.py"), "--cfg", rel(YOLO / a.cfg), "--data", "kamp.data",
           "--weights", rel(ROOT / a.weights), "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
           "--img-size", str(a.img_size), str(a.img_size), str(a.img_size), "--device", a.device, *a.extra]
    # 설정 기록 (경로는 모두 상대 경로)
    shown = [Path(c).name if c == cmd[1] else c for c in cmd]
    (run / "config.json").write_text(json.dumps(
        {"name": a.name, "model": "yolov3-tiny (ultralytics 2020, COCO 사전학습)", "epochs": a.epochs, "batch_size": a.batch_size,
         "img_size": a.img_size, "cfg": a.cfg, "weights": a.weights, "device": a.device,
         "data": {"train": "data/train/images", "valid": "data/val/images (test 는 학습에 쓰지 않음)"},
         "command": " ".join(shown[1:]), "started_at": datetime.now().isoformat(timespec="seconds"), "pid": os.getpid()},
        ensure_ascii=False, indent=2), encoding="utf-8")

    env = {**os.environ, "PYTHONUTF8": "1", "YOLO_SAVE_DIR": "."}
    t0 = time.time()
    keep_awake(True)
    try:
        with open(run / "train.log", "w", encoding="utf-8") as log:
            print(f"[{datetime.now():%H:%M:%S}] 학습 시작: {a.owner}/{a.name} (로그: train.log)", flush=True)
            p = subprocess.Popen(cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
            rc = p.wait()
    finally:
        keep_awake(False)
    sec = int(time.time() - t0)
    print(f"[{datetime.now():%H:%M:%S}] 종료 (코드 {rc}), 소요 {sec // 60}분 {sec % 60}초", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
