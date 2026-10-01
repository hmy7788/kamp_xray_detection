"""실험 1건 실행: 학습 -> val/test 평가 -> outputs/runs/<name>/ 에 결과 저장.

예)
  python scripts/run_experiment.py --name spp_coco --cfg yolov3-spp.cfg \
      --weights weights/pretrained/yolov3-spp-ultralytics.pt --epochs 100 --batch-size 8

산출물(outputs/runs/<name>/): weights/{last,best}.pt, results.txt, results.png,
  train.log, val_eval.log, test_eval.log, metrics.json, config.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / "third_party" / "yolov3"


def run(cmd, log, env):
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=YOLO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for raw in p.stdout:
            f.write(raw.decode("utf-8", "replace"))
        p.wait()
    return p.returncode


def parse_metrics(log):
    """test.py 로그의 'all  N  targets  P  R  mAP  F1' 행을 파싱한다."""
    text = Path(log).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    m = None
    for m in re.finditer(r"\ball\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", text):
        pass
    if not m:
        return None
    n_img, n_tgt, p, r, ap, f1 = m.groups()
    return {"images": int(n_img), "targets": int(n_tgt), "precision": float(p), "recall": float(r),
            "mAP@0.5": float(ap), "F1": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="yolov3-spp.cfg")
    ap.add_argument("--weights", required=True, help="KAMP 루트 기준 상대경로 또는 절대경로")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, nargs="+", default=[320, 640])
    ap.add_argument("--device", default="")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="train.py 에 그대로 전달할 추가 인자")
    a = ap.parse_args()

    out = ROOT / "outputs" / "runs" / a.name
    if out.exists():
        sys.exit(f"이미 존재하는 실험 폴더입니다(덮어쓰기 방지): {out}")
    (out / "weights").mkdir(parents=True)
    weights = Path(a.weights) if Path(a.weights).is_absolute() else ROOT / a.weights
    env = {**os.environ, "PYTHONUTF8": "1", "YOLO_SAVE_DIR": str(out)}
    dev = ["--device", a.device] if a.device else []
    (out / "config.json").write_text(json.dumps({**vars(a), "weights_abs": str(weights)}, ensure_ascii=False, indent=2), encoding="utf-8")

    splits = ROOT / "data" / "splits"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_split.py")], check=True, capture_output=True)

    t0 = time.time()
    rc = run([sys.executable, "-u", "train.py", "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
              "--weights", str(weights), "--cfg", a.cfg, "--data", str(splits / "val.data"),
              "--img-size", *map(str, a.img_size), *dev, *a.extra], out / "train.log", env)
    hours = (time.time() - t0) / 3600
    if rc != 0:
        sys.exit(f"학습 실패(rc={rc}). {out / 'train.log'} 확인")
    for f in ("results.png",):
        if (YOLO / f).exists():
            shutil.move(str(YOLO / f), out / f)
    for f in YOLO.glob("test_batch*.jpg"):
        f.unlink()

    n_done = len((out / "results.txt").read_text().splitlines()) if (out / "results.txt").exists() else 0
    if n_done < a.epochs:  # 오류 없이 중간에 끝난 경우도 걸러낸다
        sys.exit(f"학습이 {n_done}/{a.epochs} epoch에서 끝났습니다(불완전). {out / 'train.log'} 확인")

    ckpt = out / "weights" / ("best.pt" if (out / "weights" / "best.pt").exists() else "last.pt")
    metrics = {"checkpoint": ckpt.name, "train_hours": round(hours, 3)}
    for split in ("val", "test"):
        run([sys.executable, "-u", "test.py", "--cfg", a.cfg, "--batch-size", str(a.batch_size),
             "--data", str(splits / f"{split}.data"), "--weights", str(ckpt), *dev], out / f"{split}_eval.log", env)
        metrics[split] = parse_metrics(out / f"{split}_eval.log")
    for f in YOLO.glob("test_batch*.jpg"):
        f.unlink()
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
