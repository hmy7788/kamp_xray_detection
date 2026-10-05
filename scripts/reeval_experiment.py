"""끝난 실험을 학습 없이 다시 평가한다 (config.json 의 설정 사용).

  python scripts/reeval_experiment.py 01_yolov3spp_coco [--img-size 640]

기본 평가 해상도는 학습 때의 --img-size 마지막 값이다. 기존 평가 로그/지표는
val_eval_img<해상도>.log, metrics_img<해상도>.json 으로 보존하고 metrics.json 을 새 결과로 바꾼다.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import ROOT, parse_metrics, run_eval  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--img-size", type=int, default=None)
    a = ap.parse_args()

    out = ROOT / "runs" / a.name
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    img = a.img_size or cfg["img_size"][-1]
    ckpt = out / "weights" / ("best.pt" if (out / "weights" / "best.pt").exists() else "last.pt")
    env = {**os.environ, "PYTHONUTF8": "1", "YOLO_SAVE_DIR": str(out)}
    dev = ["--device", cfg["device"]] if cfg.get("device") else []

    if (out / "metrics.json").exists():
        old = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        old_img = old.get("eval_img_size", 512)  # 이 필드가 없는 옛 결과는 test.py 기본값(512)으로 평가된 것
        if old_img == img:
            sys.exit(f"이미 {img} 해상도로 평가된 결과입니다.")
        shutil.copy2(out / "metrics.json", out / f"metrics_img{old_img}.json")
    else:  # 기록용 프로세스가 중간에 죽어 평가가 안 된 실험을 마무리하는 경우
        old, old_img = {}, None

    metrics = {"checkpoint": ckpt.name, "train_hours": old.get("train_hours"), "eval_img_size": img}
    for split in ("val", "test"):
        if old_img is not None and (out / f"{split}_eval.log").exists():
            shutil.move(str(out / f"{split}_eval.log"), out / f"{split}_eval_img{old_img}.log")
        rc = run_eval([sys.executable, "-u", "test.py", "--cfg", cfg["cfg"], "--batch-size", str(cfg["batch_size"]),
                       "--img-size", str(img), "--data", str(ROOT / "data" / ("splits_nomark" if cfg.get("dataset") == "nomark" else "splits") / f"{split}.data"),
                       "--weights", str(ckpt), *dev], out / f"{split}_eval.log", env)
        metrics[split] = parse_metrics(out / f"{split}_eval.log")
        m = metrics[split]
        print(f"{a.name} {split} @{img}: " + (f"P {m['precision']:.3f} R {m['recall']:.3f} mAP@0.5 {m['mAP@0.5']:.3f} F1 {m['F1']:.3f}" if m else f"실패(rc={rc})"), flush=True)
    for f in (ROOT / "src" / "yolov3").glob("test_batch*.jpg"):
        f.unlink()
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
