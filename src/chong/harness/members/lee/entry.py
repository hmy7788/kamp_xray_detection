"""lee 의 모델 진입점 (ultralytics 계열). common/baselines/yolo_ref/entry.py 와 같은 방식이고, arch 처리만 넓다.

model 영역 (configs):
  arch: yolo26n          # 사전학습 가중치 이름 → YOLO("yolo26n.pt")
        yolo26n-p2.yaml  # 구조 yaml → YOLO(yaml) 에 같은 크기 사전학습 가중치(yolo26n.pt)를 옮겨 싣는다
  imgsz: 1024
  epochs: 40
  batch: 8
  conf_min: 0.001        # 예측 때 남길 최소 확신도 (채점이 임계값을 고른다)
  max_det: 50
  train_kwargs: {}       # ultralytics train() 에 그대로 넘길 추가 인자

약속 (docs/contracts/experiment.md 2절): 사진·라벨 경로는 data["images"], data["labels"] 로만 얻는다.
ultralytics 는 images/ ↔ labels/ 폴더 쌍이 필요하므로 out_dir/dataset/ 에 하드링크로 만든다.
이 파일을 직접 실행하지 않는다. 실행은 python common/run.py configs/lee/<exp_id>.yaml 뿐이다.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pandas as pd


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _build_dataset(data: dict, ds: Path) -> None:
    for part, ids in (("train", data["train_ids"]), ("val", data["val_ids"])):
        (ds / "images" / part).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / part).mkdir(parents=True, exist_ok=True)
        for i in ids:
            _link(data["images"][i], ds / "images" / part / f"{i}.png")
            lp = data["labels"][i]
            if lp.exists():
                _link(lp, ds / "labels" / part / f"{i}.txt")
    (ds / "data.yaml").write_text(
        f"path: {ds.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: defect\n", encoding="utf-8")


def _load_model(arch: str):
    """'yolo26n' → 사전학습 .pt. 'yolo26n-p2.yaml' → 구조 yaml + 같은 크기의 사전학습 가중치를 옮겨 싣는다."""
    from ultralytics import YOLO

    if arch.endswith(".yaml"):
        model = YOLO(arch)
        m = re.match(r"^(yolo(?:v?\d+)[nsmlx])", arch)
        if m:
            model.load(f"{m.group(1)}.pt")
        return model
    return YOLO(f"{arch}.pt")


PREDICT_CHUNK = 8


def predict_chunked(model, paths: list[str], imgsz: int, conf: float, max_det: int, chunk: int = PREDICT_CHUNK):
    """경로 목록을 chunk 장씩 끊어 예측한다. ultralytics 8.4 는 목록 전체를 한 묶음으로 GPU 에 올려서
    369장을 한 번에 넘기면 6GB GPU 에서 메모리가 터진다 (2026-10-05 lee_001 에서 확인)."""
    for s in range(0, len(paths), chunk):
        yield from model.predict(paths[s:s + chunk], imgsz=imgsz, conf=conf, max_det=max_det, verbose=False, stream=True)


def run_split(cfg: dict, data: dict, seed: int, out_dir: Path) -> Path:
    from ultralytics import YOLO

    m = cfg.get("model", {})
    arch = str(m.get("arch", "yolo26n"))
    imgsz = int(m.get("imgsz", 1024))
    epochs = int(m.get("epochs", 40))
    batch = int(m.get("batch", 8))
    conf_min = float(m.get("conf_min", 0.001))
    max_det = int(m.get("max_det", 50))

    ds = out_dir / "dataset"
    _build_dataset(data, ds)

    best = out_dir / "train" / "weights" / "best.pt"
    results = out_dir / "train" / "results.csv"
    done = results.exists() and best.exists() and len(results.read_text().splitlines()) - 1 >= epochs
    if done:
        # 학습이 이미 끝난 폴더 (예: 학습 직후 프로세스가 죽어 예측·채점만 남은 경우). 가중치를 그대로 쓴다.
        print(f"[entry] 학습 완료본이 있어 학습을 건너뛴다: {best}")
    else:
        model = _load_model(arch)
        model.train(data=str(ds / "data.yaml"), imgsz=imgsz, epochs=epochs, batch=batch, seed=seed,
                    deterministic=True, project=str(out_dir), name="train", exist_ok=True, verbose=False,
                    plots=False, workers=0, **dict(m.get("train_kwargs", {}) or {}))
    model = YOLO(str(best if best.exists() else out_dir / "train" / "weights" / "last.pt"))

    rows = []
    paths = [str(data["images"][i]) for i in data["val_ids"]]
    for iid, res in zip(data["val_ids"], predict_chunked(model, paths, imgsz, conf_min, max_det)):
        b = res.boxes
        n0 = len(rows)
        if b is not None and len(b):
            for (cx, cy, w, h), c in zip(b.xywhn.cpu().numpy(), b.conf.cpu().numpy()):
                if w > 0 and h > 0:
                    rows.append(dict(image_id=iid, cx=float(cx), cy=float(cy), w=float(w), h=float(h), conf=float(c)))
        if len(rows) == n0:
            rows.append(dict(image_id=iid, cx=None, cy=None, w=None, h=None, conf=None))
    out = out_dir / "preds.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    return out
