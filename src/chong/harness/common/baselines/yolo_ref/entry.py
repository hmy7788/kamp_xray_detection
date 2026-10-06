"""참조 기준 모델: KAMP 실습 자료와 같은 YOLO 계열, 기본 설정. (ultralytics)

model 영역 (configs):
  arch: yolov8n      # ultralytics 모델 이름. 첫 실행 때 사전학습 가중치를 내려받는다
  imgsz: 640
  epochs: 30
  batch: 16
  conf_min: 0.001    # 예측 때 남길 최소 확신도 (채점이 임계값을 고르므로 낮게)
  max_det: 50

ultralytics 는 사진 경로의 images/ 를 labels/ 로 바꿔 라벨을 찾는다. 우리 라벨은 labels/team/ 한 폴더라
그 규칙과 안 맞으므로, out_dir/dataset/ 에 images/, labels/ 를 하드링크로 만들어 넘긴다 (복사 없음).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)          # 같은 드라이브면 하드링크 (공간 0)
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


def run_split(cfg: dict, data: dict, seed: int, out_dir: Path) -> Path:
    from ultralytics import YOLO

    m = cfg.get("model", {})
    arch = m.get("arch", "yolov8n")
    imgsz = int(m.get("imgsz", 640))
    epochs = int(m.get("epochs", 30))
    batch = int(m.get("batch", 16))
    conf_min = float(m.get("conf_min", 0.001))
    max_det = int(m.get("max_det", 50))

    ds = out_dir / "dataset"
    _build_dataset(data, ds)

    model = YOLO(f"{arch}.pt")
    model.train(data=str(ds / "data.yaml"), imgsz=imgsz, epochs=epochs, batch=batch, seed=seed, deterministic=True,
                project=str(out_dir), name="train", exist_ok=True, verbose=False, plots=False, workers=0,
                **{k: v for k, v in m.get("train_kwargs", {}).items()})
    best = out_dir / "train" / "weights" / "best.pt"
    model = YOLO(str(best if best.exists() else out_dir / "train" / "weights" / "last.pt"))

    rows = []
    paths = [str(data["images"][i]) for i in data["val_ids"]]
    for iid, res in zip(data["val_ids"], model.predict(paths, imgsz=imgsz, conf=conf_min, max_det=max_det,
                                                       verbose=False, stream=True)):
        b = res.boxes
        if b is None or len(b) == 0:
            rows.append(dict(image_id=iid, cx=None, cy=None, w=None, h=None, conf=None))
            continue
        xywhn = b.xywhn.cpu().numpy()
        conf = b.conf.cpu().numpy()
        n0 = len(rows)
        for (cx, cy, w, h), c in zip(xywhn, conf):
            if w <= 0 or h <= 0:   # 크기 0 상자는 예측이 아니다 (초기 에폭에서 나온다)
                continue
            rows.append(dict(image_id=iid, cx=float(cx), cy=float(cy), w=float(w), h=float(h), conf=float(c)))
        if len(rows) == n0:
            rows.append(dict(image_id=iid, cx=None, cy=None, w=None, h=None, conf=None))
    out = out_dir / "preds.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    # 가중치는 저장소에 넣지 않는다 (.gitignore). 공간이 아까우면 train/weights/last.pt 를 지워도 된다.
    return out
