"""data/{train,val,test} (YOLO txt 라벨)를 torchvision 검출 모델용 데이터셋으로 읽는다.

- 이미지: 회색조 PNG -> 3채널로 복제 (COCO 사전학습 모델이 3채널 입력을 기대)
- 라벨: 0 cx cy w h (0~1 비율) -> 픽셀 좌표 xyxy, 클래스는 1 (0은 배경)
- 빈 라벨(박스 0개) 이미지도 그대로 사용 (빈 타깃)
- PIL로 읽으므로 한글 경로도 문제없다
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"


class YoloDetDataset(torch.utils.data.Dataset):
    def __init__(self, split, train=False, data_dir=DATA, limit=None):
        self.images = sorted((Path(data_dir) / split / "images").glob("*.png"))
        if limit:
            step = max(len(self.images) // limit, 1)  # 앞쪽만 자르지 않고 고르게 뽑는다
            self.images = self.images[::step][:limit]
        self.label_dir = Path(data_dir) / split / "labels"
        self.train = train

    def __len__(self):
        return len(self.images)

    @property
    def names(self):
        return [p.name for p in self.images]

    def read_boxes(self, i, w, h):
        lab = self.label_dir / (self.images[i].stem + ".txt")
        boxes = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            if line.strip():
                _, cx, cy, bw, bh = map(float, line.split()[:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
        keep = (b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)  # 너비·높이가 1px 미만인 박스는 학습 오류의 원인
        return b[keep]

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        w, h = img.size
        boxes = self.read_boxes(i, w, h)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        if self.train and random.random() < 0.5:  # 좌우 반전
            x = x.flip(-1)
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([w, 0, w, 0])
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64), "image_id": torch.tensor(i)}
        return x, target


def collate(batch):
    return tuple(zip(*batch))
