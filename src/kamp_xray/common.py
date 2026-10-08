"""공통 함수: 경로, 데이터 버전 확인, 매니페스트·정답 읽기, 예측 기록 저장.

예측 기록(record) 형식은 모든 모델이 같다. 그래서 채점(metrics.py)은 모델과 무관하다.
    {"image_id": "<manifest의 image_id>", "name": "<파일명>.png", "w": 가로, "h": 세로,
     "gt":   [[x1, y1, x2, y2], ...],              # 원본 픽셀 좌표
     "dets": [[x1, y1, x2, y2, score], ...]}       # 원본 픽셀 좌표, 신뢰도 0.001 이상 전부
"""
import csv
import hashlib
import json
import os
import platform
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # 저장소 루트
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
WEIGHTS = ROOT / "weights"                          # 제출 가중치(weights/<model>/) + 사전학습 가중치(weights/pretrained/)
RESULTS = ROOT / "results"                          # 제출 시점의 결과 (보고서 수치의 근거, 읽기 전용으로 취급)
OUTPUTS = ROOT / "outputs"                          # 재현 실행 결과가 새로 생기는 곳
YOLOV3_DIR = ROOT / "third_party" / "yolov3"

MANIFEST_SHA256 = "1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623"
SPLIT_SIZES = {"train": 1767, "val": 369, "test": 396}
BOX_COUNTS = {"train": 3225, "val": 606, "test": 663}

MODELS = {
    "dfine_n": "D-FINE-N (최종 모델)",
    "yolov3_tiny": "YOLOv3-tiny (베이스라인)",
}


def setup_console():
    """Windows 콘솔에서 한글 출력이 깨지지 않게 한다."""
    import sys
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")


def manifest_sha256():
    """manifest.csv의 sha256. Git의 줄바꿈 변환(CRLF)에 영향받지 않도록 \r\n을 \n으로 바꿔 계산한다."""
    raw = (DATA / "manifest.csv").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def check_data_version():
    digest = manifest_sha256()
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"데이터 버전이 다릅니다: manifest sha256 {digest} (기대값 {MANIFEST_SHA256})")
    return digest


def read_manifest():
    with open(DATA / "manifest.csv", encoding="utf-8", newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def split_ids(manifest, split):
    return sorted(i for i, r in manifest.items() if r["split"] == split)


def read_gt(label_path, w, h):
    """YOLO txt(0 cx cy w h, 0~1 비율) → 픽셀 xyxy 목록."""
    out = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split()[:5])
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def imread(path, flags=None):
    """한글 경로에서도 동작하는 이미지 읽기 (cv2.imread는 Windows 한글 경로를 못 읽는다)."""
    import cv2
    flags = cv2.IMREAD_COLOR if flags is None else flags
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def make_records(manifest, ids, dets_list):
    """image_id 목록과 모델 검출 결과로 공통 예측 기록을 만든다."""
    recs = []
    for i, dets in zip(ids, dets_list):
        r = manifest[i]
        w, h = int(r["width"]), int(r["height"])
        recs.append({"image_id": i, "name": Path(r["image_path"]).name, "w": w, "h": h,
                     "gt": read_gt(DATA / r["label_path"], w, h),
                     "dets": [[float(v) for v in d] for d in dets]})
    return recs


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device):
    import torch
    if device != "cpu" and torch.cuda.is_available():
        return torch.device(f"cuda:{device}" if str(device).isdigit() else device)
    return torch.device("cpu")


def env_info():
    info = {"python": platform.python_version(), "os": f"{platform.system()} {platform.release()}"}
    for mod in ("torch", "torchvision", "numpy", "cv2", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["gpu"] = None
    return info


def sanitize_paths(path):
    """로그·설정 파일의 개인 PC 경로를 <ROOT>, <HOME>으로 바꾼다 (사용자 폴더 이름 등 식별 정보 제거)."""
    path = Path(path)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for p, tag in ((ROOT, "<ROOT>"), (Path.home(), "<HOME>")):
        for form in {str(p), p.as_posix(), str(p).replace("\\", "\\\\")}:
            text = re.sub(re.escape(form), tag, text, flags=re.IGNORECASE)
    path.write_text(text, encoding="utf-8")


def default_weights(model):
    """제출 가중치 위치. 직접 학습했다면 outputs/<model>/weights/... 를 --weights로 넘긴다."""
    return {"dfine_n": WEIGHTS / "dfine_n" / "best", "yolov3_tiny": WEIGHTS / "yolov3_tiny" / "best.pt"}[model]


def keep_awake(on):
    """긴 학습 중 Windows 절전을 막는다 (시스템 설정은 바꾸지 않음)."""
    if os.name != "nt":
        return
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
