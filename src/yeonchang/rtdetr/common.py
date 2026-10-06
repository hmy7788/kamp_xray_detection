"""RT-DETR 실험 공통 함수: 경로, 데이터 버전 확인, 정답 읽기, 로컬 경로 치환."""
import csv
import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # 저장소 루트
DATA = ROOT / "data"
RUNS = ROOT / "runs" / "yeonchang"
WEIGHTS = ROOT / "weights"  # COCO 사전학습 가중치 (Git 제외)
MANIFEST_SHA256 = "1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623"


def check_data_version():
    """manifest.csv sha256이 확정 v2와 같은지 확인한다. Windows autocrlf 변환을 고려해 \\r을 빼고 계산한다."""
    raw = (DATA / "manifest.csv").read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"데이터 버전이 다릅니다: manifest sha256 {digest} (기대값 {MANIFEST_SHA256})")
    return digest


def read_manifest():
    with open(DATA / "manifest.csv", encoding="utf-8", newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def write_data_yaml(path):
    """ultralytics용 데이터 yaml. 절대경로가 들어가므로 Git 제외 폴더(weights/)에 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"path: {DATA.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n  0: defect\n",
        encoding="utf-8",
    )
    return path


def load_gt(split, manifest):
    """split의 정답 박스를 픽셀 xyxy로 읽는다. {image_id: [(box_idx, x1, y1, x2, y2), ...]}"""
    gt = {}
    for image_id, r in manifest.items():
        if r["split"] != split:
            continue
        w, h = int(r["width"]), int(r["height"])
        boxes = []
        text = (DATA / r["label_path"]).read_text(encoding="utf-8")
        for i, line in enumerate(l for l in text.splitlines() if l.strip()):
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append((i, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        gt[image_id] = boxes
    return gt


def sanitize_paths(path):
    """로그·설정 파일의 로컬 경로를 <KAMP>, <HOME>으로 치환한다 (사용자 폴더 이름에 소속이 드러날 수 있음)."""
    path = Path(path)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for p, tag in ((ROOT, "<KAMP>"), (Path.home(), "<HOME>")):
        for form in {str(p), p.as_posix(), str(p).replace("\\", "\\\\")}:
            text = re.sub(re.escape(form), tag, text, flags=re.IGNORECASE)
    path.write_text(text, encoding="utf-8")


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def env_info():
    import platform
    info = {"python": platform.python_version(), "gpu": gpu_name()}
    for mod in ("torch", "ultralytics", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    info["PYTHONUTF8"] = os.environ.get("PYTHONUTF8")
    return info
