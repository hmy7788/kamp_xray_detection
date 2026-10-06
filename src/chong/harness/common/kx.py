"""공용 도우미. 경로, 해시, manifest/설정 읽기.

데이터는 확정본이다 (docs/contracts/data.md). data/train, data/val, data/test 아래에 images/, labels/ 가 있고,
어느 사진이 어디에 있는지는 data/manifest.csv 의 split 열이 기준이다. 모든 코드는 여기서 경로를 얻는다.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.csv"
MANIFEST_SHA = DATA / "manifest.sha256"
SPLIT_INFO = DATA / "split_info.json"
DATA_PASS = DATA / "PASS"
CONDITIONS_CSV = DATA / "conditions.csv"
COMMON = ROOT / "common"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

PARTS = ("train", "val", "test")
MEMBERS = ("lee", "jung", "heo")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def member() -> str:
    m = os.environ.get("KAMP_MEMBER", "").strip()
    if not m:
        raise EnvironmentError("환경 변수 KAMP_MEMBER 가 없다. 예: $env:KAMP_MEMBER = 'lee'")
    return m


def read_yaml(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, obj) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------- 데이터 ----------------

def load_manifest():
    """data/manifest.csv 를 DataFrame 으로. 한 줄이 사진 한 장, 2,532장 전부 유효. split 열이 train/val/test."""
    import pandas as pd

    if not MANIFEST.exists():
        raise FileNotFoundError("data/manifest.csv 가 없다. 저장소를 다시 받아라 (고정 파일이다)")
    df = pd.read_csv(MANIFEST, dtype={"image_id": str, "burst_id": str, "labeler": str, "machine": str})
    df["labeler"] = df["labeler"].fillna("")
    return df


def check_manifest_sha() -> bool:
    return MANIFEST_SHA.exists() and MANIFEST_SHA.read_text().split()[0] == sha256_file(MANIFEST)


def manifest_sha() -> str:
    return MANIFEST_SHA.read_text().split()[0] if MANIFEST_SHA.exists() else ""


def data_passed() -> bool:
    """python common/check_data.py 가 통과하면 data/PASS 가 생긴다."""
    return DATA_PASS.exists()


def image_path(image_id: str, split: str | None = None) -> Path:
    """사진 경로 data/<split>/images/<id>.png. split 을 모르면 세 폴더에서 찾는다."""
    if split:
        return DATA / split / "images" / f"{image_id}.png"
    for part in PARTS:
        p = DATA / part / "images" / f"{image_id}.png"
        if p.exists():
            return p
    raise FileNotFoundError(f"사진이 없다: {image_id}")


def label_path(image_id: str, split: str | None = None) -> Path:
    """라벨 경로 data/<split>/labels/<id>.txt."""
    if split:
        return DATA / split / "labels" / f"{image_id}.txt"
    return image_path(image_id).parent.parent / "labels" / f"{image_id}.txt"


def path_maps(ids=None) -> tuple[dict[str, Path], dict[str, Path]]:
    """image_id -> 사진 경로, image_id -> 라벨 경로. ids 가 None 이면 전부."""
    df = load_manifest()
    if ids is not None:
        df = df[df.image_id.isin(set(ids))]
    imgs = {r.image_id: DATA / r.image_path for r in df.itertuples(index=False)}
    labs = {r.image_id: DATA / r.label_path for r in df.itertuples(index=False)}
    return imgs, labs


def read_yolo_labels(path: Path) -> list[tuple[float, float, float, float]]:
    """YOLO txt -> [(cx, cy, w, h)] 정규화. 파일이 없으면 빈 리스트."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 5:
            out.append(tuple(float(x) for x in p[1:]))
    return out


# ---------------- 분할 ----------------

def load_split(allow_test: bool = False) -> dict:
    """manifest 의 split 열에서 {train, val, test} 사진 목록. allow_test 가 False 면 test 는 None (실수 방지)."""
    if not check_manifest_sha():
        raise RuntimeError("data/manifest.csv 가 잠금 해시와 다르다. 확정 데이터가 아니다.")
    df = load_manifest()
    s = {part: sorted(df[df.split == part].image_id.tolist()) for part in PARTS}
    if not allow_test:
        s["test"] = None
    return s


# ---------------- 환경 ----------------

def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def env_info() -> dict:
    import platform

    info = {"os": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or ""
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["torch"] = ""
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info
