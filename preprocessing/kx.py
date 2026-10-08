"""공용 도우미. 경로, 해시, manifest/분할/설정 읽기.

모든 common 스크립트와 멤버 코드는 여기서 경로를 얻는다. 원본(raw) 경로는
build_manifest.py / preprocess.py / eda_raw.py 만 쓴다 (docs/contracts/data.md).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.csv"
MANIFEST_SHA = DATA / "manifest.sha256"
SPLITS = DATA / "splits"
VERSIONS = DATA / "versions"
CONDITIONS_CSV = DATA / "conditions.csv"
COMMON = ROOT / "common"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

RAW_ALLOWED_SCRIPTS = ("build_manifest.py", "preprocess.py", "eda_raw.py")


def raw_root() -> Path:
    """원본 폴더. data/raw_path.txt 의 첫 줄. 허용 스크립트만 부른다."""
    caller = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else ""
    if caller and caller not in RAW_ALLOWED_SCRIPTS:
        raise PermissionError(
            f"{caller} 는 원본을 읽을 수 없다. 허용: {RAW_ALLOWED_SCRIPTS} (docs/contracts/data.md)"
        )
    p = (DATA / "raw_path.txt").read_text(encoding="utf-8").strip().splitlines()[0].strip()
    root = Path(p)
    if not root.exists():
        raise FileNotFoundError(f"원본 폴더가 없다: {root} (data/raw_path.txt 를 고쳐라)")
    return root


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


def sha256_dir_meta(version: str) -> str:
    """데이터 버전의 식별 해시 = meta.json 내용 해시."""
    meta = VERSIONS / version / "meta.json"
    return sha256_file(meta) if meta.exists() else ""


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


def load_manifest(valid_only: bool = True):
    """manifest.csv 를 DataFrame 으로. valid_only 면 dup_of 가 빈 행만."""
    import pandas as pd

    if not MANIFEST.exists():
        raise FileNotFoundError("data/manifest.csv 가 없다. python common/build_manifest.py 를 먼저 돌려라")
    df = pd.read_csv(MANIFEST, dtype={"image_id": str, "dup_of": str, "label_path": str, "burst_id": str})
    df["dup_of"] = df["dup_of"].fillna("")
    df["label_path"] = df["label_path"].fillna("")
    if valid_only:
        df = df[df["dup_of"] == ""].reset_index(drop=True)
    return df


def check_manifest_sha() -> bool:
    if not MANIFEST_SHA.exists():
        return False
    return MANIFEST_SHA.read_text().split()[0] == sha256_file(MANIFEST)


def load_split(version: str = "v1", allow_holdout: bool = False) -> dict:
    """분할 파일. allow_holdout 이 False 면 holdout 목록을 지워서 돌려준다 (실수 방지)."""
    p = SPLITS / f"{version}.json"
    sha_p = SPLITS / f"{version}.sha256"
    if not p.exists():
        raise FileNotFoundError(f"분할 파일이 없다: {p}")
    if sha_p.exists() and sha_p.read_text().split()[0] != sha256_file(p):
        raise RuntimeError(f"분할 파일이 잠금 해시와 다르다: {p}. 분할은 수정 금지다.")
    s = read_json(p)
    if not allow_holdout:
        s["holdout"] = None
    return s


def split_sha(version: str = "v1") -> str:
    p = SPLITS / f"{version}.sha256"
    return p.read_text().split()[0] if p.exists() else ""


def version_dir(version: str) -> Path:
    d = VERSIONS / version
    if not d.exists():
        raise FileNotFoundError(f"데이터 버전이 없다: {d}")
    return d


def version_passed(version: str) -> bool:
    return (VERSIONS / version / "PASS").exists()


def read_yolo_labels(path: Path) -> list[tuple[float, float, float, float]]:
    """YOLO txt -> [(cx, cy, w, h)] 정규화."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            out.append(tuple(float(x) for x in p[1:]))
    return out


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
