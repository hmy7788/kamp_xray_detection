"""전처리 버전을 만든다. 원본을 읽는 허용 스크립트다 (docs/contracts/data.md 4절).

사용: python common/preprocess.py v1_fill
버전 이름 = <번호>_<방법>. 방법:
  raw     색상 박스를 지우지 않음 (회색조 변환만). 리키지 측정 전용
  fill    색 픽셀을 주변 회색 픽셀 평균으로 메움
  inpaint OpenCV 인페인팅 (Telea)
  mask    색 픽셀을 사진의 배경 중앙값(회색)으로 덮음

색 픽셀 = RGB 채널 차이가 0보다 큰 픽셀. 사진은 팔레트 모드라 회색은 R=G=B 다.
결과는 유효 사진 전부(라벨 유무 무관), 회색조 PNG, 원본과 같은 크기.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402

METHODS = ("raw", "fill", "inpaint", "mask")


def color_mask(rgb: np.ndarray) -> np.ndarray:
    """색이 있는 픽셀 (채도 > 0)."""
    return (rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int)) > 0


def to_gray(rgb: np.ndarray) -> np.ndarray:
    # 회색 픽셀은 R=G=B 이므로 어느 채널이든 같다. 색 픽셀은 뒤에서 바꾼다.
    return rgb[:, :, 1].copy()


def fill_neighbors(gray: np.ndarray, mask: np.ndarray, k: int = 5) -> np.ndarray:
    """정규화 합성곱으로 색 픽셀을 주변 유효 픽셀 평균으로 메운다. 남으면 반복."""
    g = gray.astype(np.float32)
    valid = (~mask).astype(np.float32)
    out = g * valid
    todo = mask.copy()
    kernel = np.ones((k, k), np.float32)
    for _ in range(20):
        if not todo.any():
            break
        num = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(valid, -1, kernel, borderType=cv2.BORDER_REFLECT)
        ok = todo & (den > 0)
        out[ok] = num[ok] / den[ok]
        valid[ok] = 1.0
        todo &= ~ok
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def process(rgb: np.ndarray, method: str) -> np.ndarray:
    gray = to_gray(rgb)
    if method == "raw":
        # 색 픽셀은 밝기(L)로 바꿔 넣는다. 박스는 그대로 보인다.
        m = color_mask(rgb)
        lum = np.asarray(Image.fromarray(rgb).convert("L"))
        gray[m] = lum[m]
        return gray
    m = color_mask(rgb)
    if method == "fill":
        return fill_neighbors(gray, m)
    if method == "inpaint":
        return cv2.inpaint(gray, m.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    if method == "mask":
        bg = int(np.median(gray[~m])) if (~m).any() else 0
        out = gray.copy()
        out[m] = bg
        return out
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="예: v1_fill")
    ap.add_argument("--limit", type=int, default=0, help="테스트용 장수 제한")
    a = ap.parse_args()
    method = a.version.split("_", 1)[-1]
    if method not in METHODS:
        sys.exit(f"방법은 {METHODS} 중 하나. 버전 이름 예: v1_fill")

    raw = kx.raw_root()
    df = kx.load_manifest()
    if a.limit:
        df = df.head(a.limit)
    vdir = kx.VERSIONS / a.version
    if (vdir / "PASS").exists():
        sys.exit(f"{vdir} 는 이미 검사를 통과한 버전이다. 덮어쓰지 않는다. 새 버전 이름을 써라.")
    (vdir / "images").mkdir(parents=True, exist_ok=True)
    (vdir / "labels").mkdir(parents=True, exist_ok=True)

    n_color_px = 0
    for i, r in enumerate(df.itertuples(index=False)):
        rgb = np.asarray(Image.open(raw / r.src_path).convert("RGB"))
        n_color_px += int(color_mask(rgb).sum())
        out = process(rgb, method)
        Image.fromarray(out, mode="L").save(vdir / "images" / f"{r.image_id}.png")
        if r.labeled == 1 and r.label_path:
            (vdir / "labels" / f"{r.image_id}.txt").write_text((raw / r.label_path).read_text())
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)}")

    meta = {
        "version": a.version,
        "method": method,
        "params": {"color_rule": "max(RGB)-min(RGB) > 0", "fill_kernel": 5, "inpaint_radius": 3},
        "manifest_sha256": kx.sha256_file(kx.MANIFEST),
        "script_sha256": kx.sha256_file(Path(__file__)),
        "created_by": kx.member(),
        "created_at": kx.now_iso(),
        "n_images": int(len(df)),
        "n_labeled": int(df.labeled.sum()),
        "color_pixels_in_source": n_color_px,
    }
    kx.write_json(vdir / "meta.json", meta)
    print(f"완료: {vdir} ({len(df)}장, 원본 색 픽셀 {n_color_px}개). 다음: python common/check_data_version.py {a.version}")


if __name__ == "__main__":
    main()
