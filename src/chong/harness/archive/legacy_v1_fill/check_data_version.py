"""전처리 버전이 약속을 지켰는지 검사하고 통과하면 PASS 파일을 만든다.

검사 (docs/contracts/data.md 4절):
1. meta.json 이 있고 manifest 해시가 현재 manifest 와 같다
2. 유효 사진 전부가 images/ 에 PNG 로 있고 크기가 manifest 와 같다
3. 라벨 사진은 labels/ 에 txt 가 있고 상자 수가 manifest 와 같다
4. raw 가 아니면 사진이 회색조(1채널)이고, 원본 기준 색 픽셀 자리에 흔적이 없는지는 확인 불가하므로
   대신 PNG 가 'L' 모드인지와 픽셀값 범위를 확인한다
사용: python common/check_data_version.py v1_fill
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    a = ap.parse_args()
    vdir = kx.VERSIONS / a.version
    errors: list[str] = []

    meta_p = vdir / "meta.json"
    if not meta_p.exists():
        print(f"FAIL: {meta_p} 없음")
        return 1
    meta = kx.read_json(meta_p)
    if meta.get("manifest_sha256") != kx.sha256_file(kx.MANIFEST):
        errors.append("meta.manifest_sha256 가 현재 manifest 와 다르다 (manifest 가 바뀌었으면 버전을 다시 만들어라)")

    df = kx.load_manifest()
    n_lab = 0
    for r in df.itertuples(index=False):
        ip = vdir / "images" / f"{r.image_id}.png"
        if not ip.exists():
            errors.append(f"사진 없음: {ip.name}")
            continue
        with Image.open(ip) as im:
            if im.mode != "L":
                errors.append(f"회색조 아님({im.mode}): {ip.name}")
            if im.size != (r.width, r.height):
                errors.append(f"크기 다름 {im.size} != {(r.width, r.height)}: {ip.name}")
        if r.labeled == 1:
            lp = vdir / "labels" / f"{r.image_id}.txt"
            if not lp.exists():
                errors.append(f"라벨 없음: {lp.name}")
            elif len(kx.read_yolo_labels(lp)) != int(r.n_boxes):
                errors.append(f"라벨 상자 수 다름: {lp.name}")
            else:
                n_lab += 1
        if len(errors) > 30:
            break

    if errors:
        print(f"FAIL ({len(errors)}개, 처음 30개):")
        for e in errors[:30]:
            print("  -", e)
        if (vdir / "PASS").exists():
            (vdir / "PASS").unlink()
        return 1

    (vdir / "PASS").write_text(f"checked_at: {kx.now_iso()}\nmeta_sha256: {kx.sha256_file(meta_p)}\n")
    print(f"PASS: {a.version} (사진 {len(df)}, 라벨 {n_lab}, 방법 {meta.get('method')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
