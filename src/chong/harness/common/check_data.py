"""확정 데이터 검사. 통과하면 data/PASS 를 만든다. run.py 는 PASS 가 있어야 돈다.

사용: python common/check_data.py

검사:
- data/manifest.csv 가 잠금 해시(manifest.sha256)와 같다 (세 사람이 같은 목록)
- manifest 의 사진 2,532장이 split 열대로 data/<split>/images/ 에 전부 있고 크기가 manifest 와 같다
- 라벨 2,532개가 data/<split>/labels/ 에 전부 있고 형식이 맞다 ("0 cx cy w h", 0~1)
- 폴더에 manifest 에 없는 사진·라벨이 없다
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def label_errors(path: Path) -> list[str]:
    errs = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        p = line.split()
        if not p:
            continue
        if len(p) != 5 or p[0] != "0":
            errs.append(f"{path.name}:{n} 형식")
            continue
        try:
            cx, cy, w, h = (float(x) for x in p[1:])
        except ValueError:
            errs.append(f"{path.name}:{n} 숫자 아님")
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errs.append(f"{path.name}:{n} 범위 밖")
    return errs


def main() -> int:
    errs: list[str] = []
    if not kx.check_manifest_sha():
        errs.append("data/manifest.csv 가 잠금 해시와 다르다. manifest 는 고정 파일이다. 저장소를 다시 받아라")
    df = kx.load_manifest()
    if "split" not in df.columns or not set(df.split.unique()) <= set(kx.PARTS):
        errs.append("manifest 에 split 열(train/val/test)이 없거나 값이 이상하다")

    n_img_missing = n_size = n_lab_missing = 0
    for r in df.itertuples(index=False):
        p = kx.DATA / r.image_path
        if not p.exists():
            n_img_missing += 1
        else:
            with Image.open(p) as im:
                if im.size != (int(r.width), int(r.height)):
                    n_size += 1
        lp = kx.DATA / r.label_path
        if not lp.exists():
            n_lab_missing += 1
        else:
            errs += label_errors(lp)[:3]
    if n_img_missing:
        errs.append(f"사진 {n_img_missing}장이 없다. data/kamp_data.zip 을 저장소 폴더에 풀었는지 확인")
    if n_size:
        errs.append(f"사진 {n_size}장의 크기가 manifest 와 다르다")
    if n_lab_missing:
        errs.append(f"라벨 {n_lab_missing}개가 없다")

    known = set(df.image_id)
    for part in kx.PARTS:
        extra_i = [p.stem for p in (kx.DATA / part / "images").glob("*.png") if p.stem not in known]
        extra_l = [p.stem for p in (kx.DATA / part / "labels").glob("*.txt") if p.stem not in known]
        if extra_i or extra_l:
            errs.append(f"{part}: manifest 에 없는 파일 사진 {len(extra_i)} 라벨 {len(extra_l)}")

    for part in kx.PARTS:
        g = df[df.split == part]
        print(f"{part:5s}: {len(g):4d}장 | 장비 {g.groupby('machine').size().to_dict()} | 출처 {g.groupby('source').size().to_dict()}")
    print(f"manifest sha256: {kx.manifest_sha()[:12]}...")
    if errs:
        print("FAIL:")
        for e in errs[:20]:
            print("  -", e)
        if kx.DATA_PASS.exists():
            kx.DATA_PASS.unlink()
        return 1
    kx.DATA_PASS.write_text(f"{kx.now_iso()} manifest {kx.manifest_sha()} images {len(df)}\n")
    print(f"PASS → {kx.DATA_PASS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
