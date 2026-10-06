"""data/ 를 팀원에게 보낼 zip 으로 묶는다 (train/val/test 사진·라벨 + manifest + README.txt). 풀면 data/ 아래에 그대로 들어간다.

사용: python common/pack_data.py [--out ../../share/kamp_data.zip]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(kx.ROOT.parents[1] / "share" / "kamp_data.zip"))
    a = ap.parse_args()
    if not kx.check_manifest_sha():
        sys.exit("manifest 잠금 해시가 다르다. 확정 데이터가 아니다.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    import subprocess
    subprocess.call([sys.executable, str(kx.COMMON / "data_summary.py")], stdout=subprocess.DEVNULL)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for part in kx.PARTS:
            for p in sorted((kx.DATA / part).rglob("*")):
                if p.is_file():
                    z.write(p, Path("data") / p.relative_to(kx.DATA))
                    n += 1
        for name in ("manifest.csv", "manifest.sha256", "split_info.json", "README.txt"):
            z.write(kx.DATA / name, Path("data") / name)
            n += 1
    print(f"{out} ({out.stat().st_size / 2**20:.0f} MB, 파일 {n}개)")
    print("팀원: 저장소 폴더에서 압축을 풀면 data/ 아래에 들어간다 → python common/check_data.py")


if __name__ == "__main__":
    main()
