"""환경 정보를 출력하고, --lock 이면 environment.lock 에 패키지 버전을 고정해 적는다.

사용: python common/env_info.py [--lock]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402

PKGS = ["numpy", "pandas", "pillow", "opencv-python", "scipy", "scikit-learn", "scikit-image", "pyyaml",
        "matplotlib", "torch", "torchvision", "ultralytics"]


def main() -> None:
    info = kx.env_info()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if "--lock" in sys.argv:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        lines = [l for l in out.splitlines() if l.split("==")[0].lower().replace("_", "-") in PKGS]
        lock = kx.ROOT / "environment.lock"
        lock.write_text(f"# python {info['python']} / cuda {info.get('cuda', '')} / {kx.now_iso()} / {kx.member()}\n"
                        + "\n".join(sorted(lines)) + "\n")
        print(f"저장: {lock}")


if __name__ == "__main__":
    main()
