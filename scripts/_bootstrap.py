"""scripts/*.py 공통 준비: src/ 를 import 경로에 넣고, 한글 출력과 UTF-8 기본 인코딩을 맞춘다."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows conda에서 OpenMP 중복 로드 오류 방지

from kamp_xray.common import setup_console  # noqa: E402

setup_console()
