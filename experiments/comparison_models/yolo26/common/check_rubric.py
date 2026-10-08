"""docs/rubric_map.md 의 '근거 산출물' 칸에 적힌 파일이 있는지 확인한다.

사용: python common/check_rubric.py
경로는 쉼표로 나눈다. 괄호 설명과 '추가 예정'·'후보' 가 붙은 항목은 건너뛴다. 와일드카드(runs/*/metrics.json) 허용.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def main() -> int:
    text = (kx.ROOT / "docs" / "rubric_map.md").read_text(encoding="utf-8")
    missing_total = 0
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| 번호") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        num, item, _, artifacts, _ = cells[:5]
        found, missing = [], []
        for a in artifacts.split(","):
            a = re.sub(r"\(.*?\)", "", a).strip()
            if not a or "추가 예정" in a or "후보" in a or "/" not in a:
                continue
            a = a.split("의 ")[0].strip()
            if re.fullmatch(r"docs/decisions/\d{3}", a):   # 결정 번호만 적으면 NNN_*.md 로 찾는다
                a = a + "_*.md"
            if any(ch in a for ch in "*?"):
                hits = list(kx.ROOT.glob(a))
                (found if hits else missing).append(a)
            else:
                (found if (kx.ROOT / a).exists() else missing).append(a)
        state = "OK" if not missing else "미완"
        missing_total += len(missing)
        print(f"{num}. {item}: {state}  있음 {len(found)}  없음 {missing}")
    return 1 if missing_total else 0


if __name__ == "__main__":
    sys.exit(main())
