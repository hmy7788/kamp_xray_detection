"""예측 CSV 형식 검사 (docs/contracts/experiment.md 3절).

사용: python common/check_preds.py runs/<exp_id>            preds_val.csv 를 분할의 val 목록과 대조
      python common/check_preds.py runs/<exp_id> --final    preds_test.csv 를 test 목록과 대조 (reproduce.py 만 쓴다)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402

COLS = ["image_id", "seed", "cx", "cy", "w", "h", "conf"]


def check_file(csv: Path, ids: list[str], seeds: list[int]) -> list[str]:
    errs = []
    if not csv.exists():
        return [f"{csv.name} 없음"]
    df = pd.read_csv(csv, dtype={"image_id": str})
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        return [f"{csv.name}: 열 없음 {missing}"]
    for s in seeds:
        d = df[df.seed == s]
        absent = set(ids) - set(d.image_id)
        if absent:
            errs.append(f"{csv.name} seed {s}: 채점 대상 사진 {len(absent)}장이 빠짐 (빈 행이라도 있어야 함)")
        extra = set(d.image_id) - set(ids)
        if extra:
            errs.append(f"{csv.name} seed {s}: 채점 대상이 아닌 id {len(extra)}개 (리키지 의심)")
    filled = df.dropna(subset=["conf"])
    for c in ["cx", "cy", "w", "h", "conf"]:
        if ((filled[c] < 0) | (filled[c] > 1)).any():
            errs.append(f"{csv.name}: {c} 가 0~1 밖")
    if (filled[["w", "h"]] <= 0).any().any():
        errs.append(f"{csv.name}: w 또는 h 가 0 이하")
    return errs


def main() -> int:
    rd = Path(sys.argv[1])
    final = "--final" in sys.argv
    cfg = kx.read_yaml(rd / "config.yaml")
    seeds = [int(s) for s in cfg["seeds"]]
    if final:
        split = kx.load_split(allow_test=True)
        errs = check_file(rd / "preds_test.csv", split["test"], seeds)
    else:
        split = kx.load_split()
        errs = check_file(rd / "preds_val.csv", split["val"], seeds)
    if errs:
        print("FAIL:")
        for e in errs:
            print("  -", e)
        return 1
    print(f"PASS: {rd.name} 예측 형식")
    return 0


if __name__ == "__main__":
    sys.exit(main())
