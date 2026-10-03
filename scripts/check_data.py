"""data/subsets/n500 에 이미지·라벨이 올바르게 배치되었는지 점검한다.

manifest.csv(저장소에 포함)에 적힌 500장과 대조해 누락/초과 파일, 빈 라벨, bbox 합계를 확인한다.
실행: python scripts/check_data.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "subsets" / "n500"
EXPECT_BOX = 1147  # 라벨된 500장의 bbox 합계


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8")))
    names = {r["name"] for r in rows}
    imgs = {p.name for p in (SRC / "images").glob("*")} if (SRC / "images").exists() else set()
    labs = {p.stem for p in (SRC / "labels").glob("*.txt")} if (SRC / "labels").exists() else set()
    stems = {n.rsplit(".", 1)[0] for n in names}

    problems = []
    if not imgs and not labs:
        problems.append(f"{SRC.relative_to(ROOT)}/images, labels 폴더가 없거나 비어 있습니다 (README '데이터 준비' 참고)")
    if names - imgs:
        problems.append(f"없는 이미지 {len(names - imgs)}장 (예: {sorted(names - imgs)[:2]})")
    if stems - labs:
        problems.append(f"없는 라벨 {len(stems - labs)}개 (예: {sorted(stems - labs)[:2]})")
    if imgs - names:
        problems.append(f"manifest에 없는 이미지 {len(imgs - names)}장")

    n_box = empty = 0
    for s in stems & labs:
        lines = [l for l in (SRC / "labels" / f"{s}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        n_box += len(lines)
        empty += not lines
    if stems <= labs and n_box != EXPECT_BOX:
        problems.append(f"bbox 합계 {n_box} (기대값 {EXPECT_BOX}). 다른 라벨 버전일 수 있습니다")
    if empty:
        problems.append(f"빈 라벨 {empty}개")

    print(f"manifest {len(names)}장 | 이미지 {len(imgs)}장 | 라벨 {len(labs)}개 | bbox {n_box}")
    if problems:
        print("문제 발견:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("OK: 데이터 배치가 올바릅니다.")


if __name__ == "__main__":
    main()
