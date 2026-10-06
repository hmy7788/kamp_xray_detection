"""라벨 없는 사진 2,032장을 멤버에게 나눈다 (묶음 단위, 장비×월 비율 유지).

python data/labeling/make_assignments.py [--members lee,jung,heo] [--overlap-bursts 9]

출력: data/labeling/assignments.csv
  image_id, burst_id, machine, month, assignee, overlap, train_ok
  - assignee : 누가 라벨링하는가. overlap=1 인 사진은 assignee 가 "all" (세 명 모두 한다)
  - overlap  : 세 명이 모두 라벨링해서 일치율을 재는 사진
  - train_ok : 1 이면 학습에 넣어도 된다. 9월 사진은 0 (holdout 이 9월이라 시간 분리를 지키기 위해)
규칙: 같은 묶음(burst)은 한 사람이 통째로 맡는다. 결정적(같은 입력 → 같은 출력).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "common"))
import kx  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="lee,jung,heo")
    ap.add_argument("--overlap-bursts", type=int, default=9, help="세 명이 모두 라벨링할 묶음 수 (장비별로 고르게)")
    a = ap.parse_args()
    members = [m.strip() for m in a.members.split(",") if m.strip()]

    df = kx.load_manifest(valid_only=True)
    df = df[df["labeled"].astype(int) == 0].copy()
    df["burst_id"] = df["burst_id"].astype(str)
    df["machine"] = df["machine"].astype(str)
    df["month"] = df["month"].astype(int)

    bursts: dict[str, list[dict]] = defaultdict(list)
    for r in df.sort_values("image_id").to_dict("records"):
        bursts[r["burst_id"]].append(r)

    # 1) 겹침 묶음: 장비별로 6장짜리 묶음을 고르게 뽑는다 (9월 제외, 묶음 id 순)
    overlap: set[str] = set()
    per_machine = max(1, a.overlap_bursts // 3)
    for m in sorted({r["machine"] for r in df.to_dict("records")}):
        cands = sorted(b for b, rows in bursts.items()
                       if rows[0]["machine"] == m and rows[0]["month"] != 9 and len(rows) == 6)
        step = max(1, len(cands) // per_machine) if cands else 1
        overlap.update(cands[::step][:per_machine])

    # 2) 나머지 묶음: 장비×월 층 안에서 사진 수가 가장 적은 멤버에게 준다
    load = {m: 0 for m in members}
    assignee: dict[str, str] = {}
    strata: dict[tuple, list[str]] = defaultdict(list)
    for b, rows in bursts.items():
        if b in overlap:
            continue
        strata[(rows[0]["machine"], rows[0]["month"])].append(b)
    for key in sorted(strata):
        for b in sorted(strata[key], key=lambda x: (-len(bursts[x]), x)):
            m = min(members, key=lambda mm: (load[mm], members.index(mm)))
            assignee[b] = m
            load[m] += len(bursts[b])

    out = HERE / "assignments.csv"
    lines = ["image_id,burst_id,machine,month,assignee,overlap,train_ok"]
    for b in sorted(bursts):
        for r in bursts[b]:
            who = "all" if b in overlap else assignee[b]
            lines.append(f"{r['image_id']},{b},{r['machine']},{r['month']},{who},{int(b in overlap)},{int(r['month'] != 9)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_over = sum(len(bursts[b]) for b in overlap)
    print(f"라벨 없는 사진 {len(df)}장, 묶음 {len(bursts)}개 → {out}")
    print(f"겹침(세 명 모두): 묶음 {len(overlap)}개, {n_over}장")
    for m in members:
        rows = [r for b, rows in bursts.items() if assignee.get(b) == m for r in rows]
        by_m = defaultdict(int)
        for r in rows:
            by_m[r["machine"]] += 1
        n9 = sum(1 for r in rows if r["month"] == 9)
        print(f"  {m:5s}: {len(rows):4d}장 (+겹침 {n_over}) | 장비 " + " ".join(f"{k}:{v}" for k, v in sorted(by_m.items()))
              + f" | 9월(학습 제외) {n9}")


if __name__ == "__main__":
    main()
