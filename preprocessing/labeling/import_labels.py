"""Label Studio 에서 내보낸 YOLO zip 을 팀 라벨 폴더에 넣고, 진행률·일치율을 본다.

가져오기:  python data/labeling/import_labels.py import --member kim --zip <내보낸 zip>
진행 확인: python data/labeling/import_labels.py check

저장 위치: data/labels_team/<member>/<image_id>.txt   (공식 라벨과 형식 같음: "0 cx cy w h", 0~1 정규화)
           data/labels_team/<member>/index.csv        (image_id, n_boxes, imported_at)
공식 라벨(data/versions/*/labels)과는 절대 섞지 않는다. 채점에는 쓰지 않는다 (docs/labeling.md).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEAM = ROOT / "data" / "labels_team"
PREFIX = re.compile(r"^(?:[0-9a-f]{8}-)?(?:\d{4}__)?")  # Label Studio 해시 접두사, 내보내기 순번 접두사


def parse_lines(text: str) -> list[tuple[float, float, float, float]] | None:
    out = []
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        if len(p) != 5:
            return None
        try:
            vals = [float(x) for x in p[1:]]
        except ValueError:
            return None
        if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
            return None
        out.append(tuple(vals))
    return out


def do_import(member: str, zip_path: Path) -> None:
    dst = TEAM / member
    dst.mkdir(parents=True, exist_ok=True)
    assigned = {r["image_id"] for r in csv.DictReader((HERE / "assignments.csv").open(encoding="utf-8"))
                if r["assignee"] in (member, "all")}
    n_ok = n_bad = n_unknown = 0
    index: dict[str, int] = {}
    if (dst / "index.csv").exists():
        for r in csv.DictReader((dst / "index.csv").open(encoding="utf-8")):
            index[r["image_id"]] = int(r["n_boxes"])
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".txt") or Path(name).name == "classes.txt":
                continue
            stem = PREFIX.sub("", Path(name).stem)
            if stem not in assigned:
                n_unknown += 1
                continue
            boxes = parse_lines(z.read(name).decode("utf-8", "replace"))
            if boxes is None:
                n_bad += 1
                print(f"  형식 오류: {name}")
                continue
            (dst / f"{stem}.txt").write_text("".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes),
                                             encoding="utf-8")
            index[stem] = len(boxes)
            n_ok += 1
    with (dst / "index.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_id", "n_boxes", "imported_at"])
        for iid in sorted(index):
            wr.writerow([iid, index[iid], now])
    print(f"{member}: 가져옴 {n_ok}, 형식 오류 {n_bad}, 배정 밖(무시) {n_unknown} → {dst}")
    print(f"누적 {len(index)}/{len(assigned)}장. 다음: git add data/labels_team/{member} && git commit")


def _match(a: list, b: list) -> int:
    """a 의 상자 중심이 b 의 상자 안에 들어가는 개수 (1:1)."""
    used = set()
    n = 0
    for cx, cy, _, _ in a:
        for j, (bx, by, bw, bh) in enumerate(b):
            if j in used:
                continue
            if abs(cx - bx) <= bw / 2 and abs(cy - by) <= bh / 2:
                used.add(j)
                n += 1
                break
    return n


def do_check() -> None:
    rows = list(csv.DictReader((HERE / "assignments.csv").open(encoding="utf-8")))
    members = sorted({r["assignee"] for r in rows} - {"all"})
    overlap_ids = [r["image_id"] for r in rows if r["overlap"] == "1"]
    labels: dict[str, dict[str, list]] = {}
    print("진행률")
    for m in members:
        need = [r["image_id"] for r in rows if r["assignee"] in (m, "all")]
        d = TEAM / m
        got = {p.stem: parse_lines(p.read_text(encoding="utf-8")) or [] for p in d.glob("*.txt")} if d.exists() else {}
        labels[m] = got
        done = sum(1 for i in need if i in got)
        nb = sum(len(v) for v in got.values())
        print(f"  {m:5s}: {done:4d}/{len(need)} 장 ({100 * done // max(1, len(need)):3d}%), 상자 {nb}, 9월 제외 학습 가능 "
              f"{sum(1 for r in rows if r['assignee'] in (m, 'all') and r['train_ok'] == '1' and r['image_id'] in got)}")
    print("\n겹침 사진 일치율 (중심이 상대 상자 안에 들어간 비율, 사진 둘 다 라벨링한 것만)")
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            both = [x for x in overlap_ids if x in labels[a] and x in labels[b]]
            if not both:
                print(f"  {a}-{b}: 아직 없음")
                continue
            hit = sum(_match(labels[a][x], labels[b][x]) for x in both)
            na = sum(len(labels[a][x]) for x in both)
            nbb = sum(len(labels[b][x]) for x in both)
            same_n = sum(1 for x in both if len(labels[a][x]) == len(labels[b][x]))
            print(f"  {a}-{b}: {len(both)}장, 개수 일치 {same_n}/{len(both)}, "
                  f"상자 일치 {hit}/{na} ({a} 기준) {hit}/{nbb} ({b} 기준)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import")
    p.add_argument("--member", required=True)
    p.add_argument("--zip", required=True)
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "import":
        zp = Path(a.zip)
        if not zp.exists():
            sys.exit(f"zip 이 없다: {zp}")
        do_import(a.member, zp)
    else:
        do_check()


if __name__ == "__main__":
    main()
