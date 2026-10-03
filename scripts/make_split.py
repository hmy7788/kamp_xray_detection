"""n500 을 호기별로 층화하고 촬영 묶음(group) 단위로 train/val/test = 8:1:1 로 분할한다.

- 호기는 data/manifest.csv 의 unit(파일명 접두가 아니라 원본 폴더 기준 실제 호기)을 쓴다.
- 같은 호기에서 GAP_SEC 초 이내로 연속 촬영된 이미지는 한 묶음으로 보고 같은 분할에 넣어
  비슷한 이미지가 train 과 test 에 갈라지는 누수를 막는다.
- 호기마다 따로 8:1:1 로 나눠 각 분할에 세 호기가 모두 들어가게 한다.

출력 (data/splits/)
  split.csv            이미지별 분할 결과 (상대경로, 재현/검토용)
  train.txt val.txt test.txt   YOLO 이미지 목록 (이 PC 의 절대경로, 실행 시 재생성)
  val.data test.data   train.py / test.py 용 데이터 설정 (valid= 가 val / test)
실행: python scripts/make_split.py   (먼저 manifest.csv 필요: scripts/build_manifest.py)
표시 제거본: python scripts/make_split.py --src data/nomark --out data/splits_nomark  (분할 배정은 동일, 목록의 이미지 경로만 다름)
"""
import argparse
import csv
import random
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "subsets" / "n500"
OUT = ROOT / "data" / "splits"
GAP_SEC = 60
RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}
TRIES, SEED = 20000, 42


def load():
    rows = []
    for r in csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8")):
        m = re.match(r"(\d+)_(\d{8})_(\d{6})\(\d+\)", r["name"])
        t = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
        rows.append({"name": r["name"], "unit": r["unit"], "time": t, "n_box": int(r["n_box"])})
    rows.sort(key=lambda r: (r["unit"], r["time"], r["name"]))
    gid = 0
    for i, r in enumerate(rows):
        if i and (r["unit"] != rows[i - 1]["unit"] or (r["time"] - rows[i - 1]["time"]).total_seconds() > GAP_SEC):
            gid += 1
        r["group"] = gid
    return rows


def assign(groups, rng):
    """묶음을 무작위 순서로 섞어, 목표 이미지 수가 가장 모자란 분할에 탐욕적으로 채운다."""
    total = sum(len(g) for g in groups.values())
    target = {k: v * total for k, v in RATIO.items()}
    got = {k: 0 for k in RATIO}
    out = {}
    ids = list(groups)
    rng.shuffle(ids)
    for g in ids:
        k = max(RATIO, key=lambda s: target[s] - got[s])
        out[g] = k
        got[k] += len(groups[g])
    return out


def cost(groups, assignment):
    """한 호기 안에서 이미지 수/bbox 수 비율이 목표(8:1:1)에 가까울수록 작다. 빈 분할은 큰 벌점."""
    total_img = sum(len(g) for g in groups.values())
    total_box = sum(r["n_box"] for g in groups.values() for r in g)
    c = 0.0
    for s, ratio in RATIO.items():
        rs = [r for g, k in assignment.items() if k == s for r in groups[g]]
        if not rs:
            return 1e9
        c += abs(len(rs) / total_img - ratio) * 10 + abs(sum(r["n_box"] for r in rs) / total_box - ratio) * 20
    return c


def main():
    global SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="이미지/라벨 폴더 (기본 data/subsets/n500). 표시 제거본이면 data/nomark")
    ap.add_argument("--out", default=None, help="목록 출력 폴더 (기본 data/splits)")
    a = ap.parse_args()
    if a.src:
        SRC = ROOT / a.src
    if a.out:
        OUT = ROOT / a.out
    # manifest 의 이름(확장자 포함)과 SRC 안의 실제 파일명(확장자가 다를 수 있음: 제거본은 .png)을 stem 으로 연결
    files = {p.stem: p.name for p in (SRC / "images").iterdir()}
    rows = load()
    rng = random.Random(SEED)
    best_all = {}
    for unit in sorted({r["unit"] for r in rows}):
        groups = {}
        for r in rows:
            if r["unit"] == unit:
                groups.setdefault(r["group"], []).append(r)
        best = min((assign(groups, rng) for _ in range(TRIES)), key=lambda a: cost(groups, a))
        best_all.update(best)
        print(f"{unit}: 묶음 {len(groups)}개, 비용 {cost(groups, best):.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "group", "unit", "date", "time", "n_box"])
        for r in rows:
            w.writerow([r["name"], best_all[r["group"]], r["group"], r["unit"],
                        r["time"].strftime("%Y-%m-%d"), r["time"].strftime("%H:%M:%S"), r["n_box"]])

    for s in RATIO:
        names = [r["name"] for r in rows if best_all[r["group"]] == s]
        (OUT / f"{s}.txt").write_text("".join(f"{(SRC / 'images' / files[Path(n).stem]).as_posix()}\n" for n in names), encoding="utf-8")
    names_file = (ROOT / "data" / "classes.names").as_posix()
    for s in ("val", "test"):
        (OUT / f"{s}.data").write_text(
            f"classes= 1\ntrain={(OUT / 'train.txt').as_posix()}\nvalid={(OUT / f'{s}.txt').as_posix()}\nnames={names_file}\n",
            encoding="utf-8")

    units = sorted({r["unit"] for r in rows})
    print("분할별 이미지 수 / bbox 수 (호기별 이미지 수)")
    for s in RATIO:
        rs = [r for r in rows if best_all[r["group"]] == s]
        per = " / ".join(f"{u} {sum(r['unit'] == u for r in rs)}" for u in units)
        print(f"{s:5s} 이미지 {len(rs):3d}  bbox {sum(r['n_box'] for r in rs):4d}  ({per})")


if __name__ == "__main__":
    main()
