"""원본을 훑어 data/manifest.csv 를 만든다. 원본을 읽는 허용 스크립트다.

규칙 (docs/contracts/data.md):
- 장비는 폴더 이름(N호기)으로 판별한다. 파일명 접두사는 믿지 않는다.
- 시각은 파일명 NNN_YYYYMMDD_HHMMSS(N) 에서 읽는다.
- 내용 해시가 같은 파일은 대표 한 장만 유효(dup_of 빈칸).
- 라벨은 '라벨링 6종 세트/labels/*.txt'. 라벨 이미지(.jpg 이지만 실제 BMP)의 해시로 원본과 짝짓는다.
"""
from __future__ import annotations

import collections
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402

BURST_GAP_SEC = 60
NAME_RE = re.compile(r"^(\d+)_(\d{8})_(\d{6})\((\d+)\)$")


def find_machine_dirs(raw: Path) -> list[tuple[int, Path]]:
    out = []
    for p in raw.rglob("*"):
        if p.is_dir():
            m = re.match(r"^(\d)호기", p.name)
            if m:
                out.append((int(m.group(1)), p))
    return sorted(out)


def is_bmp(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"BM"
    except OSError:
        return False


def main() -> None:
    raw = kx.raw_root()
    machines = find_machine_dirs(raw)
    if not machines:
        sys.exit("N호기 폴더를 찾지 못했다. data/raw_path.txt 를 확인하라")

    # 라벨
    label_dir = next(raw.rglob("labels"), None)
    label_dirs = [p for p in raw.rglob("labels") if p.parent.name.startswith("라벨링")]
    if not label_dirs:
        sys.exit("라벨 폴더('라벨링 6종 세트/labels')를 찾지 못했다")
    label_dir = label_dirs[0]
    labels = {p.stem: p for p in label_dir.glob("*.txt")}
    # 라벨 이미지 해시 (stem 충돌 시 장비 판별용)
    label_img_sha: dict[str, str] = {}
    for d in label_dir.parent.glob("images*"):
        for p in d.iterdir():
            if p.stem in labels and p.stem not in label_img_sha and is_bmp(p):
                label_img_sha[p.stem] = kx.sha256_file(p)

    rows = []
    for machine, mdir in machines:
        for p in sorted(mdir.rglob("*")):
            if not p.is_file() or not is_bmp(p):
                continue
            m = NAME_RE.match(p.stem)
            if not m:
                print(f"[건너뜀] 파일명 형식이 다름: {p}")
                continue
            ts = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
            with Image.open(p) as im:
                w, h = im.size
            rows.append(
                dict(
                    stem=p.stem,
                    machine=machine,
                    src_path=str(p.relative_to(raw)).replace("\\", "/"),
                    sha256=kx.sha256_file(p),
                    width=w,
                    height=h,
                    ts=ts,
                    suffix_n=int(m.group(4)),
                )
            )
    df = pd.DataFrame(rows)
    print(f"BMP 파일 {len(df)}개, 장비별 {df.groupby('machine').size().to_dict()}")

    # 중복: 같은 해시 → 첫 행(장비, 경로 순)이 대표
    df = df.sort_values(["sha256", "machine", "src_path"]).reset_index(drop=True)
    rep = df.groupby("sha256")["stem"].transform("first")
    first_idx = ~df.duplicated("sha256", keep="first")
    df["dup_of"] = ""
    df.loc[~first_idx, "dup_of"] = rep[~first_idx]
    n_dup = int((~first_idx).sum())

    # image_id: stem. 대표끼리 stem 이 겹치면(1·2호기 접두사 충돌) 장비 접미사를 붙인다
    df["image_id"] = df["stem"]
    valid = df[df["dup_of"] == ""]
    clash = valid[valid.duplicated("stem", keep=False)]
    if len(clash):
        print(f"[주의] 서로 다른 내용인데 파일명이 같은 사진 {clash['stem'].nunique()}개 → image_id 에 __m<장비> 를 붙임")
        for i in clash.index:
            df.loc[i, "image_id"] = f"{df.loc[i, 'stem']}__m{df.loc[i, 'machine']}"
    # dup_of 는 대표의 image_id 로
    id_by_sha = df[df["dup_of"] == ""].set_index("sha256")["image_id"]
    df.loc[df["dup_of"] != "", "dup_of"] = df.loc[df["dup_of"] != "", "sha256"].map(id_by_sha)

    # 라벨 짝짓기: stem 이 같고 (해시가 있으면 해시도 같은) 대표 사진
    df["labeled"] = 0
    df["label_path"] = ""
    df["n_boxes"] = pd.NA
    unmatched = []
    for stem, lp in labels.items():
        cand = df[(df["stem"] == stem) & (df["dup_of"] == "")]
        if stem in label_img_sha:
            c2 = df[(df["sha256"] == label_img_sha[stem]) & (df["dup_of"] == "")]
            if len(c2):
                cand = c2
        if len(cand) == 0:
            unmatched.append(stem)
            continue
        i = cand.index[0]
        df.loc[i, "labeled"] = 1
        df.loc[i, "label_path"] = str(lp.relative_to(raw)).replace("\\", "/")
        df.loc[i, "n_boxes"] = len(kx.read_yolo_labels(lp))
    if unmatched:
        print(f"[주의] 원본과 짝이 안 맞는 라벨 {len(unmatched)}개: {unmatched[:5]} ...")

    # 묶음(burst): 같은 장비, 60초 이내 연쇄. 대표 사진만으로 계산
    df = df.sort_values(["machine", "ts", "src_path"]).reset_index(drop=True)
    df["burst_id"] = ""
    for machine, g in df[df["dup_of"] == ""].groupby("machine"):
        prev, b = None, 0
        for i, ts in zip(g.index, g["ts"]):
            if prev is None or (ts - prev).total_seconds() > BURST_GAP_SEC:
                b += 1
            df.loc[i, "burst_id"] = f"m{machine}_b{b:04d}"
            prev = ts
    df["month"] = df["ts"].dt.month
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cols = ["image_id", "machine", "src_path", "sha256", "width", "height", "ts", "month", "burst_id",
            "labeled", "label_path", "n_boxes", "suffix_n", "dup_of"]
    df = df[cols]
    kx.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(kx.MANIFEST, index=False, encoding="utf-8", lineterminator="\n")
    kx.MANIFEST_SHA.write_text(kx.sha256_file(kx.MANIFEST) + "  manifest.csv\n")

    v = df[df["dup_of"] == ""]
    print("---- manifest 요약 ----")
    print(f"유효 사진 {len(v)} (중복 제외 {n_dup}) / 라벨 {int(v['labeled'].sum())}")
    print("장비별 유효:", v.groupby("machine").size().to_dict())
    print("장비별 라벨:", v[v.labeled == 1].groupby("machine").size().to_dict())
    print("월별 라벨:", v[v.labeled == 1].groupby("month").size().to_dict())
    print("묶음 수:", v["burst_id"].nunique(), " 라벨 묶음 수:", v[v.labeled == 1]["burst_id"].nunique())
    print("해상도:", v.groupby(["machine", "width", "height"]).size().to_dict())
    print(f"저장: {kx.MANIFEST} / {kx.MANIFEST_SHA}")


if __name__ == "__main__":
    main()
