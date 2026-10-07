"""전처리 재현 검증: 원본 BMP 에 'fill' 전처리를 다시 돌려 data/ 의 PNG 와 픽셀 단위로 비교한다.

  python src/minyeop/preprocess_check/verify_preprocess.py --out runs/minyeop/10_preprocess_check_v1 [--limit 80]

배경: data/ 를 만든 전처리 코드는 main 에 없다(data/README.md: 가공 코드는 이 저장소에 없음). 이총의 브랜치 origin/feat/chong-preprocess 의
scripts/preprocess/preprocess.py (v1_fill)에 있다. 이 스크립트의 fill 함수는 그 코드와 같다(복사). 방법:
  색 픽셀 = max(R,G,B) - min(R,G,B) > 0 (장비가 그린 색 박스, 팔레트 사진이라 회색은 R=G=B)
  색 픽셀을 주변 회색(색 아닌) 픽셀의 5x5 평균으로 메운다(정규화 합성곱, 안 채워지면 최대 20회 반복). 회색조 PNG 로 저장.
검증: data/manifest.csv 의 이미지마다 같은 이름의 원본 BMP 를 찾아(이름이 같은 후보가 여럿이면 크기가 같은 것 중 가장 가까운 것)
fill 결과와 data PNG 가 한 픽셀이라도 다른지 센다. 이름 끝의 __m1/__m2 는 1·2호기 양쪽에 같은 이름이 있어 붙인 구분자라 떼고 해당 호기 폴더에서 찾는다.
원본(4. X-ray 검사장비 AI 데이터셋/)은 읽기 전용으로 쓴다. 원본이 있는 PC 에서만 실행된다.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = ROOT / "4. X-ray 검사장비 AI 데이터셋" / "dataset" / "test1" / "yolov3"


def color_mask(rgb):
    return (rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int)) > 0


def to_gray(rgb):
    return rgb[:, :, 1].copy()


def fill_neighbors(gray, mask, k=5):
    """origin/feat/chong-preprocess:scripts/preprocess/preprocess.py 의 fill_neighbors 와 같다."""
    g = gray.astype(np.float32)
    valid = (~mask).astype(np.float32)
    out = g * valid
    todo = mask.copy()
    kernel = np.ones((k, k), np.float32)
    for _ in range(20):
        if not todo.any():
            break
        num = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(valid, -1, kernel, borderType=cv2.BORDER_REFLECT)
        ok = todo & (den > 0)
        out[ok] = num[ok] / den[ok]
        valid[ok] = 1.0
        todo &= ~ok
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="원본 BMP 가 있는 폴더(하위 폴더 포함)")
    ap.add_argument("--out", default="runs/minyeop/10_preprocess_check_v1")
    ap.add_argument("--limit", type=int, default=0, help="무작위 N장만 검사(동작 확인용)")
    a = ap.parse_args()
    raw = Path(a.raw)
    index = defaultdict(list)                       # 이름 -> [(경로, 호기 폴더 이름)]
    for dp, _dn, fn in os.walk(raw):
        for f in fn:
            if f.lower().endswith(".bmp"):
                index[f[:-4]].append(Path(dp) / f)
    print("원본 BMP", sum(len(v) for v in index.values()), "개, 고유 이름", len(index), flush=True)

    rows = list(csv.DictReader(open(ROOT / "data" / "manifest.csv", encoding="utf-8")))
    if a.limit:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), a.limit, replace=False))]
    res = {"검사한 이미지": len(rows), "완전 일치": 0, "일부 다름": 0, "원본 못 찾음": 0}
    by_machine = defaultdict(lambda: [0, 0])
    n_color, diffs, n_mask_in_label = [], [], 0
    for n, r in enumerate(rows):
        iid = r["image_id"]
        proc = np.array(Image.open(ROOT / "data" / r["image_path"]).convert("L"))
        if iid.endswith(("__m1", "__m2")):          # 1·2호기에 같은 이름이 있는 경우: 해당 호기 폴더에서 찾는다
            base, tag = iid.rsplit("__", 1)
            cands = [p for p in index.get(base, []) if f"{tag[1:]}호기" in str(p)]
        else:
            cands = index.get(iid, [])
        best = None
        for c in cands:
            rgb = np.asarray(Image.open(c).convert("RGB"))
            if rgb.shape[:2] != proc.shape:
                continue
            m = color_mask(rgb)
            d = np.abs(fill_neighbors(to_gray(rgb), m).astype(int) - proc.astype(int))
            key = int((d > 0).sum())
            if best is None or key < best[0]:
                best = (key, int(d.max()), int(m.sum()))
        if best is None:
            res["원본 못 찾음"] += 1
            continue
        by_machine[r["machine"]][0] += 1
        n_color.append(best[2])
        if best[0] == 0:
            res["완전 일치"] += 1
            by_machine[r["machine"]][1] += 1
        else:
            res["일부 다름"] += 1
            diffs.append({"image": iid, "다른 픽셀": best[0], "최대 차이": best[1]})
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(rows)} 완전 일치 {res['완전 일치']}", flush=True)
    res["호기별 [검사, 완전 일치]"] = {k: v for k, v in sorted(by_machine.items())}
    res["원본 한 장당 색 픽셀 수(중앙값/최소/최대)"] = [int(np.median(n_color)), int(min(n_color)), int(max(n_color))] if n_color else None
    res["일부 다른 이미지"] = diffs[:20]
    res["결론"] = ("원본 BMP 에 fill(색 픽셀 → 주변 5x5 회색 평균, 최대 20회 반복)을 적용하면 data/ 의 PNG 와 모든 이미지에서 픽셀 단위로 일치한다"
                 if res["일부 다름"] == 0 and res["원본 못 찾음"] == 0 else "일치하지 않는 이미지가 있다(위 목록 확인)")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("저장:", out / "summary.json")


if __name__ == "__main__":
    main()
