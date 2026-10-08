"""1단계: 데이터 검사. 확정 데이터(v2)가 맞는지, 분할·라벨이 보고서와 같은지 확인한다.

    python scripts/check_data.py

검사 항목: manifest sha256 / split별 이미지 수(1,767·369·396)와 상자 수(3,225·606·663) /
이미지·라벨 파일 존재와 짝 / 라벨 형식(클래스 0, 0~1 좌표) / 이미지 크기가 manifest와 같은지(표본) / 회색조 여부(표본)
결과는 outputs/data_check.json 에 저장한다. 하나라도 어긋나면 종료 코드 1.
"""
import _bootstrap  # noqa: F401

import random
from collections import Counter

from kamp_xray.common import (BOX_COUNTS, DATA, MANIFEST_SHA256, OUTPUTS, SPLIT_SIZES, imread, manifest_sha256,
                              read_manifest, save_json)


def main():
    problems = []
    digest = manifest_sha256()
    if digest != MANIFEST_SHA256:
        problems.append(f"manifest sha256 불일치: {digest}")
    man = read_manifest()
    n_img, n_box, by_machine, by_source = Counter(), Counter(), Counter(), Counter()
    for i, r in man.items():
        s = r["split"]
        n_img[s] += 1
        by_machine[(s, r["machine"])] += 1
        by_source[(s, r["source"])] += 1
        img, lab = DATA / r["image_path"], DATA / r["label_path"]
        if not img.exists():
            problems.append(f"이미지 없음: {r['image_path']}")
        if not lab.exists():
            problems.append(f"라벨 없음: {r['label_path']}")
            continue
        if img.parent.name != "images" or lab.parent.name != "labels" or img.parent.parent != lab.parent.parent:
            problems.append(f"images/labels 형제 폴더 규칙 위반: {i}")
        for line in lab.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            v = line.split()
            if len(v) != 5 or v[0] != "0" or not all(0.0 <= float(x) <= 1.0 for x in v[1:]):
                problems.append(f"라벨 형식 오류: {r['label_path']}: {line}")
            n_box[s] += 1
    for s in SPLIT_SIZES:
        if n_img[s] != SPLIT_SIZES[s]:
            problems.append(f"{s} 이미지 수 {n_img[s]} (기대 {SPLIT_SIZES[s]})")
        if n_box[s] != BOX_COUNTS[s]:
            problems.append(f"{s} 상자 수 {n_box[s]} (기대 {BOX_COUNTS[s]})")
        extra = {p.stem for p in (DATA / s / "images").glob("*.png")} - {i for i, r in man.items() if r["split"] == s}
        if extra:
            problems.append(f"{s}/images 에 manifest에 없는 이미지 {len(extra)}장")

    random.seed(0)
    sample = random.sample(sorted(man), 200)
    n_color = 0
    for i in sample:
        r = man[i]
        im = imread(DATA / r["image_path"])
        if im.shape[0] != int(r["height"]) or im.shape[1] != int(r["width"]):
            problems.append(f"크기 불일치: {i} {im.shape[:2]} vs manifest {r['height']}x{r['width']}")
        if not ((im[..., 0] == im[..., 1]).all() and (im[..., 1] == im[..., 2]).all()):
            n_color += 1
    if n_color:
        problems.append(f"색 픽셀이 남은 이미지 {n_color}장 (표본 200장 중)")

    res = {"manifest_sha256": digest, "expected_sha256": MANIFEST_SHA256,
           "images": dict(n_img), "boxes": dict(n_box),
           "images_by_split_machine": {f"{s}/{m}호기": n for (s, m), n in sorted(by_machine.items())},
           "images_by_split_source": {f"{s}/{src}": n for (s, src), n in sorted(by_source.items())},
           "grayscale_sample": {"checked": len(sample), "with_color": n_color},
           "problems": problems, "ok": not problems}
    save_json(OUTPUTS / "data_check.json", res)
    print(f"manifest sha256 {digest} ({'일치' if digest == MANIFEST_SHA256 else '불일치'})")
    for s in SPLIT_SIZES:
        print(f"  {s:5}: 이미지 {n_img[s]:5}  상자 {n_box[s]:5}")
    if problems:
        print(f"문제 {len(problems)}건:")
        for p in problems[:30]:
            print("  -", p)
        raise SystemExit(1)
    print("데이터 검사 통과 →", OUTPUTS / "data_check.json")


if __name__ == "__main__":
    main()
