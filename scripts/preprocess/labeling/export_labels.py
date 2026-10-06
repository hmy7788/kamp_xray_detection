"""세 프로젝트의 라벨을 YOLO txt 로 내보내 kamp_data/labels/team/ 한 폴더에 넣는다. 진행률도 보여준다.

서버가 떠 있는 상태에서:  venv/Scripts/python export_labels.py

라벨링은 "클릭 = 점" 방식(KeyPoint)이다. 점을 중심으로 고정 크기 네모를 만든다.
네모 한 변 = 10px × (사진 높이 / 332)  → 1·2호기(높이 332) 10px, 3호기(높이 444) 13px. 공식 라벨 중앙값과 같다.
(네모로 그린 라벨이 섞여 있으면 그건 그대로 쓴다.)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
HOST = "http://localhost:8080"
DATA = Path(r"<KAMP-harness>\data")
BOX_PX_AT_332 = 10.0
PREFIX = re.compile(r"^[0-9a-f]{8}-")


def token() -> str:
    m = re.search(r"TOKEN\s*=\s*(\S+)", (HERE / "계정.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else sys.exit("TOKEN 없음")


def image_id_of(task: dict) -> str:
    url = task["data"]["image"]
    name = url.split("d=")[-1].replace("%5C", "/").replace("\\", "/").replace("%2F", "/").split("/")[-1]
    name = name.replace("%28", "(").replace("%29", ")")
    return PREFIX.sub("", Path(name).stem)


def boxes_of(annotation: dict) -> list[tuple[float, float, float, float]]:
    """네모(rectanglelabels)는 그대로, 점(keypointlabels)은 고정 크기 네모로. 점 위에 이미 네모가 있으면 점은 건너뛴다."""
    rects, points = [], []
    for r in annotation.get("result", []):
        v = r.get("value", {})
        W, H = r.get("original_width"), r.get("original_height")
        if not W or not H:
            continue
        if r.get("type") in ("keypointlabels", "keypoint"):
            side_w = BOX_PX_AT_332 * (H / 332.0) / W
            side_h = BOX_PX_AT_332 * (H / 332.0) / H
            points.append((v["x"] / 100, v["y"] / 100, side_w, side_h))
        elif r.get("type") == "rectanglelabels":
            rects.append(((v["x"] + v["width"] / 2) / 100, (v["y"] + v["height"] / 2) / 100, v["width"] / 100, v["height"] / 100))
    out = list(rects)
    for cx, cy, w, h in points:
        if not any(abs(cx - rx) <= rw / 2 and abs(cy - ry) <= rh / 2 for rx, ry, rw, rh in rects):
            out.append((cx, cy, w, h))
    return out


def main() -> None:
    s = requests.Session()
    s.headers["Authorization"] = f"Token {token()}"
    out = DATA / "labels" / "team"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for p in s.get(f"{HOST}/api/projects/", params={"page_size": 100}).json()["results"]:
        key = p["title"].split()[0]
        if key not in ("lee", "jung", "heo"):
            continue
        r = s.get(f"{HOST}/api/projects/{p['id']}/export", params={"exportType": "JSON", "download_all_tasks": "false"})
        if r.status_code != 200:
            print(f"{key}: 내보내기 실패 {r.status_code} {r.text[:100]}")
            continue
        n = n_pts = n_draft = 0
        done_ids = set()
        for task in r.json():
            anns = task.get("annotations", [])
            if not anns:
                continue
            ann = max(anns, key=lambda a: a.get("updated_at", ""))      # 같은 사진에 여러 개면 가장 최근 것
            # Skip(건너뜀) 은 "찍을 것이 없음" 으로 본다 → 빈 라벨
            boxes = [] if ann.get("was_cancelled") else boxes_of(ann)
            iid = image_id_of(task)
            (out / f"{iid}.txt").write_text(
                "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes), encoding="utf-8")
            done_ids.add(task["id"])
            n += 1
            n_pts += len(boxes)
        # Ctrl+Enter 를 안 누른 사진: 임시 저장본(초안)이라도 가져온다
        page = 1
        while True:
            t = s.get(f"{HOST}/api/tasks", params={"project": p["id"], "page": page, "page_size": 200, "fields": "all"})
            if t.status_code != 200:
                break
            tasks = t.json().get("tasks") or t.json().get("results") or []
            if not tasks:
                break
            for task in tasks:
                if task["id"] in done_ids:
                    continue
                drafts = [d for d in task.get("drafts", []) if d.get("result")]
                if not drafts:
                    continue
                d = max(drafts, key=lambda x: x.get("updated_at", ""))
                boxes = boxes_of(d)
                iid = image_id_of(task)
                (out / f"{iid}.txt").write_text(
                    "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes), encoding="utf-8")
                n_draft += 1
                n_pts += len(boxes)
            page += 1
        n_img = len(list((DATA / "images" / "team" / key).glob("*.png")))
        print(f"{key}: 확정 {n} + 초안 {n_draft} = {n + n_draft}/{n_img}장, 점 {n_pts}개")
        total += n
    print(f"→ {out}  (txt {len(list(out.glob('*.txt')))}개)")


if __name__ == "__main__":
    main()
