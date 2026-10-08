"""Label Studio 에 사람별 프로젝트 3개를 만들고 사진 폴더를 연결한다 (업로드 없음, 폴더 연결).

서버가 떠 있는 상태에서:  venv\Scripts\python setup_projects.py
토큰은 계정.txt 의 TOKEN 줄을 읽는다. 이미 있는 프로젝트는 건너뛴다.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
HOST = "http://localhost:8080"
IMAGES = Path(r"<SHARE>\kamp_data\images")
MEMBERS = {"lee": "이총", "jung": "정연창", "heo": "허민엽"}
LABEL_CONFIG = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" brightnessControl="true" contrastControl="true"/>
  <RectangleLabels name="label" toName="image" strokeWidth="1">
    <Label value="foreign" background="#ff3b3b" hotkey="1"/>
  </RectangleLabels>
</View>"""


def token() -> str:
    txt = (HERE / "계정.txt").read_text(encoding="utf-8")
    m = re.search(r"TOKEN\s*=\s*(\S+)", txt)
    if not m:
        sys.exit("계정.txt 에 TOKEN 줄이 없다")
    return m.group(1)


def main() -> None:
    s = requests.Session()
    s.headers["Authorization"] = f"Token {token()}"
    for _ in range(60):
        try:
            if s.get(f"{HOST}/api/projects/").status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)
    else:
        sys.exit("서버가 안 떠 있다. start_server.bat 먼저.")

    existing = {p["title"]: p for p in s.get(f"{HOST}/api/projects/", params={"page_size": 100}).json()["results"]}
    for key, name in MEMBERS.items():
        title = f"{key} ({name})"
        if title in existing:
            print(f"있음: {title} (건너뜀)")
            continue
        r = s.post(f"{HOST}/api/projects/", json={
            "title": title,
            "description": f"{name} 이 그릴 사진 {len(list((IMAGES / 'team' / key).glob('*.png')))}장. 안내: README 참고",
            "label_config": LABEL_CONFIG,
            "show_instruction": True,
            "expert_instruction": "금속구(지름 몇 픽셀 검은 점)에 딱 맞게 작은 네모. 1~3개가 세로로. 확대 필수. 키 1 → 드래그 → Ctrl+Enter.",
        })
        r.raise_for_status()
        pid = r.json()["id"]
        st = s.post(f"{HOST}/api/storages/localfiles", json={
            "project": pid,
            "title": f"images/team/{key}",
            "path": str(IMAGES / "team" / key),
            "regex_filter": r".*\.png$",
            "use_blob_urls": True,
        })
        st.raise_for_status()
        sid = st.json()["id"]
        sync = s.post(f"{HOST}/api/storages/localfiles/{sid}/sync")
        sync.raise_for_status()
        n = s.get(f"{HOST}/api/projects/{pid}/").json().get("task_number")
        print(f"만듦: {title} → 사진 {n}장 연결")
    print("\n완료. 브라우저에서 http://localhost:8080 확인.")


if __name__ == "__main__":
    main()
