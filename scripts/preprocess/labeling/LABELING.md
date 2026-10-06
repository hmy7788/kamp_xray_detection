# 팀 라벨링 (라벨 없는 2,032장)

세 명이 공용 라벨링 서버(Label Studio)에서 금속구를 클릭해 찍는다. 결과는 서버 한 곳에 저장되고,
끝나면 `data/labels/team/` 에 txt 로 내보낸다. 팀원은 아무것도 설치하지 않는다.

## 서버 (lee 의 노트북에서 돌린다)

- 위치: `<LABELSTUDIO>\` (저장소 밖. venv, 데이터베이스, 보조 프로그램)
- 켜기: `start_server.bat` (Label Studio + 보조 프로그램 `fixed_box_backend.py`) → 별도 PowerShell 창에서 터널
  `cloudflared.exe tunnel --url http://localhost:8080` → 나온 주소를 팀원에게 보낸다 (주소는 켤 때마다 바뀜).
- 사진은 업로드가 아니라 `data/images/` 폴더를 연결한 것이다. 라벨은 `labelstudio/data/label_studio.sqlite3` 에 쌓인다.
- 계정·초대 링크: `labelstudio/계정.txt`.

## 찍는 법 (팀원, 안내 팝업과 같음)

1. 처음 한 번: 화면 아래 `Auto-Annotation` 켜기 → `Auto-Accept Suggestions` 켜기.
2. 오른쪽 도구 막대 맨 아래 분홍 아이콘(단축키 `M`) 한 번 클릭 → 점 도구.
3. `0 예시 (보기만)` 프로젝트에서 금속구 모양과 정답 네모 크기를 본다.
4. 자기 이름 프로젝트에서 `Label All Tasks` → 금속구 한가운데 클릭 → 1초 뒤 빨간 네모 자동 생성 (크기 자동).
5. 잘못 찍으면 `Ctrl+Z` 또는 네모 클릭 후 `Backspace`. 다 찍으면 `Ctrl+Enter` (저장 + 다음 사진).
6. 못 찾겠으면 안 찍고 `Ctrl+Enter`. 중간에 꺼도 저장돼 있다.

금속구는 테스트피스에 1~3개가 세로로 박혀 있고 지름 몇 픽셀의 검은 점이라 확대가 필수다.
흐릿한 얼룩(지운 자리)이 아니라 점 자체를 찍는다. 연속 사진은 점이 몇 픽셀씩 움직이니 매 장 새로 찍는다.

## 네모 크기

클릭 좌표를 보조 프로그램이 받아 한 변 `10px × (사진 높이 / 332)` 네모로 바꾼다 (1·2호기 10px, 3호기 13px).
공식 라벨 한 변 중앙값과 같다. 사람마다 크기가 달라지는 문제가 없다.

## 완료 (2026-10-05)

2,032장 전부 찍었다 (lee 680 / jung 678 / heo 674). Skip(건너뜀)은 빈 라벨로 가져왔다. 내보낸 txt 는 분할 폴더(`data/<split>/labels/`)로 옮겨져 고정됐다.
라벨을 고쳐야 하면: 서버에서 고치고 `labelstudio/export_labels.py` 로 내보낸 뒤, 해당 사진의 split 폴더에 넣고 manifest 해시를 다시 잠그는 PR 로 처리한다 (데이터 통째로 재배포).
