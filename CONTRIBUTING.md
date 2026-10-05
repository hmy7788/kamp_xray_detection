# 협업 가이드

3명(허민엽, 정연창, 이총)이 각자 모델을 돌리고 결과를 합치는 방식입니다. 모두가 **같은 데이터 분할과 같은 평가**를 쓰기 때문에 결과를 비교할 수 있습니다.

## 폴더와 담당

| 폴더 | 누가 | 규칙 |
|---|---|---|
| `src/<이름>/`, `notebooks/<이름>/`, `runs/<이름>/` | 본인 | 자유롭게 작업. **다른 사람 폴더는 수정하지 않음** |
| `data/`, `scripts/`, `src/yolov3/`, `runs/baseline/` | 공유 | 별도 PR로 바꾸고 나머지 2명이 확인 |

- 내 폴더 이름: **허민엽 `minyeop`, 정연창 `yeonchang`, 이총 `chong`** (한글 경로는 이미지 읽기·인코딩 오류를 일으켜 영문만 씁니다)
- 공유 영역을 바꾸면(특히 분할, 평가 방식) 모든 사람의 결과가 비교 불가가 될 수 있습니다. 바꾸기 전에 팀에 알립니다.

## 브랜치
- `main`은 보호하고 **PR로만** 합칩니다. 항상 동작하는 상태를 유지합니다.
- 작업 브랜치: `feat/<이름>-<내용>` (예: `feat/chong-yolov8`), 공통 코드 수정은 `fix/<내용>`, 문서는 `docs/<내용>`.
- 자기 폴더만 바꾸면 파일이 겹치지 않아 병합 충돌이 거의 없습니다.

## 커밋
- 메시지: `종류: 한글 요약` (`feat`, `fix`, `docs`, `chore`, `data`). 변경 이유를 본문에 적습니다.
- 작업 단위로 쪼개서 커밋합니다.

## 실험 규칙
- 실험은 `scripts/run_experiment.py`로 실행하고 이름 앞에 내 폴더를 붙입니다.
  ```bash
  python scripts/run_experiment.py --name minyeop/05_yolov3tiny_img640 --dataset nomark --cfg yolov3-tiny.cfg       --weights weights/pretrained/yolov3-tiny.pt --epochs 100 --batch-size 8
  ```
- 평가는 `python scripts/evaluate.py minyeop/05_yolov3tiny_img640` (val에서 임계값을 정해 고정, test는 한 번만).
- 번호는 본인 폴더 안에서 이어 붙입니다. **이미 있는 이름이면 덮어쓰지 않고 중단**됩니다.
- 끝까지 돌지 못한 실험은 최종 결과로 쓰지 않고 폴더 이름을 `_incomplete_...`로 바꿉니다.
- 숫자는 `eval_report_*.json`에서만 가져옵니다. 손으로 옮겨 적지 않습니다.
- **표시(색 박스)를 지운 데이터(`--dataset nomark`)로만 학습·평가합니다.** 표시가 있는 이미지는 모델이 결함이 아니라 표시를 보고 맞힙니다 (`docs/dataset.md` 8절).

## 올리면 안 되는 것
- 가중치 `*.pt`, 이미지·라벨(KAMP 데이터), 정답 좌표가 든 검출 캐시(`preds_*.json`)
- 개인 PC 경로: 로그와 `config.json`의 로컬 경로는 `<KAMP>`로 치환하고 올리기 전에 확인
  ```bash
  git grep --untracked -nI "OneDrive\|Users[\/]" -- runs docs notebooks scripts src
  ```
- 소속·로고 등 식별 정보 (대회 규칙, 성명과 팀명만 허용)
- 노트북은 출력(output)을 지우고 커밋

## 환경
설치와 데이터 준비는 [README.md](README.md)를 따릅니다.
