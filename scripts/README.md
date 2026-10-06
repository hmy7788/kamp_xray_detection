# scripts — 공유 파이프라인

데이터 전처리, 분할, 학습 실행, 평가처럼 **팀이 같이 쓰는 실행 코드**를 두는 폴더입니다. 공유 영역이라 바꾸려면 PR과 팀 확인이 필요합니다. (같은 분할, 같은 평가로 채점해야 모델 비교표가 의미가 있기 때문입니다.)

## 현재 상태
- `preprocess/` — **확정 데이터 v2 를 만든 전처리·라벨링·분할 코드와 설명** (2026-10-06 추가). 단계별 설명은 [preprocess/README.md](preprocess/README.md). 기록용이며 `data/` 를 다시 만들지 않습니다.
- 학습 실행·평가 코드는 아직 없습니다 (2026-10-05에 전부 새로 짜기로 해서 지웠습니다).

## 담당
- **데이터 전처리, 분할**: 이총(`chong`) — `preprocess/` 에 있습니다. v2 데이터 계보: KAMP 원본 BMP → 중복 제거 → 장비 색상 박스 제거(주변 회색 메움) → 회색조 PNG → 팀 라벨링 2,032장 → 묶음 단위 분할.
- **학습 실행, 평가**: 팀 합의 후 담당 지정 _(미정)_

## 이전 구현 참고
지운 코드는 Git 이력에 남아 있습니다. 마지막으로 존재한 커밋은 `71813fd`입니다.
```bash
git show 71813fd:scripts/<파일명>.py          # 내용 보기
git checkout 71813fd -- scripts/<파일명>.py   # 복구
```
이전에 있던 파일: `build_manifest`, `check_data`, `check_ring_cue`, `evaluate`, `make_split`, `reeval_experiment`, `remove_marks`, `run_experiment`, `watch_experiment`. (`prepare_n500.py`는 커밋하지 않아 Git에 없습니다.)

## 새로 만들 때 지킬 것
- 한글 경로: 이미지는 `cv2.imdecode(np.fromfile(...))`로 읽고, 설정 파일은 UTF-8로 엽니다. 실행 시 `PYTHONUTF8=1`.
- 신뢰도 임계값은 val에서 정해 고정하고 test는 한 번만 봅니다.
- 실험 결과는 `runs/<이름>/<실험>/`에, 이미 있으면 덮어쓰지 않고 중단합니다.
