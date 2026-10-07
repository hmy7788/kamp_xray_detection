# 데이터 계약

이 문서가 데이터 쪽 약속의 유일한 정의다. 바꾸려면 PR로 바꾸고 다른 한 명이 승인한다.

## 1. 확정 데이터 (data/)

데이터는 2026-10-05 에 확정했다. 사진은 더 가공하지 않는다. 이 사진이 이 프로젝트의 **원본**이다.

```
data/
├─ train/images/<image_id>.png, train/labels/<image_id>.txt   1,767장  학습
├─ val/images,   val/labels                                    369장  검증 (평소 실험의 채점)
├─ test/images,  test/labels                                   396장  최종 테스트 (최종 후보만 한 번 채점)
├─ manifest.csv, manifest.sha256   사진 목록 (고정, 잠금). split 열이 어느 폴더에 있는지의 기준
├─ split_info.json                 분할 규칙·시드 기록
├─ README.txt                      분할별·장비별·출처별 개수 요약 (data_summary.py 가 만든다)
├─ conditions.csv                  조건 축 값 (compute_conditions.py 가 만든다)
└─ PASS                            check_data.py 통과 표시 (git 제외)
```

- 사진은 git 에 넣지 않는다. `kamp_data.zip` (train/val/test 사진·라벨 + manifest + README.txt) 을 드라이브로 받아 저장소 폴더에 풀면 `data/` 아래에 들어간다.
- 라벨 txt 와 manifest 는 git 에 넣는다. 세 사람의 데이터가 같은지는 `data/manifest.sha256` 과 `python common/check_data.py` 로 확인한다.
- 사진·라벨 경로는 `common/kx.py` 의 `image_path / label_path / path_maps` 로만 얻는다. 경로를 직접 조립하지 않는다.
- 사진은 회색조 PNG. 장비가 그려 넣던 색상 박스는 이미 지워져 있다 (어떻게 지웠는지: `archive/legacy_v1_fill/README.md`).
- 공식 라벨(500)과 팀 라벨(2,032)은 폴더로 구분하지 않는다. manifest 의 `source` 열(official / team)과 `labeler` 열로만 구분한다.

## 2. 사진 목록 (data/manifest.csv)

고정 파일이다. 다시 만들지 않는다. 한 줄이 사진 한 장, 2,532줄 전부 유효.

| 열 | 뜻 |
|---|---|
| image_id | 파일명에서 확장자를 뺀 것. 유일 |
| split | train / val / test. 사진이 있는 폴더 |
| machine | 1, 2, 3 (촬영 장비) |
| width, height | 픽셀 크기 |
| ts | 촬영 시각 (파일명에서 읽음) |
| month | 촬영 월 (6, 7, 8, 9) |
| burst_id | 같은 장비에서 60초 이내로 이어진 사진 묶음. 분할의 최소 단위 |
| suffix_n | 파일명 괄호 숫자 (뜻 모름, 보관만) |
| source | official (공식 라벨) / team (팀 라벨) |
| labeler | 팀 라벨을 찍은 사람 (lee / jung / heo). official 은 빈칸 |
| image_path, label_path | data/ 기준 상대 경로 (<split>/images/..., <split>/labels/...) |

## 3. 라벨

- 형식: YOLO txt. 한 줄에 상자 하나, `0 cx cy w h` (0~1 정규화). 클래스는 0 하나 (defect = 금속구).
- 공식 라벨 500개는 KAMP 가 준 것 그대로.
- 팀 라벨 2,032개는 라벨링 서버(Label Studio)에서 금속구를 클릭해 찍었다 (2026-10-05 완료). 클릭 중심에 공식 라벨 중앙값 크기(한 변 10px × 사진 높이/332 → 1·2호기 10px, 3호기 13px)의 네모가 자동으로 붙었다.
- 빈 라벨(상자 0개) 105장: 제품이 반만 찍혀 테스트피스가 없는 사진(7/27 1호기 104장) 등. "이물질 없음" 사진으로 쓸 수 있다.
- 라벨은 손으로 만들거나 고치지 않는다 (후크로 차단). 고칠 게 있으면 서버에서 고치고 다시 내보낸 뒤 PR 한다. 절차는 `docs/labeling.md`.

## 4. 분할 (data/train, data/val, data/test)

- 2026-10-05 에 한 번 나누고 폴더로 고정했다 (`archive/legacy_v1_fill/make_splits_v2.py`, seed 42). 수정 금지 (후크로 차단).
- 대상: 2,532장 전부 (공식 + 팀 라벨 구분 없음). 학습 / 검증 / 테스트 한 번 나누기 (폴드 없음).
- 묶음(burst) 단위라 같은 묶음은 한쪽에만 들어간다. test 15% (396장), val 15% (369장), train 나머지 (1,767장).
- test: 최종 보고 때 `reproduce.py --final` 로만 채점한다. 평소 실험 코드는 `kx.load_split()` 이 test 목록을 None 으로 돌려주므로 쓸 수 없다.
- 비율·방식을 바꾸고 싶으면 결정 002 에 적고 PR 로 데이터를 통째로 다시 배포한다 (manifest 해시가 바뀐다).

## 5. 검사 (data/PASS)

`python common/check_data.py` 가 manifest 잠금, 사진 2,532장 존재·크기, 라벨 2,532개 존재·형식, manifest 에 없는 파일 유무를 검사한다. 통과하면 `data/PASS` 가 생기고, `run.py` 는 PASS 가 있어야 돈다.

## 6. 조건 (common/conditions.yaml, data/conditions.csv)

- 조건 축의 정의는 `common/conditions.yaml` 한 곳에만 있다. 축을 추가하면 채점표에 칸이 자동으로 늘어난다.
- `python common/compute_conditions.py` 가 라벨 상자마다 축 값을 계산해 `data/conditions.csv` 에 쓴다.
- 경계값(크기 3분위 등)은 결정 006 에서 확정한다. 그 전까지는 yaml 의 기본값을 쓴다.
