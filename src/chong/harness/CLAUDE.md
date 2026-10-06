# KAMP X-ray 이물질 탐지 (3인 팀: lee 이총 · jung 정연창 · heo 허민엽)

식품 완제품 X-ray 사진에서 이물질(금속구)을 찾고 위치를 표시한다. 평가는 리더보드가 아니라 보고서다:
데이터 진단 15 / 모델 개발 40 / 오류분석 15 / 현장 활용 10 / 창의성 10 / 재현성 10.
세 사람이 각자 GPU에서 학습하고 git 으로 합친다. 모델만 슬롯이고 데이터·채점·하네스는 공유다.

## 데이터는 확정됐다 (2026-10-05)
- `data/train` 1,767 / `data/val` 369 / `data/test` 396장. 각 폴더에 `images/`, `labels/`. 2,532장 전부 라벨 있음 (공식 500 + 팀 2,032).
- `data/manifest.csv` 가 고정 목록이다 (split, source, labeler, machine, 묶음). 이 사진과 분할이 **원본**이다. 더 가공하지 않는다. 계보는 `archive/legacy_v1_fill/README.md`.
- 분할은 묶음(burst) 단위 한 번 나누기, 폴드 없음. test 는 최종 후보만 `reproduce.py --final` 로 한 번 채점한다.

## 절대 규칙 (후크가 같은 규칙을 코드로 막는다)
1. `data/train`, `data/val`, `data/test`, `data/manifest.*` 는 수정하지 않는다. 사진·라벨 경로는 `common/kx.py` 로만 얻는다 (`kx.load_split()`, `kx.path_maps()`).
2. 평소 실험은 train 으로 학습하고 val 로만 채점한다. `data/test` 는 읽지 않는다.
3. 테스트(test) 채점은 최종 후보가 정해진 뒤 `python reproduce.py --final` 로 한 번만 한다 (KAMP_FINAL=1).
4. 실험 하나는 변경점 하나다. parent_exp 대비 바뀐 키가 하나여야 한다.
5. 실험은 `configs/<이니셜>/<exp_id>.yaml` 로만 정의하고 `python common/run.py <설정>` 으로만 실행한다. entry.py·train 을 직접 돌리지 않는다.
6. 기록(record.json)과 채점(metrics.json)이 없는 실험은 없는 실험이다.
7. `reports/experiments.csv` 는 `common/aggregate.py` 가 만든다. 손으로 고치지 않는다.
8. `common/`, `docs/contracts/`, `.claude/` 는 PR 로만 바꾼다 (사람이 KAMP_ALLOW_SHARED=1 을 켜고). 자기 폴더는 `members/<이니셜>/`, `configs/<이니셜>/` 뿐이다.

## 데이터 사실 (2026-10-05 기준)
- 2,532장, 단일 클래스(defect = 금속구). 장비 1호기 943 / 2호기 805 / 3호기 784. 월 6·7·8·9월. 상자 4,494개, 빈 라벨(이물질 없음) 105장.
- 해상도: 1호기 316/352/412×332 혼재, 2호기 316×332, 3호기 576×444. 2호기가 어둡고 대비가 크다.
- 같은 제품에 금속구 1~3개가 세로로 박힌 테스트피스. 정상(음성) 사진은 없다 → 사진 단위 오경보율은 못 잰다.
- 상자: 공식 라벨 한 변 중앙값 10px (3호기 13px). 팀 라벨은 클릭 중심에 같은 크기 네모로 자동 생성.
- 4초 간격 6장 묶음(burst)이 기본 단위, 이웃 사진 상관 0.9. 분할은 묶음 단위 (train 1,767 / val 369 / test 396).
- 색상 박스(장비가 그린 네모)는 이미 지워진 사진이다. 라벨 중심의 99.8% 가 그 안에 있었기 때문 (docs/history/eda_raw_2026-10-02.md).

## 작업 전에 읽을 문서
데이터 → `docs/contracts/data.md` / 모델·실험 → `docs/contracts/experiment.md` / 채점·분석 → `docs/contracts/evaluation.md` / 협업·git → `docs/contracts/collaboration.md` / 라벨링 → `docs/labeling.md`
결정은 `docs/decisions/`, 사람의 통찰은 `docs/insights.md`, 평가표 대응은 `docs/rubric_map.md`.

## 명령
`/setup` 시작 절차 · `/eda` 데이터 진단 · `/cycle` 실험 한 바퀴(심사 뒤 사람이 고름) · `/run <설정> "<이유>"` · `/analyze <exp>` · `/sync` 하루 동기화 · `/decide <번호>` · `/report-check`

## 환경
`KAMP_MEMBER=<이니셜>` 필수. 후크는 Python 이라 OS 무관. 한글 출력은 PYTHONUTF8=1 (settings.json 이 켠다).
