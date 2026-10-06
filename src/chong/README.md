# src/chong (이총)

이총의 모델 코드 폴더입니다. **이 폴더 안은 본인만 수정합니다.** (다른 팀원 폴더, `data/`, `scripts/`, `src/yolov3/`는 PR과 팀 확인이 필요합니다. [CONTRIBUTING.md](../../CONTRIBUTING.md))

## 담당 모델
- **YOLO26** (ultralytics 8.4, COCO 사전학습) — 입력 1024 (01), P2 헤드(4px 격자) 변형 (02)

## 구조: `harness/` 는 실험 하네스의 사본
실험은 별도 하네스 저장소(설정 파일 하나 = 실험 하나, `run.py` 가 검사 → 학습 → 채점 → 기록을 한 번에)에서 돌렸고,
그 코드를 그대로 복사해 두었습니다. 결과는 `runs/chong/<번호>_<모델>_<설정>/` 에 팀 형식으로 내보냈습니다.

```
harness/
├─ common/            실행기(run.py), 채점(evaluate.py, metrics/), 조건 축(conditions.yaml), 경로 도우미(kx.py)
│   └─ baselines/     하네스 공용 기준선 코드 (이번 실험에서는 쓰지 않음)
├─ members/lee/
│   ├─ entry.py                ultralytics 학습·예측 진입점 (arch 가 .yaml 이면 구조 + 사전학습 가중치 이식)
│   ├─ synth_normal_check.py   합성 정상 사진 검사 (아래)
│   └─ export_to_team.py       하네스 결과 → runs/chong/ 팀 형식 변환
├─ configs/lee/       실험 설정 (lee_001 = 01_yolo26n_img1024, lee_002 = 02_yolo26n-p2_img1024)
├─ archive/legacy_v1_fill/   확정 데이터 v2 를 만든 전처리 코드 (scripts/ 로 PR 예정)
├─ docs/contracts/    데이터·실험·채점 약속, docs/insights.md 관찰 기록
└─ reproduce.py
```

## 재현 (Windows, PYTHONUTF8=1)
하네스는 자기 루트 아래 `data/` 를 읽습니다. 저장소의 `data/` 를 연결한 뒤 실행합니다.
```powershell
$env:PYTHONUTF8 = "1"; $env:KAMP_MEMBER = "lee"
cmd /c mklink /J src\chong\harness\data data            # 확정 데이터 v2 연결 (복사 아님)
cd src\chong\harness
python common\check_data.py                             # manifest 해시·사진·라벨 검사 → data\PASS
python common\compute_conditions.py                     # 조건 축 (data\conditions.csv)
python common\run.py configs\lee\lee_001.yaml           # 학습 → val 예측 → 채점 → runs\lee_001\{metrics.json, preds_val.csv}
python members\lee\synth_normal_check.py runs\lee_001   # 합성 정상 사진 검사 → runs\lee_001\synth_normal.md
python members\lee\export_to_team.py lee_001 01_yolo26n_img1024 --team ..\..\..   # 팀 형식으로 내보내기
```
환경: Python 3.13, torch 2.11 (cu128), ultralytics 8.4.152. 팀 환경(torch 2.6, ultralytics 8.4.173)에서도 같은 API 입니다.
주의: ultralytics 8.4 는 `predict()` 에 경로 목록을 넘기면 **목록 전체를 한 묶음으로 GPU 에 올립니다.** 369장을 한 번에 넘기면 6GB GPU 에서 메모리가 터지므로 entry.py 는 8장씩 끊어 예측합니다.

## 합성 정상 사진 검사 (synth_normal_check.py)
이 데이터는 같은 시편(테스트피스)을 반복 촬영한 것이라 이물질이 **제품 기준 항상 같은 자리**(어두운 띠 왼쪽 끝 3곳)에 있습니다.
모델이 이물질을 보고 찾는지, 자리를 외운 것인지 가르기 위해 val 사진으로 세 묶음을 만들어 예측합니다.
- synth_normal: 라벨 자리를 주변 회색으로 메운 사진 (이물질 없음). 여기서 찾으면 오경보.
- control: 라벨은 두고 임의 자리를 같은 방식으로 메운 사진. 여기서 찾으면 "메운 흔적" 이 단서.
- original: 손대지 않은 사진.
결과는 `runs/chong/<실험>/synth_normal.md` 에 있습니다. 01 실험에서는 지운 자리를 76% 에서 여전히 찾았고(control 0%), 16px 더 넓게 메워도 62% 라 **자리 외우기가 확인**됐습니다.

## 채점 방식 (팀 공통 표와의 관계)
- `runs/chong/<실험>/report_val.json` 과 `docs/experiments.md` 의 줄은 팀 계산기(`src/minyeop/faster_rcnn/report.py`)로 만들었습니다 (IoU 0.5, val F1 최대 임계값, 부트스트랩 1,000회).
- `harness_metrics.json` 은 하네스 채점입니다. 같은 IoU 0.5 매칭에 더해 **중심 적중**(예측 중심이 정답 상자 안)과 **조건별 재현율**(호기·해상도·월·라벨 출처·크기·대비·배경·가장자리)이 들어 있습니다. 전체 수치는 팀 계산기와 같습니다 (01: AP 0.982, TP/FP/FN 599/7/7).

## 연결되는 다른 폴더
- 실험 결과: `runs/chong/<번호>_<모델>_<설정>/`
- 공통 분할·평가는 `scripts/`의 것을 그대로 씁니다 (같은 분할, 같은 지표로 비교해야 하므로).
