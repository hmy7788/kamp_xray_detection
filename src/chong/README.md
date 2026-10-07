# src/chong (이총)

이총의 모델 코드 폴더입니다. **이 폴더 안은 본인만 수정합니다.** (다른 팀원 폴더, `data/`, `scripts/`, `src/yolov3/`는 PR과 팀 확인이 필요합니다. [CONTRIBUTING.md](../../CONTRIBUTING.md))

## 담당 모델
- **YOLO26n** (ultralytics 8.4, COCO 사전학습) — 입력 1024 (01, test 채점 완료), 입력 640 (08, 현장 경량화 비교용)
- 경량화는 같은 가중치의 **추론 조건**을 바꾸는 방식입니다: 입력 1024 → 640, PyTorch → ONNX Runtime CPU (1·4스레드). INT8 은 다른 PC 실험(06)에서 느려져 채택하지 않았습니다.

## 구조
```
harness/          실험 하네스 사본 (설정 파일 하나 = 실험 하나, run.py 가 검사 → 학습 → 채점 → 기록)
├─ common/            실행기(run.py), 채점(evaluate.py, metrics/), 조건 축(conditions.yaml), 경로 도우미(kx.py)
├─ members/lee/
│   ├─ entry.py                ultralytics 학습·예측 진입점 (끊긴 학습 이어 가기, 8장씩 끊어 예측)
│   ├─ synth_normal_check.py   합성 정상 사진 검사 (이물질 자리를 옆 조각 복사로 덮음)
│   ├─ synth_band.py           띠 안 합성 이물질 생성(build) · 추론(evaluate) · 비교표(report)  → runs/chong/09
│   ├─ synth_band_gallery.py   09 결과 갤러리(HTML) 와 공유 zip
│   ├─ speed.py                한 장씩 넣었을 때 GPU·CPU 속도 측정
│   └─ export_to_team.py       하네스 결과 → runs/chong/ 팀 형식 변환
├─ configs/lee/       실험 설정 (lee_001 = 01, lee_002 = 02 P2 변형(미완), lee_003 = 08)
├─ archive/legacy_v1_fill/   확정 데이터 v2 를 만든 전처리 코드
├─ docs/contracts/    데이터·실험·채점 약속, docs/insights.md 관찰 기록
└─ reproduce.py       --final <exp> [--existing-weights]  test 채점 (재학습 또는 기존 가중치)
detectability/    다른 PC(2026-10-06)에서 만든 위치별 검출 진단·배포 조건 비교 코드 → runs/chong/03~07. [detectability/README.md](detectability/README.md)
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
python common\run.py configs\lee\lee_003.yaml           # 640 학습 (08)
python members\lee\synth_normal_check.py runs\lee_001   # 합성 정상 사진 검사 → runs\lee_001\synth_normal.md
python reproduce.py --final lee_001 --existing-weights  # test 1회 채점 (기존 가중치) → runs\lee_001\metrics_test.json
python members\lee\export_to_team.py lee_001 01_yolo26n_img1024 --team ..\..\..   # 팀 형식으로 내보내기
```
띠 안 합성 비교(09): 점 신호 템플릿은 `runs/chong/09_synth_band_lightweight/templates/*.npz` (05 에서 val 공식 라벨로 추출). `KAMP_TEMPLATE_SRC` 로 위치를 지정합니다.
```powershell
python members\lee\synth_band.py build                                  # 띠 찾기 → 자리 5,930곳 → samples.jsonl, qa.png
python members\lee\synth_band.py evaluate --tag lee001_gpu1024 --weights runs\lee_001\work\seed42\train\weights\best.pt --imgsz 1024
python members\lee\synth_band.py evaluate --tag lee001_gpu640  --weights <같은 가중치> --imgsz 640
python members\lee\synth_band.py evaluate --tag lee001_onnx640_cpu_t4 --weights <onnx 640> --imgsz 640 --device cpu --threads 4
python members\lee\synth_band.py report                                 # compare.md
python members\lee\synth_band_gallery.py --zip                          # gallery/index.html + share zip
```
환경: Python 3.13, torch 2.11 (cu128), ultralytics 8.4.152, onnxruntime 1.30. 팀 환경(torch 2.6, ultralytics 8.4.173)에서도 같은 API 입니다.
주의: ultralytics 8.4 는 `predict()` 에 경로 목록을 넘기면 **목록 전체를 한 묶음으로 GPU 에 올립니다.** 369장을 한 번에 넘기면 6GB GPU 에서 메모리가 터지므로 entry.py 는 8장씩 끊어 예측합니다.
ONNX 는 `nms=True, conf=0.001, iou=0.7, max_det=50` 으로 내보내 PyTorch 와 같은 검출 경로(NMS 포함)를 유지합니다. ultralytics 는 onnxruntime 세션에 스레드 수를 넘기지 않아 synth_band.py 가 세션 생성을 감싸서 지정합니다.

## 합성 정상 사진 검사 (synth_normal_check.py)
이 데이터는 같은 시편(테스트피스)을 반복 촬영한 것이라 이물질이 **제품 기준 항상 같은 자리**(어두운 띠 왼쪽 끝 3곳)에 있습니다.
모델이 이물질을 보고 찾는지, 자리를 외운 것인지 가르기 위해 val 사진으로 세 묶음을 만들어 예측합니다.
- synth_normal: 라벨 자리를 바로 옆 같은 크기 조각으로 덮은 사진 (이물질 없음). 여기서 찾으면 오경보.
- control: 라벨은 두고 임의 자리를 같은 방식으로 덮은 사진. 여기서 찾으면 "덮은 흔적" 이 단서.
- original: 손대지 않은 사진.
결과는 `runs/chong/01_yolo26n_img1024/synth_normal.md` 에 있습니다. **2026-10-06 정정:** 처음 결과(지운 자리 76% 검출, "자리 외우기")는 메우기 함수의 버그였습니다. 지운 픽셀의 어두운 값을 이웃으로 다시 읽어 흐릿한 점이 남았고 모델은 그 자국을 찾은 것입니다. 옆 조각 복사로 고친 뒤에는 **합성 정상 오경보 0.3%(1/350), 지운 자리 검출 0, 대조 0%** 입니다. 모델은 점 자체를 봅니다. 교훈: 합성 사진은 확대해 눈으로 확인한 뒤 결론을 냅니다.

## 띠 안 합성 이물질 + 경량화 비교 (synth_band.py, 09)
05 의 합성 762곳은 72% 가 밝은 제품 몸통에 있어 "데이터에 없는 조건" 이었습니다. 실제 이물질은 전부 어두운 띠 안에 있으므로, 09 는 띠를 찾아 그 안에만 넣습니다.
- 배경: val 33장 (호기별 11~12장, 공식 라벨 사진). 띠 = 제품 마스크 안의 어둡고 긴 성분. 자리는 띠를 따라 10px 간격, 기존 라벨에서 12px 이상 떨어진 곳, 둘레 밝기가 띠 중앙값보다 8 이상 밝으면 제외.
- 신호: 05 에서 추출한 잔차 템플릿(6~8px) × 진하기 1.0 / 0.7 / 0.5 / 0.35 / 0.25. 한 사진에 한 곳씩 넣고 예측.
- 검출 = 넣은 뒤 확신도가 임계값(0.42, 0.10) 이상이고 넣기 전에는 미만. 전 설정에서 넣기 전 검출 0.
- 설정 6개를 같은 5,930개로 비교: 01 GPU 1024 / 01 GPU 640 / 08 GPU 640 / 01 ONNX CPU 4·1스레드 / 08 ONNX CPU 4스레드. 결과는 [runs/chong/09_synth_band_lightweight/compare.md](../../runs/chong/09_synth_band_lightweight/compare.md), 그림은 `gallery/index.html`.
- 결론: 실제 진하기면 띠 안 어디든 100% 잡고(위치 무관), 30% 옅으면 88%, 반이면 48%. 3호기가 가장 취약. 640 은 3호기에서만 크게 손해. ONNX 는 수치 보존, CPU 4스레드 27 FPS.

## 채점 방식 (팀 공통 표와의 관계)
- `harness_metrics.json` / `harness_metrics_test.json` 은 하네스 채점입니다. IoU 0.5 매칭에 더해 **중심 적중**(예측 중심이 정답 상자 안)과 **조건별 재현율**(호기·해상도·월·라벨 출처·크기·대비·배경·가장자리), 임계값 훑기가 들어 있습니다.
- 01 의 val 전체 수치는 2026-10-06 팀 계산기 결과(`report_val.json`)와 같았습니다 (AP 0.982, TP/FP/FN 599/7/7). 팀 계산기 스크립트는 지금 main 에 없어 08 과 test 는 하네스 채점만 있습니다.
- test 는 `reproduce.py --final lee_001 --existing-weights` 로 2026-10-07 한 번 채점했습니다. 재학습하지 않고 val 분석에 쓴 가중치 그대로이며, 팀원 모델들과 같은 조건(train 학습 → test 채점)입니다.

## 연결되는 다른 폴더
- 실험 결과: `runs/chong/<번호>_<모델>_<설정>/`
- 공통 분할은 `data/manifest.csv` 의 split 열을 그대로 씁니다 (같은 분할로 비교해야 하므로).
