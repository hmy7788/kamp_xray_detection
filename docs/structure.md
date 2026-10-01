# 프로젝트 디렉터리 구조

원본 실습 폴더(`4. X-ray 검사장비 AI 데이터셋/`)는 그대로 두고, 그 옆에 ML 프로젝트 표준 구조를 만들었습니다. 기존 파일은 아직 옮기지 않았습니다(아래 "이관 계획" 참고).

```
KAMP/
├─ README.md  CLAUDE.md
├─ environment.yml            # (예정) 제출용 환경 정의. requirements.txt 포함
├─ configs/                   # 실험 설정 (분할, 하이퍼파라미터, 모델별 cfg)
├─ data/                      # [있음] 정리된 데이터
│  ├─ classes.names
│  ├─ samples/  subsets/      #   실습 세트, n015~n400
│  └─ splits/                 #   train/val/test 분할 결과 (make_split.py 산출물)
├─ src/                       # 재사용 코드 (노트북에서 import)
│  ├─ data/                   #   분할, 데이터 점검, YOLO 목록/데이터파일 생성
│  ├─ models/                 #   학습·추론 래퍼 (모델별)
│  └─ evaluation/             #   P/R/F1, FN·FP 집계, 오류분석
├─ scripts/                   # 실행 진입점: make_split.py [있음], train.py, evaluate.py, run_pipeline.py
├─ notebooks/                 # EDA, 오류분석, 결과 시각화, 리팩토링 노트북
├─ third_party/               # 외부 코드 (호환성 수정본 yolov3, OpenLabeling)
├─ weights/                   # 사전학습·체크포인트 (git/제출 제외, 실험별 별도 저장)
├─ outputs/runs/<실험명>/     # 실험별 산출물: 가중치, 로그, 예측 결과, 지표 json
├─ reports/                   # 보고서용 그림(figures), 표(tables)
├─ submission/                # 제출물 패키지 (소스 ZIP, 예측결과 CSV, 발표자료)
└─ docs/                      # dataset / original_folder_inventory / troubleshooting / experiment_results / evaluation / strategy / structure
```

## 평가 항목과의 대응

| 평가 항목 | 담당 위치 |
|---|---|
| 데이터 이해·진단 (15) | `notebooks/` EDA, `docs/dataset.md`, `data/splits/` |
| 모델 개발·비교 (40) | `src/models`, `configs/`, `outputs/runs/` (모델별 실험 폴더), `reports/tables` |
| 영향요인·오류분석 (15) | `src/evaluation`, `notebooks/` 오류분석, `reports/figures` |
| 현장 활용방안 (10), 창의성 (10) | 보고서 (`reports/`, `submission/`) |
| 코드·재현성 (10) | `scripts/run_pipeline.py`, `environment.yml`, `submission/` |

## 이관 결과 (복사 완료)

| 현재 | 이동 후 | 비고 |
|---|---|---|
| `4. .../dataset/test1/yolov3/*.py`, `utils/`, `*.cfg` | `third_party/yolov3/` | 호환성 수정 유지. 이미지·라벨·`X선이물검출기/`·`result/`는 제외 |
| `4. .../dataset/OpenLabeling-master/` | `third_party/OpenLabeling/` | 데모 XML은 제외 |
| `4. .../dataset/yolov3_refactored.ipynb` | `notebooks/` | 경로 설정 수정 필요 |
| `4. .../dataset/실습별 가중치파일/`, `test1/yolov3/weights/last.pt` | `weights/legacy/` | 각 약 250MB, 이동 후 원본과 중복되지 않게 처리 |
| 원본 실습 폴더 | 원본 데이터 출처로 유지 | 노트북 실행 과정에서 일부 변경됨: `original_folder_inventory.md` 4절 |

이관은 모두 복사로 진행했고 원본은 삭제하지 않았습니다. 새 위치에서 `notebooks/yolov3_refactored.ipynb`를 끝까지 실행해 학습·추론·평가 동작을 확인했습니다. 가중치 `last.pt`는 `weights/legacy/last_806ep_plus2.pt`로 복사했습니다.

- 원본 삭제 여부는 아직 정하지 않았습니다(디스크 사용량이 큰 것은 `weights` 약 1.7GB와 `X선이물검출기/` 복사본).

## 규칙

- 실험 하나 = `outputs/runs/<날짜>_<모델>_<설정>/` 폴더 하나. 가중치를 공용 `last.pt`에 덮어쓰지 않는다.
- 분할·seed·하이퍼파라미터는 `configs/`에 두고 코드에 하드코딩하지 않는다.
- 제출물에는 소속·로고 등 식별 정보를 넣지 않는다.
