# 프로젝트 디렉터리 구조

3명이 각자 모델을 돌리고 결과를 합치는 협업을 위해 **공유 영역**과 **개인 영역**을 폴더로 나눴다 (2026-10-05 개편). 규칙은 [CONTRIBUTING.md](../CONTRIBUTING.md).

```
KAMP/
├─ README.md  CONTRIBUTING.md  CLAUDE.md  requirements.txt
├─ data/                         [공유·고정]
│  ├─ manifest.csv  classes.names
│  ├─ splits/  splits_nomark/    분할 (split.csv만 Git)
│  └─ archive/                   이전 분할 (splits_n400, splits_old_prefix)
├─ scripts/                      [공유] 데이터 준비·학습·평가 파이프라인
├─ src/
│  ├─ yolov3/                    [공유] 수정본 YOLOv3 베이스라인
│  └─ minyeop/  yeonchang/  chong/   [개인] 모델 코드
├─ notebooks/
│  ├─ shared/                    기존 노트북
│  └─ minyeop/  yeonchang/  chong/   [개인]
├─ runs/
│  ├─ baseline/                  [공유] 기준 베이스라인, _invalid/
│  └─ minyeop/  yeonchang/  chong/   [개인] <번호>_<모델>_<설정>/
├─ docs/                         문서
└─ weights/                      Git 제외
```

## 설계 원칙
- **같은 분할, 같은 평가**: `data/`와 `scripts/`(특히 `evaluate.py`)는 모두가 공유한다. 모델이 달라도 같은 기준으로 채점되어야 비교표가 의미가 있다.
- **충돌 방지**: 각자 `<이름>/` 폴더 안에서만 작업하면 파일이 겹치지 않는다. 노트북(JSON)도 사람별로 나눴다.
- **영문 폴더명**: 한글 경로는 이미지 읽기·인코딩 오류를 여러 번 일으켰다.

## 평가 항목과의 대응
| 평가 항목 | 위치 |
|---|---|
| 데이터 이해·진단 (15) | `docs/dataset.md`, `data/` |
| 모델 개발·비교 (40) | `src/`, `runs/` (`evaluate.py`가 만든 `eval_report_*.json`) |
| 영향요인·오류분석 (15) | `runs/*/preds_*.json`에서 FN/FP 집계 (미착수) |
| 현장 활용(10), 창의성(10) | 보고서 |
| 코드·재현성 (10) | `scripts/`, `requirements.txt`, `README.md` |

## 이력: 개편 전 구조와 이동 내용 (2026-10-05)
| 이전 | 이후 |
|---|---|
| `third_party/yolov3/` | `src/yolov3/` |
| `third_party/OpenLabeling/` | 삭제 (어디서도 쓰이지 않음. 필요하면 https://github.com/Cartucho/OpenLabeling) |
| `outputs/runs/01~04_*` | `runs/baseline/` |
| `outputs/runs/_aborted_*`, `_incomplete_*` | `runs/baseline/_invalid/` |
| `notebooks/*.ipynb` | `notebooks/shared/` |
| `data/splits_n400`, `data/splits_old_prefix` | `data/archive/` |
| `configs/`, `reports/`, `submission/`, `outputs/`, 빈 `src/*` | 삭제 (필요할 때 만듦) |

이동은 `git mv`로 이력을 유지했다. 원본 실습 폴더(`4. X-ray 검사장비 AI 데이터셋/`)는 데이터 출처로 두며 조사 결과는 `docs/original_folder_inventory.md`.
