# 보고서 표·그림 ↔ 근거 파일

결과 보고서의 수치가 어느 파일에서 나왔는지 정리한다. 경로는 저장소 루트 기준이다.

## 제1장 데이터 이해 및 진단

| 보고서 | 내용 | 근거 |
|---|---|---|
| 1.1절, 표 1-1 | 영상 2,532장, 상자 4,494개, 호기별 영상 수·해상도·상자 크기 | `data/manifest.csv`, `data/conditions.csv`, `data/README.md`, `python scripts/check_data.py` |
| 1.1절 | 대비 중앙값 22.8 (test) | `data/conditions.csv`의 `local_contrast` |
| 1.2절 | 공식 라벨 500장 / 팀 라벨 2,032장, 고정 크기 상자 | `data/manifest.csv`의 `source`, `preprocessing/labeling/` |
| 1.2절 | 라벨 중심 오차 (공식 0.99px, 팀 0.90px) | `results/analysis/label_offset_analysis.md`, `experiments/comparison_models/rtdetr/label_offset.py` |
| 1.3절, 그림 1-4 | 색상 상자 제거, 원본 BMP 재적용 시 2,532장 픽셀 일치 | `preprocessing/preprocess.py`, `scripts/verify_preprocess.py`, `results/analysis/10_preprocess_check_v1/summary.json` |
| 1.3절 | 묶음 단위 분할 547묶음, train/val/test 1,767/369/396 | `data/split_info.json`, `preprocessing/make_splits_v2.py` |

## 제2장 AI 예측모델 개발 및 성능평가

| 보고서 | 내용 | 근거 |
|---|---|---|
| 표 2-1 | 평가 지표 정의 | `src/kamp_xray/metrics.py`, `docs/metrics.md` |
| 표 2-2 | 평가 데이터 구성 | `results/dfine_n/eval_report_test.json`의 `by_group` |
| 표 2-3 | 실행 환경 | `requirements.txt`, `results/<모델>/eval_report_test.json`의 `env` |
| 표 2-4 | 모델별 학습 설정 | `configs/`, `results/<모델>/original_run/config.json`, `results/comparison_models/*/config.json` |
| 표 2-5 | 모델별 test 성능 | 최종·베이스라인: `results/dfine_n/eval_report_test.json`, `results/yolov3_tiny/eval_report_test.json` / 비교 모델: `results/comparison_models/six_models/summary.json` / 표: `results/model_comparison.md` |
| 2.3절 | YOLO26 입력 640 vs 1024 | `results/comparison_models/yolo26n/` |
| 그림 2-1 | 정확도와 속도 | `results/model_comparison.md`, `results/<모델>/speed.json`, `results/comparison_models/six_models/fps_all.json` |
| 2.4절 | 최종 모델 선정 근거 (합성 이물질 AP, 속도, NMS 없음) | `results/comparison_models/six_models/summary.json`, `results/analysis/11_synth_insert_6models_v1/` |

## 제3장 영향요인 및 오류분석

| 보고서 | 내용 | 근거 |
|---|---|---|
| 3.1절 | 호기·해상도·라벨 출처별 F1 | `results/<모델>/eval_report_test.json`의 `by_group`, `results/analysis/12_group_stats_6models_v1/` |
| 3.2절, 그림 3-1 | 미검출 사례, 중심 거리 기준 재채점 | `results/<모델>/eval_report_test.json`의 `center_distance`, `results/dfine_n/original_run/eval_center_sensitivity.json` |
| 3.2절 | 임계값 영향 | `results/<모델>/eval_report_val.json`의 `threshold`, `results/comparison_models/six_models/final_model_dfine.json` |
| 3.3절, 그림 3-2, 3-3 | 합성 이물질 대비별·위치별 검출률 | `scripts/synth_eval.py`, `results/analysis/08_synth_eval_v1/`, `results/analysis/11_synth_insert_6models_v1/` |

## 제4장 현장 활용방안

| 보고서 | 내용 | 근거 |
|---|---|---|
| 4.1절 | 이미지 단위 판정, 임계값 0.7114, 결함 영상 최고 신뢰도 ≥ 0.748, 빈 라벨 ≤ 0.04 | `results/dfine_n/eval_report_test.json`의 `image_level`, `results/dfine_n/test_image_decisions.csv` |
| 그림 4-2 | 재검사 하한에 따른 검출률·오검출률 | `results/comparison_models/six_models/final_model_dfine.json` |
| 4.2절, 그림 4-3 | D-FINE-N 처리 시간 (GPU 32ms, CPU 4스레드 97ms, 1스레드 172ms) | `results/dfine_n/original_run/speed_cpu_pinned.json`, `speed_gpu_repeat.json`, `scripts/speed.py` |

## 제5장 창의성 및 차별성

| 보고서 | 내용 | 근거 |
|---|---|---|
| 5.1절 | 합성 이물질로 검출 한계 측정, 이물질 제거 시 검출 소멸 | `results/analysis/`, `results/comparison_models/rtdetr_l/eval_erase_*.json` |
| 5.2절, 그림 5-3 | RT-DETR-l 대비 경량화 효과 | `results/comparison_models/rtdetr_l/speed_cpu_pinned.json`, `results/dfine_n/original_run/speed_cpu_pinned.json` |

## 제6장 코드 구성 및 재현성

| 보고서 | 내용 | 근거 |
|---|---|---|
| 6.1절, 그림 6-1 | 저장소 구조 | `README.md` "폴더 구조" |
| 6.2절, 그림 6-2 | 실행 순서 | `run_all.ps1`, `run_all.sh`, `README.md` "단계별 실행" |
| 6.3절 | 가중치와 결과물 | `weights/README.md`, `results/` |
