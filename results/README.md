# results/ — 제출 시점 결과 (보고서 수치의 근거)

이 폴더는 **읽기 전용**으로 취급한다. 재현 실행(`run_all`)은 `outputs/`에 결과를 만들고,
`python scripts/compare.py --check`가 두 폴더의 test 지표를 비교한다.

| 경로 | 내용 |
|---|---|
| `model_comparison.md`, `.csv` | 6개 모델 비교표 (보고서 표 2-5) |
| `dfine_n/` | **최종 모델 D-FINE-N** |
| `dfine_n/test_predictions.csv` | test 예측 결과: 임계값(0.7114) 이상 상자 (image_id, file_name, class, x1, y1, x2, y2, score; 원본 픽셀 좌표) |
| `dfine_n/test_image_decisions.csv` | test 이미지별 검출 수, 최고 신뢰도, 판정(불합격/합격), 정답 상자 수 |
| `dfine_n/preds_{val,test}.json` | 신뢰도 0.001 이상 전체 검출과 정답 (채점 입력, 형식은 `src/kamp_xray/common.py`) |
| `dfine_n/eval_report_{val,test}.json` | 임계값, 전체 지표, IoU별 AP, 중심 거리 지표, 이미지 단위 판정, 조건별 분해 |
| `dfine_n/speed.json` | 처리 속도 (`scripts/speed.py`) |
| `dfine_n/original_run/` | 제출 가중치를 만든 원 학습 실행 기록: `config.json`, `train.log`, `results.csv`(epoch별 val AP), 원 평가·속도 기록 |
| `yolov3_tiny/` | **베이스라인 YOLOv3-tiny**, 구성은 위와 같음 (`original_run/results.txt`가 epoch별 학습 기록) |
| `comparison_models/` | 비교 모델 4종의 원 평가 기록, `six_models/`는 6개 모델 통합 채점·속도 |
| `analysis/` | 오류분석·합성 이물질·이물질 제거·호기별 통계·전처리 검증 결과 |

`original_run/` 의 로그·설정에 있는 경로는 개인 PC 경로를 `<KAMP>`, `<HOME>` 등으로 바꿔 두었다.
