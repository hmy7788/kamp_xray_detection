# results

보고서 수치의 근거가 되는 제출 결과임. 재현 실행 결과는 outputs 폴더에 별도로 생성됨

| 위치 | 내용 |
|---|---|
| model_comparison.md | 6개 모델의 평가 데이터 성능 비교표 |
| dfine_n | 최종 모델 D-FINE-N 결과 |
| yolov3_tiny | 베이스라인 YOLOv3-tiny 결과 |
| comparison_models | 비교 모델 4종의 학습 및 평가 기록, 6개 모델 공통 채점 결과, 속도 측정 기록 |
| analysis | 오류 분석, 합성 이물질 실험, 이물질 제거 영상 실험, 호기별 성능, 전처리 검증 결과 |

## 모델별 결과 파일

| 파일 | 내용 |
|---|---|
| test_predictions.csv | 평가 데이터 예측 상자, 영상 이름과 좌표와 신뢰도 |
| test_image_decisions.csv | 평가 데이터 영상별 최고 신뢰도와 불량 정상 판정 |
| eval_report_val.json | 검증 데이터 채점 결과와 신뢰도 임계값 |
| eval_report_test.json | 평가 데이터 채점 결과, 호기별, 해상도별, 월별, 라벨 출처별 성능 포함 |
| preds_val.json, preds_test.json | 신뢰도 0.001 이상 전체 예측과 정답 상자 |
| speed.json | 처리 속도 측정 결과 |
| original_run | 제출 가중치를 만든 학습의 설정, 로그, 에폭별 성능 |

- 좌표는 원본 영상의 픽셀 단위임
- 기록 파일의 개인 PC 경로는 제거함
