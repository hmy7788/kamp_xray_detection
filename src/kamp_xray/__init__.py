"""KAMP X-ray 완제품 이물질 탐지 (팀 엽총창) 공통 패키지.

- common.py   : 경로, 데이터 버전 확인, 매니페스트·정답 읽기, 공통 예측 기록 형식
- metrics.py  : 모든 모델이 함께 쓰는 채점 (IoU/중심거리 매칭, AP50, mAP50-95, P/R/F1, 임계값 선택)
- dfine.py    : 최종 모델 D-FINE-N 학습·추론
- yolov3_tiny.py : 베이스라인 YOLOv3-tiny 학습·추론 (third_party/yolov3 사용)
"""
