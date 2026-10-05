# src/yolov3/weights

원본 yolov3 코드가 가중치를 저장하는 기본 위치입니다. 이 프로젝트에서는 **실험 결과 폴더(`runs/<이름>/<실험>/weights/`)에 저장**하므로 보통 비어 있어야 합니다. 환경변수 `YOLO_SAVE_DIR` 없이 학습하면 여기에 `last.pt`가 저장되어 덮어써지니 주의하세요.

- `download_yolov3_weights.sh`: 원본에 있던 다운로드 스크립트. 링크가 낡아 쓰지 않습니다. 사전학습 가중치는 [../../../weights/README.md](../../../weights/README.md)의 방법으로 받습니다.
- `*.pt` 파일은 Git에 올리지 않습니다.
