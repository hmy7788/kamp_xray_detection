# third_party/yolov3 — 수정본 YOLOv3 (베이스라인 YOLOv3-tiny의 모델 코드)

[ultralytics/yolov3](https://github.com/ultralytics/yolov3)의 **2020년 버전**을 가져와 이 프로젝트 환경(PyTorch 2.6, numpy 2, Windows, 한글 경로)에서 돌아가도록 수정한 사본입니다. 베이스라인 학습·추론은 `src/kamp_xray/yolov3_tiny.py`가 이 코드를 불러 실행합니다.

## 출처와 라이선스
- 원본: https://github.com/ultralytics/yolov3 (GPL-3.0)
- 원본 라이선스(GPL-3.0) 전문: https://github.com/ultralytics/yolov3/blob/master/LICENSE

## 원본에서 수정한 부분 (되돌리지 말 것)
- 모든 `torch.load(...)`에 `weights_only=False`
- `np.int` → `int`
- `train.py`: 이어서 학습할 때 `epochs += start_epoch`(원본은 반복이 0번이 되는 버그), 환경변수 `YOLO_SAVE_DIR`로 산출물 위치 지정, DataLoader `persistent_workers`
- `utils/utils.py`: `build_targets`의 디바이스 불일치, `output_to_target`의 `float()` 변환
- `utils/datasets.py`: 한글 경로용 `imread`/`imwrite` 헬퍼, `detect.py`도 이를 사용
- `utils/datasets.py`(2026-10-06): 이미지가 1,000장을 넘으면 만드는 라벨 캐시 저장(`np.save`)이 numpy 2에서 실패해서(길이가 다른 배열 목록), object 배열로 직접 만들어 저장하도록 수정. 캐시는 `data/<split>/labels.npy`로 생기며 `.gitignore`의 `*.npy`로 제외. 학습 동작에는 영향 없음
- `yolov3-tiny.cfg`를 1클래스(`filters=18`, `classes=1`)로 변환

제출용 정리에서 쓰지 않는 원본 파일(splitdata.py, evolve.sh, gcp.sh, weights/ 다운로드 스크립트)은 뺐습니다.
