# src/yolov3 — 수정본 YOLOv3 (공유 베이스라인)

[ultralytics/yolov3](https://github.com/ultralytics/yolov3)의 **2020년 버전**을 가져와 이 프로젝트 환경(PyTorch 2.6, numpy 2, Windows, 한글 경로)에서 돌아가도록 수정한 사본입니다. 팀 모두가 쓰는 공유 베이스라인이므로 바꾸려면 PR과 팀 확인이 필요합니다.

## 출처와 라이선스
- 원본: https://github.com/ultralytics/yolov3 (GPL-3.0)
- **원본의 LICENSE 파일은 아직 이 폴더에 없습니다.** 저장소를 공개하기 전에 원본의 LICENSE를 추가해야 합니다.

## 원본에서 수정한 부분 (되돌리지 말 것)
- 모든 `torch.load(...)`에 `weights_only=False`
- `np.int` → `int`
- `train.py`: 이어서 학습할 때 `epochs += start_epoch`(원본은 반복이 0번이 되는 버그), 환경변수 `YOLO_SAVE_DIR`로 산출물 위치 지정, DataLoader `persistent_workers`
- `utils/utils.py`: `build_targets`의 디바이스 불일치, `output_to_target`의 `float()` 변환
- `utils/datasets.py`: 한글 경로용 `imread`/`imwrite` 헬퍼, `detect.py`도 이를 사용
- `utils/datasets.py`(2026-10-06): 이미지가 1,000장을 넘으면 만드는 라벨 캐시 저장(`np.save`)이 numpy 2에서 실패해서(길이가 다른 배열 목록), object 배열로 직접 만들어 저장하도록 수정. 캐시는 `data/<split>/labels.npy`로 생기며 `.gitignore`의 `*.npy`로 제외. 학습 동작에는 영향 없음
- `yolov3-tiny.cfg`를 1클래스(`filters=18`, `classes=1`)로 변환

자세한 원인과 해결은 지운 문서에 있고 Git 이력에서 볼 수 있습니다: `git show d151bfa:docs/troubleshooting.md`
