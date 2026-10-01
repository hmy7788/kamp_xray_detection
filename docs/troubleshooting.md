# 트러블슈팅 기록

2020년 버전 ultralytics/yolov3 코드(`dataset/test1/yolov3`)를 최신 환경에서 실행하며 겪은 문제와 해결 내용입니다.

## 실행 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4050 Laptop (6GB) |
| conda 환경 | `KAMP` (Python 3.10) |
| PyTorch | 2.6.0+cu124 / torchvision 0.21.0 |
| numpy / OpenCV | 2.2.6 / 5.0.0 |

원본 `requirements.txt`는 `torch>=1.6` 수준이라, 설치되는 최신 라이브러리와 충돌하는 부분이 생겼습니다.

## 1. `torch.load` 안전 로드 오류

- **증상**: `WeightsUnpickler error: Unsupported global: numpy.core.multiarray._reconstruct`
- **원인**: PyTorch 2.6부터 `torch.load`의 기본값이 `weights_only=True`로 바뀜. 체크포인트에 numpy 객체가 들어 있어 로드 실패.
- **해결**: 모든 `torch.load(...)`에 `weights_only=False` 추가
  - `train.py`, `test.py`, `detect.py`, `models.py`, `utils/utils.py`
- **주의**: 신뢰할 수 있는 체크포인트에만 사용할 것.

## 2. `np.int` 제거

- **증상**: `AttributeError: module 'numpy' has no attribute 'int'`
- **원인**: numpy 1.24+ 에서 `np.int` 별칭 제거
- **해결**: `.astype(np.int)` → `.astype(int)` (`utils/utils.py`, `utils/datasets.py`)

## 3. `custom.data` 파싱 오류 (`too many values to unpack`)

- **증상**: `parse_config.py`의 `key, val = line.split('=')`에서 `ValueError`
- **원인**: 노트북이 `custom.data`와 `train.txt`/`test.txt`를 쓸 때 **줄바꿈(`\n`)이 빠져** 여러 줄이 한 줄로 합쳐짐. (노트북 생성 스크립트에서 `\n`이 실제 개행으로 변환되어 소스에서 사라진 것)
- **해결**: 노트북 소스에 이스케이프된 `\n`이 들어가도록 수정 후 재생성.

## 4. 한글 경로 인코딩 (cp949)

- **증상**: 하위 프로세스가 `custom.data`를 읽다가 `UnicodeDecodeError: 'cp949' codec...`
- **원인**: 프로젝트 경로에 한글이 있고, `open()` 기본 인코딩이 Windows에서는 cp949
- **해결**: 노트북 설정 셀에서 `os.environ["PYTHONUTF8"] = "1"` 설정. 터미널에서 직접 실행할 때도 `PYTHONUTF8=1` 지정.

## 5. 학습이 0 epoch만 돌고 `UnboundLocalError: epoch`

- **증상**: `Starting training for 807 epochs...` 직후 `UnboundLocalError: local variable 'epoch' referenced before assignment`
- **원인**: `last.pt`가 이미 806 epoch 학습본이라 `start_epoch = 807`. 코드는 `epochs += ckpt['epoch']`(=806)로 계산해 `range(807, 807)`이 되어 **반복이 한 번도 실행되지 않음**. 원본 노트북 명령(`--epochs 15`)도 실제로는 학습이 안 되는 상태였음.
- **해결**: `train.py`의 `epochs += ckpt['epoch']` → `epochs += start_epoch`. 이제 `--epochs N`이 "추가로 N epoch"를 의미.

## 6. CPU/GPU 텐서 혼용 (`build_targets`)

- **증상**: `RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)`
- **원인**: `utils/utils.py` `build_targets`에서 `torch.arange(na)`가 CPU에 생성되어 GPU 텐서를 인덱싱
- **해결**: `torch.arange(na, device=targets.device)`

## 7. 검증 단계 CUDA 텐서 → numpy 변환 오류

- **증상**: `TypeError: can't convert cuda:0 device type tensor to numpy`
- **위치**: `utils/utils.py` `output_to_target` → `test.py`의 `plot_images`
- **원인**: 리스트에 GPU 텐서 스칼라를 그대로 담아 `np.array()`로 변환
- **해결**: `targets.append([i, cls, float(x), float(y), float(w), float(h), float(conf)])`

## 8. 한글 경로 이미지 읽기 실패 (`Image Not Found`)

- **증상**: `AssertionError: Image Not Found C:\...\KAMP\data\samples\images\....jpg` (파일은 존재함)
- **원인**: Windows에서 `cv2.imread`가 한글 등 비ASCII 경로를 읽지 못함. 기존에는 `images/xxx.jpg` 같은 ASCII 상대경로만 써서 드러나지 않았고, 프로젝트를 표준 구조로 옮기며 절대경로(한글 포함)를 쓰면서 발생.
- **해결**: `utils/datasets.py`에 `imread`(`np.fromfile` + `cv2.imdecode`)/`imwrite`(`imencode` + `tofile`) 헬퍼를 추가하고 `LoadImages`, `load_image`, `detect.py` 저장부에서 사용.

## 9. 노트북이 쓴 `train.txt`가 cp949로 저장됨

- **증상**: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc7` (train/test 목록 로드 시)
- **원인**: 노트북 커널은 실행 중에 `PYTHONUTF8=1`을 설정해도 자신의 기본 인코딩은 그대로(cp949)라, 인코딩 없이 `open(..., "w")`로 쓴 파일이 cp949로 저장됨. 하위 프로세스(UTF-8 모드)는 이를 못 읽음.
- **해결**: 노트북에서 `open(..., "w", encoding="utf-8")`로 명시.

## 10. 사전학습 가중치 다운로드 경로

- **증상**: 저장소의 `download_yolov3_weights.sh`와 `attempt_download`가 가리키는 Google Drive/옛 릴리스 링크는 신뢰할 수 없음. `releases/download/v9.0/yolov3-spp-ultralytics.pt`는 404.
- **해결**: `https://github.com/ultralytics/yolov3/releases/download/v8/` 아래 파일이 200 응답. `yolov3-spp-ultralytics.pt`(252MB), `yolov3-tiny.pt`(35MB)를 `weights/pretrained/`에 받음. 두 파일 모두 `torch.load` 후 `epoch=-1`, 키 `epoch/best_fitness/training_results/model/optimizer` 확인.

## 11. `yolov3-tiny.cfg` 1클래스 변환

- 공식 cfg는 COCO 80클래스 기준. `filters=255 → 18`(=(1+5)×3), `classes=80 → 1`로 변환(`sed`). 사전학습 가중치의 출력층은 크기가 달라 `train.py`가 `numel` 불일치 키를 자동으로 건너뜀.

## 12. 공용 `last.pt` 덮어쓰기 방지 (`YOLO_SAVE_DIR`)

- 원본 `train.py`는 `weights/last.pt`, `results.txt`를 현재 폴더에 고정 저장. `--nosave`여도 마지막 epoch는 저장되어 덮어씀(806 epoch 원본 소실 사례).
- **해결**: `train.py`에 환경변수 `YOLO_SAVE_DIR`를 도입해 `wdir`, `results_file` 위치를 실험 폴더로 지정. `scripts/run_experiment.py`가 `outputs/runs/<실험명>/`으로 지정하고, 폴더가 이미 있으면 실행을 거부.

## 13. 지표 파싱 실패 (지수 표기)

- **증상**: `metrics.json`의 test 결과가 `null`.
- **원인**: `test.py` 출력에 `2.81e-05` 같은 지수 표기가 있는데 정규식이 `[\d.]+`만 허용.
- **해결**: `[\d.eE+-]+`로 수정.

## 14. GPU 메모리 (RTX 4050 6GB)

- YOLOv3-SPP는 batch 4(학습 해상도 최대 약 640)에서 약 6.0GB를 사용해 한계에 가까움. batch 8은 OOM 가능성이 높아 4로 설정. tiny는 batch 8에서 약 3.6GB.

## 15. `best.pt`는 마지막 epoch에서 갱신되지 않음

- `train.py`는 `(best_fitness == fi) and not final_epoch`일 때만 `best.pt`를 저장. 마지막 epoch가 최고여도 `last.pt`에만 반영됨. 평가 스크립트는 `best.pt`가 있으면 그것을, 없으면 `last.pt`를 사용. (fitness = 0.1·mAP + 0.9·F1, 검증셋 기준)

## 16. 검증 시간이 학습보다 오래 걸림

- 학습 초기(학습 안 된 모델)에는 conf 임계값이 0.001이라 NMS 후보가 많아 검증이 epoch당 약 30초(학습 약 10초). 학습이 진행되면 줄어듦.

## 17. 학습이 epoch당 약 1분 걸림 (GPU 사용률 0%)

- **관찰**: YOLOv3-tiny는 학습 40 iteration이 3~4초인데 epoch당 약 1분(100 epoch ≈ 1시간 40분), SPP는 학습 약 20초 + 검증 약 13초인데 epoch당 약 52초. 학습 중 `nvidia-smi` GPU 사용률이 계속 0%.
- **원인(추정, 아직 측정으로 확정하지 못함)**: `train.py`가 `num_workers = min(CPU, batch_size, 8)`로 DataLoader 워커를 만드는데, Windows는 프로세스를 fork하지 않고 새로 띄워(spawn) 매 epoch마다 워커 프로세스(train용 + 검증용)가 torch를 다시 import한다. GPU는 계산이 아니라 이 워커 기동 대기에 놀고 있다.
- **조치**: 두 DataLoader에 `persistent_workers=True`(워커가 0개가 아닐 때) 추가. 실행 중인 실험에는 반영되지 않았고, 다음 실험부터 적용된다. 효과는 아직 측정하지 않았다.

## 18. 학습이 오류 없이 19/100 epoch에서 끝남 (원인 미확인)

- **관찰**: `01_yolov3spp_coco`가 epoch 18에서 예고 없이 종료. 트레이스백과 "epochs completed" 메시지가 없고 프로세스 종료 코드가 0이라 `run_experiment.py`는 정상 완료로 보고 평가까지 진행해 `metrics.json`(val F1 0.10, test F1 0.097)을 기록했다. 이 수치는 19 epoch짜리 미완성 모델의 것이라 **사용하지 않는다**.
- **조치**: `run_experiment.py`가 `results.txt`의 epoch 수가 `--epochs`보다 적으면 불완전으로 종료하도록 검사 추가. 문제 폴더는 `outputs/runs/_incomplete_01_yolov3spp_coco_19of100`으로 이름을 바꿔 보관.
- **미확인**: 종료 원인. GPU 메모리가 6.1GB로 한계에 가까운 SPP(batch 4)에서만 발생했는지, 다시 재현되는지는 재실행으로 확인해야 한다.

## 19. 학습이 중간에 멈춘 채 프로세스만 남음 (절전 추정)

- **관찰**: `02_yolov3tiny_coco`가 2026-09-30 17:49에 epoch 46에서 진행을 멈췄다. 다음 날 08:51에 확인하니 `run_experiment.py`/`train.py` 프로세스는 살아 있지만 로그·`results.txt`가 15시간째 갱신되지 않았고 GPU 사용률 0%, 메모리 거의 0이었다. 같은 시점에 백그라운드 작업이 "2시간 시간 제한으로 종료"로 보고됨(벽시계 기준으로 제한에 걸린 것으로 보임).
- **원인(추정, 미확인)**: PC 절전/대기 모드 진입. 절전에서 깨어난 뒤 CUDA 컨텍스트와 DataLoader 워커가 복구되지 않은 것으로 보인다. SPP 19/100 종료(`troubleshooting.md` 18)가 같은 원인인지는 모른다.
- **조치**: 멈춘 프로세스를 수동 종료. 폴더 이름 변경은 "다른 프로세스가 사용 중" 오류(WinError 32)로 실패해 `outputs/runs/02_yolov3tiny_coco`로 남아 있음(원인 미확인, 재부팅 후 재시도). **긴 학습 전에 전원 설정에서 절전/최대 절전을 끄고(전원 연결 시), 덮개를 닫지 않는다.** `run_experiment.py`의 epoch 수 검사는 조기 종료를 감지하지만 "멈춤"은 감지하지 못하므로 로그 갱신 시간을 확인한다.

## 20. 참고: 남아 있는 경고 (동작에는 영향 없음)

- `torch.cuda.*DtypeTensor constructors are no longer recommended` (`utils/utils.py:355`)
- `Conversion of an array with ndim > 0 to a scalar is deprecated` (`test.py:228`)
- `WARNING: smart bias initialization failure.` (`models.py`, 사전학습 가중치 사용 시 무시 가능)
- `Apex recommended ...` (Apex 미설치 안내)

## 운영상 주의

- **`weights/last.pt` 덮어쓰기**: `--nosave`는 "마지막 epoch만 저장"이라는 뜻이므로 `last.pt`가 **덮어써짐**. 호환성 검증 중 원본 806 epoch 가중치가 2 epoch 추가 학습본으로 바뀌었고 백업이 없음. 학습 전에 `last.pt`를 복사해 두는 것을 권장.
- **노트북 실행 부산물**: `test1/yolov3/X선이물검출기/`(원본 bmp 복사본, 약 2,800장), `train.txt`/`test.txt`(무작위 재분할), `custom.data`(로컬 절대경로)가 새로 생성/변경됨.
- **확장자가 깨진 파일**(`.jpgnb`, `.jpgignore` 등)은 원래 `.ipynb`, `.dockerignore` 등이 치환된 것으로, 코드가 읽지 않음.
