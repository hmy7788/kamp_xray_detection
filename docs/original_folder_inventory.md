# 원본 폴더 전수 조사

`4. X-ray 검사장비 AI 데이터셋/` 를 읽기 전용으로 조사한 결과입니다 (2026-09-30). 총 7,896개 파일, 약 3GB.

## 1. 구성

```
4. X-ray 검사장비 AI 데이터셋/dataset/
├─ yolov3_20201200.ipynb            실습 노트북 (원본)
├─ yolov3_refactored.ipynb          리팩토링본 (내가 만듦)
├─ OpenLabeling-master/             라벨링 도구
│  └─ main/  input/(bmp 15) output/PASCAL_VOC(xml 833) output/YOLO_darknet(txt 15) config.ini class_list.txt
├─ 라벨링 6종 세트/
│  ├─ images 15|50|100|200|300|400/   jpg(실제 내용은 BMP) 15/50/100/200/300/400장
│  └─ labels/                         YOLO txt 500개
├─ 실습별 가중치파일/                 last15/50/100/200/300/400.pt (각 약 250MB)
└─ test1/yolov3/
   ├─ *.py (train, test, detect, models, splitdata), utils/, yolov3-spp.cfg, requirements.txt
   ├─ custom.data classes.names train.txt test.txt (+ 확장자 깨진 사본들)
   ├─ images/ labels/ (15장)  result/ (추론 결과 15장)  weights/last.pt
   ├─ X선이물검출기(06.23_09.22)/     원본 bmp 2,809장 (호기 3개, NgImage 폴더 131개)
   └─ X선이물검출기/                  호기별 재정리 사본 2,532장 (노트북 실행이 생성 — 원본 자료 아님)
```

## 2. 데이터 사실 (직접 확인)

| 항목 | 내용 |
|---|---|
| 원본 이미지 | bmp 2,809장 = 1호기(SN77128) 1,031 + 2호기(SN77127) 920 + 3호기(SN12053) 858 |
| 촬영 기간 | 2020-06-22 ~ 2020-09-22, 46일, NgImage 폴더 131개 |
| 성격 | 전부 장비가 NG로 판정한 이미지. **정상 이미지 폴더 없음** |
| 중복 | 동일 내용 파일이 같은 호기의 두 폴더에 중복 저장: 277건. 고유 이미지 2,532장. 1·2호기 간 파일명만 같은 충돌 3건(내용 다름) |
| 이미지 크기 | 1호기 352x332, 2호기 316x332, 3호기 576x444 (호기별 고정) |
| 파일명 접두 | `002` = 1·2호기, `001` = 3호기. **호기 번호가 아님** |
| 라벨된 이미지 | 400장(1호기 124 / 2호기 109 / 3호기 167), bbox 1,047개, 클래스 1개(defect). 원본 bmp와 내용이 모두 동일 |
| 라벨 없는 이미지 | 고유 기준 약 2,132장 |
| 서브셋 | 15⊂50⊂100⊂200⊂300⊂400 (중첩). 합계 1,065장이나 고유 400장 |
| 라벨 파일 | 공용 `labels/` 500개 중 100개는 어떤 서브셋에도 쓰이지 않음 |
| 확장자 | 서브셋 `.jpg`는 이름만 jpg이고 실제는 BMP |
| 데모 잔여물 | `OpenLabeling/.../PASCAL_VOC` xml 833개 중 X-ray는 35개, 나머지 798개는 도구 데모(`people_walking_*` 795, `img_*` 3) |

## 3. 가중치 파일

- `실습별 가중치파일/last{15,...,400}.pt`: 이미지 수별 학습본으로 추정(학습 분할·epoch 미확인)
- `test1/yolov3/weights/last.pt`: 원래 806 epoch 학습본이었으나 **현재는 2 epoch 추가 학습본으로 덮어써짐**(아래 참고). 원본은 복구 불가.

## 4. 원본 폴더에서 내가 변경·생성한 것 (읽기 전용이 아님)

"원본 보존"이라고 설명했으나, 노트북 실행과 코드 검증 과정에서 원본 폴더 안 파일이 바뀌었습니다.

| 종류 | 위치(`dataset/test1/yolov3/`) | 내용 |
|---|---|---|
| 코드 수정 | `train.py`, `test.py`, `detect.py`, `models.py`, `utils/utils.py`, `utils/datasets.py` | 호환성 수정(`docs/troubleshooting.md`). `train.py`의 `YOLO_SAVE_DIR`, `datasets.py`의 `imread`/`imwrite`는 `third_party/`에만 있음 |
| 덮어씀 | `weights/last.pt` | 806 epoch → 학습 검증으로 변경됨 (**원본 소실**) |
| 덮어씀 | `train.txt`, `test.txt`, `custom.data`, `results.txt` | 노트북 분할/학습 실행 결과. 원래 내용은 `C:/test1/...` 경로 |
| 생성 | `X선이물검출기/` (2,532장, 약 375MB), `runs/`(TensorBoard 로그 10개), `result/` 갱신, `results.png`, `train.shapes`, `test.shapes`, `__pycache__` | 실행 부산물 |

- 새 작업은 모두 `data/`, `third_party/`, `outputs/`에서 하고 있으며, 원본 폴더는 더 이상 수정하지 않습니다.
- 원본 데이터(bmp, 라벨링 세트, 원래 가중치 6개)는 **변경되지 않았습니다**(라벨 400장은 원본 bmp와 MD5 일치).
- 정리(삭제)할 후보: `X선이물검출기/`, `runs/`, `__pycache__`. 사용자 확인 후 진행.

## 5. 이 조사로 바로잡은 이전 설명

| 이전 설명 | 실제 |
|---|---|
| "라벨된 이미지는 호기 001/002뿐, 3호기 없음" | 파일명 접두일 뿐. 1호기 124 / 2호기 109 / 3호기 167장으로 세 호기 모두 있음 |
| "원본 2,809장 vs 재정리 2,532장 차이는 원인 미확인" | 같은 호기 내 동일 파일 277건 중복(확인 완료) |
| 분할 통계의 "001호기/002호기" | 실제로는 3호기 / 1·2호기 혼합. 실제 호기 기준 분할 분포는 `docs/dataset.md` 6절 |
