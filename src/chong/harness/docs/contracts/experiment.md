# 실험 계약

모델은 바꿔 끼우는 슬롯이다. 어떤 코드로 모델을 짜든 아래 세 가지만 지키면 채점·집계·앙상블이 돌아간다.
(1) 설정 파일 하나로 실험을 정의한다. (2) 정해진 형식의 예측 CSV를 낸다. (3) `common/run.py` 로만 실행한다.
데이터는 data/train 으로 학습하고 data/val 로 채점한다 (폴드 없음). 최종 후보만 train+val 로 학습해 data/test 를 한 번 채점한다.

## 1. 설정 파일 (configs/<멤버>/<exp_id>.yaml)

공통 필드는 이름을 바꾸지 않는다. 집계와 채점이 읽는다.

```yaml
exp_id: lee_001          # <멤버>_<세 자리 번호>
member: lee              # KAMP_MEMBER 와 같아야 한다
family: yolo_ref         # 모델 계열 이름
seeds: [42]              # 노이즈 바닥·최종 후보는 [42, 7, 2024] (시드별로 학습·채점해 평균·표준편차)
parent_exp: null         # 이 실험이 무엇을 바꾼 실험인지. 첫 실험만 null
change: "입력 해상도 640 -> 1024"   # 변경점 하나를 한 줄로. parent 대비 model 영역 diff 가 한 항목이어야 한다
rubric_items: [2, 3]     # 평가표 1~6 중 이 실험이 근거가 되는 항목
reason: ""               # 사람이 이 실험을 고른 이유. 비어 있으면 run.py 가 거부한다
entry: members/lee/entry.py   # 진입점. 공용 기준선은 common/baselines/<이름>/entry.py
model:                   # 자유 영역. 멤버 코드만 읽는다
  arch: yolov8n
  imgsz: 1024
  epochs: 50
```

## 2. 멤버 진입점 (members/<멤버>/entry.py)

```python
def run_split(cfg: dict, data: dict, seed: int, out_dir: Path) -> Path:
    """train_ids 로 학습하고 val_ids 전부에 대한 예측 CSV 경로를 돌려준다.
    data = {"name": "val" | "test", "train_ids": [...], "val_ids": [...],
            "images": {image_id: Path}, "labels": {image_id: Path}}   # 사진·라벨 경로는 여기서만 얻는다
    가중치는 out_dir/weights/ 에 둔다 (git 제외)."""
```

- 진입점은 `common/run.py` 만 부른다. `python members/<멤버>/entry.py` 처럼 직접 실행하지 않는다 (후크로 차단).
- 멤버 코드는 `data["images"]`, `data["labels"]` 의 경로만 읽는다. 다른 경로를 조립하거나 `data/` 를 직접 훑지 않는다.
- ultralytics 처럼 images/ ↔ labels/ 폴더 쌍이 필요한 도구는 `common/baselines/yolo_ref/entry.py` 처럼 out_dir 아래에 하드링크로 실험용 폴더를 만든다.

## 3. 예측 파일 (runs/<exp_id>/preds_val.csv, 최종은 preds_test.csv)

| 열 | 뜻 |
|---|---|
| image_id | manifest 의 image_id |
| seed | 시드 |
| cx, cy, w, h | 0~1 정규화. YOLO 라벨과 같은 뜻 |
| conf | 0~1 확신도 |

- 채점 대상 사진(val) 전부가 한 번 이상 등장해야 한다. 아무것도 못 찾은 사진은 `cx,cy,w,h,conf` 를 비운 행 하나를 남긴다.
- 시드가 여러 개면 같은 파일에 seed 열로 구분한다.
- `common/check_preds.py` 가 형식을 검사한다. 통과하지 못한 예측은 채점되지 않는다.

## 4. 실험 기록 (runs/<exp_id>/record.json)

`common/run.py` 가 쓴다. 손으로 고치지 않는다.

```json
{
  "exp_id": "lee_001", "member": "lee", "family": "yolo_ref",
  "manifest_sha256": "...",
  "config_sha256": "...", "git_commit": "...",
  "parent_exp": null, "change": "...", "reason": "...", "rubric_items": [2, 3],
  "env": {"os": "...", "python": "...", "torch": "...", "cuda": "...", "gpu": "..."},
  "n_train": 1771, "n_val": 380, "started_at": "...", "finished_at": "...",
  "runs": {"42": {"status": "ok", "minutes": 3.2}},
  "summary": {"map50_mean": 0.0, "map50_std": 0.0, "img_f1_mean": 0.0}
}
```

## 5. 실험 폴더 (runs/<exp_id>/)

```
config.yaml        실행 시점의 설정 사본
record.json        기록
preds_val.csv      검증 예측 (최종 후보는 preds_test.csv 도)
metrics.json       채점 결과 (common/evaluate.py). 최종은 metrics_test.json
analysis.md        분석가가 쓰는 해석 (조건별 변화, 이전 실험 대비)
errors/            오류 사례 목록 csv (이미지는 git 제외)
weights/           git 제외
```

## 6. 공유 표 (reports/experiments.csv)

`python common/aggregate.py` 가 모든 record.json 과 metrics.json 을 모아 다시 만든다. 손으로 고치지 않는다. 검사를 통과한 실험만 들어간다.

## 7. 실험 번호와 변경점 규칙

- 실험 하나는 변경점 하나다. parent_exp 대비 `model` 영역에서 바뀐 키가 둘 이상이면 run.py 가 거부한다.
- 가중치는 저장소에 넣지 않는다. 예측·기록·지표·분석은 넣는다.
