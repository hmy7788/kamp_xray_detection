# 평가 계약

채점은 `python common/evaluate.py runs/<exp_id>` 한 명령으로 끝난다. 사람도 에이전트도 지표를 손으로 계산하지 않는다.

## 1. 주 지표

| 지표 | 정의 |
|---|---|
| map50 | 상자 겹침(IoU) 0.5 기준 평균 정밀도. 공모전 관행 |
| map50_95 | IoU 0.5~0.95 평균 |
| img_f1 | 사진 단위 "이물질 있음/없음" F1. 확신도 임계값은 검증(val) 예측 전체에서 상자 F1이 최대인 값 하나를 쓴다 |
| box_recall, box_precision | IoU 0.5, 임계값 적용 후 |

## 2. 보조 지표 (오류 분석용)

| 지표 | 정의 |
|---|---|
| center_hit_recall | 예측 중심이 정답 상자 안에 들어오면 맞은 것으로 치는 재현율. 상자가 15px 정도라 2px 흔들림을 걸러 보기 위한 것 |
| center_hit_precision | 같은 기준의 정밀도 |

## 3. 조건별 표

`common/conditions.yaml` 의 모든 축 × 위 지표. 축이 늘면 표도 는다. 정답 상자 기준으로 재현율을 나눈다.

## 4. 임계값 표

확신도 임계값을 0.05 간격으로 훑어서 (재현율, 정밀도, 재검사 비율 = 양성 판정 사진 비율, 놓친 상자 수)를 적는다. "목표 재현율을 만족하는 가장 높은 임계값"과 그때의 재검사 비율을 `safety` 항목으로 따로 적는다. 목표 재현율은 결정 007 전까지 0.99.

## 5. 시드

폴드는 없다 (data/train 으로 학습, data/val 로 채점). 지표마다 시드별 값, 평균, 표준편차를 적는다. 개선 주장은 "평균 차이가 노이즈 바닥 실험(같은 설정, 시드 3개)의 표준편차보다 큰가"로 심사한다.

## 6. metrics.json 구조

```json
{
  "exp_id": "...", "threshold": 0.35,
  "part": "val", "overall": {"map50": {"mean": 0, "std": 0, "seeds": [0, 0, 0]}, "img_f1": {}},
  "by_condition": {"machine": {"1": {"box_recall": 0, "center_hit_recall": 0, "n": 0}}, "size_bin": {}},
  "threshold_sweep": [{"thr": 0.05, "recall": 0, "precision": 0, "recheck_rate": 0, "missed": 0}],
  "safety": {"target_recall": 0.99, "threshold": 0.2, "recheck_rate": 0.0}
}
```

## 7. 테스트 (test)

최종 테스트 점수(data/test, `metrics_test.json`)는 `python reproduce.py --final` 로만 낸다. 최종 후보가 정해진 뒤 한 번만 돌린다. 그 전에는 코드도 사람도 test 목록을 읽지 않는다.
