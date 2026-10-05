# runs — 실험 결과

한 번의 실험(학습 + 평가)이 만든 결과를 **실험마다 폴더 하나**로 저장합니다.

```
runs/
├─ baseline/        팀 기준 베이스라인 (공유)
├─ minyeop/         허민엽의 실험
├─ yeonchang/       정연창의 실험
└─ chong/           이총의 실험
       └─ <번호>_<모델>_<설정>/    예: 01_yolov3tiny_nomark/
```

## 실험 폴더에 들어가는 것 (예시)
| 파일 | 내용 | Git |
|---|---|---|
| `config.json` | 실행 설정 (모델, batch, epochs, 데이터) | 포함 |
| `train.log`, `*_eval.log` | 로그 | 포함 |
| `results.txt`, `results.png` | epoch별 학습 곡선 | 포함 |
| `metrics.json`, `eval_report_*.json` | 평가 지표 | 포함 |
| `results.csv`, `report_<split>.json` | epoch별 지표, 보고 표 계산 결과 (부트스트랩 구간 등) | 포함 |
| `figures/` | 검출 결과 그림 (데이터 이미지가 그려져 있어 저장소가 비공개인 동안에만 올림) | 포함 |
| `weights/` | 가중치 (`*.pt`, 수십~수백 MB) | **제외** |
| `preds_*.json` | 검출 결과 캐시 (정답 bbox 좌표 = KAMP 라벨 포함) | **제외** |
| `status.json`, `progress.log` | 진행 상황 | **제외** |

## 규칙
- 비교 실험은 **같은 분할과 같은 평가 방식**을 씁니다.
- 숫자는 평가 결과 파일에서만 가져오고 손으로 옮겨 적지 않습니다.
- **올리기 전에** 로그와 `config.json`에 개인 PC 경로(사용자 이름 포함)가 없는지 확인하고, 있으면 `<KAMP>`로 치환합니다. 대회 제출물에는 소속·로고 등 식별 정보를 넣을 수 없습니다.
