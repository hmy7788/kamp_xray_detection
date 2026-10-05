# docs — 문서

데이터, 실험, 평가 기준 같은 **프로젝트 문서**를 두는 폴더입니다.

## 현재 문서
| 문서 | 내용 |
|---|---|
| [metrics.md](metrics.md) | 검출 모델의 성능 지표 정리 (박스·이미지 단위, 조건별 분해, 신뢰도, 불확실성, 주의사항) |

2026-10-05에 이전 문서 7개를 지웠고, 새로 작성하는 문서를 이 폴더에 둡니다.

## 이전 문서 복구
지운 문서는 Git 이력에 남아 있습니다. 마지막으로 존재한 커밋은 `d151bfa`입니다.
```bash
git show d151bfa:docs/<파일명>.md          # 내용 보기
git checkout d151bfa -- docs/<파일명>.md   # 복구
```

| 이전 문서 | 내용 |
|---|---|
| `dataset.md` | 데이터 현황, 호기별 통계, **장비 표시(색 박스) 문제**, 라벨 500장, 표시 제거 방식 비교 |
| `original_folder_inventory.md` | 원본 폴더 전수 조사 |
| `troubleshooting.md` | 환경·코드 호환 문제와 해결 기록 |
| `experiment_results.md` | 실험 설정과 결과 (삭제 전 데이터 기준) |
| `evaluation.md` | 대회 평가 기준(배점) |
| `strategy.md` | 우승 전략 검토 |
| `structure.md` | 디렉터리 구조와 개편 내역 |

## 참고
- 다른 폴더의 README와 `CLAUDE.md`, `CONTRIBUTING.md`에는 위 문서를 가리키는 링크가 남아 있어 지금은 끊어져 있습니다.
- 새로 겪은 문제와 해결은 새 문서에 기록하세요.
