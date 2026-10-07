# 협업 계약

## 1. 저장소

- 저장소 하나. `main` 은 보호한다.
- 각자 자기 브랜치(`lee`, `jung`, `heo`)에서 일하고 하루 한 번 `main` 에 합친다.
- `members/<자기 이니셜>/`, `configs/<자기 이니셜>/`, `runs/` 는 자기 브랜치에서 자유롭게 커밋한다.
- `common/`, `docs/contracts/`, `data/splits/`, `data/labels/`, `data/manifest.*`, `.claude/` 는 PR로만 바꾸고 다른 한 명이 승인한다. 에이전트는 이 폴더에 직접 쓰지 못한다 (후크). 사람이 PR 작업을 할 때는 환경 변수 `KAMP_ALLOW_SHARED=1` 을 켜고 한다.
- 전달 통로는 git 뿐이다. 메신저나 드라이브로 코드·설정을 주고받지 않는다. 가중치는 마지막 앙상블 때만 드라이브로 모은다.

## 2. 멤버 식별

환경 변수 `KAMP_MEMBER` 에 이니셜을 넣는다. 에이전트와 후크는 이 값으로 "내 폴더"와 "남의 폴더"를 구분한다. 없으면 run.py 가 멈춘다.

```
PowerShell:  $env:KAMP_MEMBER = "lee"      (영구: setx KAMP_MEMBER lee)
bash:        export KAMP_MEMBER=lee
```

## 3. 하루 동기화 (/sync)

1. `main` 을 받아 합치고 `common/aggregate.py` 로 공유 표를 다시 만든다.
2. 보는 순서: 각자 어제 실험과 고른 이유 → 조건별 표에서 가장 낮은 칸 → 열린 결정 파일 → `docs/insights.md` 새 항목.
3. 결정이 나면 `/decide <번호>` 로 결정 파일을 채운다.

## 4. 영역 관리자

| 영역 | 관리자 |
|---|---|
| 앞단 (데이터 검사, 라벨링 서버, 분할, 조건) | lee |
| 뒷단 (채점, 집계, 보고서 점검) | (정할 것) |
| 하네스와 문서 (.claude, docs) | (정할 것) |

## 5. 새 사람·새 머신 시작 절차 (/setup)

저장소 받기 → `data/kamp_data.zip` 을 받아 `data/` 에 풀기 → `KAMP_MEMBER` 설정 → `/setup` (`python reproduce.py`: 데이터 검사, 조건, 환경 기록). 전부 통과해야 실험을 시작한다.
