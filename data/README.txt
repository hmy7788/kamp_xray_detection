KAMP X-ray 이물질 탐지 — 확정 데이터 (2026-10-05)
============================================================

폴더 구조
  data/train/images/*.png, data/train/labels/*.txt   학습
  data/val/images,   data/val/labels                검증 (평소 실험 채점)
  data/test/images,  data/test/labels               최종 테스트 (최종 후보만 한 번 채점)
  data/manifest.csv                                 사진 목록 (split, source, labeler, machine, 크기, 촬영 시각, burst_id)

사진 수: 2532장 전부 라벨 있음. 라벨 형식: YOLO txt, 한 줄에 상자 하나 '0 cx cy w h' (0~1 비율). 클래스 0 = defect(금속구)
분할 규칙: burst-level random: test 0.15, val 0.15, train rest (묶음 burst 단위, seed 42). 같은 촬영 묶음(60초 이내 연속)은 한쪽에만 들어간다
manifest sha256: 1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623

[분할 × 장비] 사진 수
machine  1호기  2호기  3호기    합계
split                       
train    672  573  522  1767
val      134  102  133   369
test     137  130  129   396
합계       943  805  784  2532

[분할 × 출처] 사진 수  (official = KAMP 공식 라벨 500, team = 팀이 찍은 라벨 2,032)
source  official  team    합계
split                       
train        356  1411  1767
val           59   310   369
test          85   311   396
합계           500  2032  2532

[장비 × 출처] 사진 수
source   official  team    합계
machine                      
1호기           156   787   943
2호기           177   628   805
3호기           167   617   784
합계            500  2032  2532

[분할 × 월] 사진 수
month    6    7    8     9    합계
split                           
train  315  582  148   722  1767
val     60  129   22   158   369
test    72  130   14   180   396
합계     447  841  184  1060  2532

[장비 × 해상도] 사진 수
resolution  316x332  352x332  412x332  576x444    합계
machine                                             
1호기             431      392      120        0   943
2호기             805        0        0        0   805
3호기               0        0        0      784   784
합계             1236      392      120      784  2532

[분할별 라벨 상자]
       상자 수  빈 라벨(상자 0개) 사진
split                      
train  3225              59
val     606              19
test    663              27
  전체 상자 4494개, 사진당 상자 수 분포 {0: 105, 1: 1393, 2: 1, 3: 1033}
  빈 라벨은 제품이 반만 찍혀 테스트피스가 없는 사진(7/27 1호기 104장 등). 이물질 없음 사진으로 쓸 수 있다

[팀 라벨 작성자]
labeler
heo     674
jung    678
lee     680

상자 크기: 공식 라벨 한 변 중앙값 10px(1·2호기)·13px(3호기). 팀 라벨은 클릭 중심에 같은 크기 네모를 자동 생성
데이터 계보: KAMP 원본 BMP → 중복 제거 → 장비 색상 박스 제거(주변 회색 메움) → 회색조 PNG. 저장소 archive/legacy_v1_fill/README.md
