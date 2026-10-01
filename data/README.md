# data

정리된 학습용 데이터 **사본**입니다. 원본(`4. X-ray 검사장비 AI 데이터셋/dataset/`)은 수정하지 않았습니다.

```
data/
├─ classes.names        # defect
├─ samples/             # 실습 기본 세트 (test1/yolov3): 이미지 15 / 라벨 15
│  ├─ images/  labels/
└─ subsets/             # 이미지 수별 서브셋 (이미지와 라벨이 1:1로 짝지어진 것만 복사)
   ├─ n015/ n050/ n100/ n200/ n300/ n400/
   │  ├─ images/  labels/
```

- 라벨은 YOLO 포맷(`class x y w h`, 0~1 정규화), 클래스 0 = `defect`.
- 서브셋은 `라벨링 6종 세트/images N`의 이미지와 공용 `labels/`에서 파일명이 같은 라벨을 골라 복사했습니다. 미사용 라벨 100개와 OpenLabeling 데모 XML, 원본 bmp, 가중치 파일은 포함하지 않습니다.
- 통계와 유의 사항은 [../docs/dataset.md](../docs/dataset.md) 참고.
- 학습 코드(`test1/yolov3`)는 아직 이 폴더를 가리키지 않습니다. 사용하려면 `custom.data`와 `train.txt`/`test.txt`를 이 경로 기준으로 다시 만들어야 합니다.

## Git 업로드 시 포함 범위

`.gitignore`로 `samples/images`, `samples/labels`, `subsets/`(이미지·라벨), 실행마다 재생성되는 목록 파일(`splits/*.txt|*.data`)은 제외됩니다. `classes.names`, `manifest.csv`, `splits/split.csv`만 올라갑니다.
