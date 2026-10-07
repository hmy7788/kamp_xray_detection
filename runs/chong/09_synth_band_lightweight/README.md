# 09 띠 안 합성 이물질 + 경량화 비교 (2026-10-07, 이총)

실제 이물질이 늘 있는 **어두운 띠 안**에 점 신호를 넣고(진하기 5단계, 5,930곳), 같은 합성 사진으로 설정 6개를 비교했습니다. 최종 모델은 `lee003_onnx640_cpu_t4` (08 가중치, ONNX CPU) 입니다.
비교표는 [compare.md](compare.md), 사진은 `gallery/index.html` (배경 사진별로 자리 색 = 잡은 가장 옅은 진하기, 자리마다 넣기 전·1.0·0.7·0.5·0.35·0.25 조각과 설정별 확신도).

| 파일/폴더 | 내용 |
|---|---|
| `compare.md` | 설정 6개의 원본 val 성능·속도, 원본 val 369장 mAP50·mAP50-95·F1 (하네스 계산기), 진하기별·호기별·위치별 검출률, 확신도 중앙값 |
| `samples.jsonl` | 합성 자리 5,930개 (배경 사진, x, y, 진하기, 템플릿, 띠 안 상대 위치) |
| `backgrounds.json`, `build_info.json` | 배경 33장과 띠 정보, 생성 설정 |
| `templates/*.npz` | 점 신호 템플릿 (05 에서 val 공식 라벨로 추출한 잔차) |
| `qa.png`, `placement_m{1,2,3}.png` | 합성 품질 확인용 확대 조각, 호기별 자리 배치 그림 |
| `<설정>/summary.json`, `<설정>/predictions.json` | 설정별 요약과 자리마다 넣기 전·후 확신도 |
| `<설정>/val_full.json` | 원본 val 369장 전체 채점 (mAP50, mAP50-95, 임계값별 P/R/F1·중심 적중, 사진별 예측) |
| `gallery/` | HTML 갤러리 (crops/ 조각, marked/ 전체 사진) |

설정 이름: `lee003_gpu640` (08 = 640 학습) · `lee003_onnx640_cpu_t4` (08, ONNX Runtime CPU 4스레드, **최종 배포**) · `lee001_gpu1024` (01 가중치, 1024) · `lee001_gpu640` (01 가중치, 640 추론) · `lee001_onnx640_cpu_t4` / `_t1` (01, ONNX CPU 4·1스레드).
가중치와 ONNX 파일은 Git 에서 제외됩니다. 코드는 `src/chong/harness/members/lee/synth_band.py`, `val_full.py`, `synth_band_gallery.py`.
