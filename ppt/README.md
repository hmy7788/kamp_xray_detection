# ppt — 보고서·발표 자료

대회 **결과 보고서 초안**과 보고서·발표(PPT)에 쓸 **그림**을 관리하는 폴더입니다. 최종 제출본은 대회 제공 원본 양식으로 만들어 PDF로 내고, 이 폴더는 그 재료입니다. 소속·로고 등 식별 정보는 넣지 않습니다(성명·팀명만).

## 구성
| 경로 | 내용 |
|---|---|
| `report_draft.md` | 보고서 초안(양식의 6개 장 + 표지 + 설문 캡처). 빈 곳은 `[작성 필요]`, `[확정 필요]`, `[확인 필요]`로 표시, 맨 앞에 체크리스트 |
| `make_figures.py` | `ppt/figures/`의 그림을 결과 파일에서 직접 만드는 스크립트 (`python ppt/make_figures.py`, `PYTHONUTF8=1`) |
| `figures/` | 생성·복사된 그림(PNG) |
| `presentation.pptx`, `presentation.pdf` | **발표자료** 29장(제출은 PDF+PPT). 최종 모델 D-FINE-N 기준, 팀명은 [확정 필요] |
| `make_example_figures.py` | 발표자료용 예시 사진·그래프(fig5a~e, fig3n) 생성. `PYTHONUTF8=1 python ppt/make_example_figures.py [a b c d e n]` |
| `make_pptx.py` | 발표자료 생성 스크립트(`python ppt/make_pptx.py`, python-pptx). 숫자는 `docs/`와 같은 값이라 결과가 바뀌면 이 파일의 문자열을 고친다. 파일 속성(작성자 등)은 비운다 |
| `export_pdf.ps1` | PowerPoint(Windows)로 PDF 변환, `-PngDir`로 슬라이드 PNG도 내보냄(레이아웃 점검용) |

## 그림 목록 (장별)
| 그림 | 내용 | 장 | 만드는 곳 |
|---|---|---|---|
| `fig0_story_flow.png` | 이야기 흐름(문제 → 의심 → 검증 → 발견 → 조치) | 표지·요약 | `make_figures.py` |
| `fig1a_data_labels.png` | 정답 박스 크기 분포(공식 vs 팀 라벨), 호기·해상도별 이미지 수 | 1 | `make_figures.py` |
| `fig2a_saturation.png` | F1은 세 모델 동일, mAP50-95는 R50이 앞섬 | 2 | `make_figures.py` |
| `fig2b_center_distance.png` | 중심 거리 허용 R별 F1: R ≥ 3px에서 포화 | 2 | `make_figures.py` |
| `fig2c_accuracy_speed.png` | 정확도(mAP50-95)와 속도(GPU·CPU), 촬영 간격 4초 | 2, 4 | `make_figures.py` |
| `fig3a_fn_cases.png` | 오류로 집계된 7건의 확대(박스 크기 차이) | 3 | 복사: `runs/minyeop/02_.../figures/test/miss_cases.png` |
| `fig3b_dot_removal.png` | 점 제거 후 남은 검출(방식·크기별) | 3 | `make_figures.py` |
| `fig3c_false_alarm.png` | 가짜 정상 이미지 오경보(방식별, 모델별) | 3 | `make_figures.py` |
| `fig3d_synth_contrast_curve.png` | 점의 대비별 합성 점 검출률(원래/무작위 자리) | 3 | `make_figures.py` |
| `fig3e_synth_heatmap.png` | 합성 점 검출률 격자(진하기 × 크기) | 3 | `make_figures.py` |
| `fig3f_groups_chart.png` | 호기·해상도별 성능과 공식 라벨 비율 | 3 | 복사: `runs/minyeop/07_group_stats_v1/figures/` |
| `fig3g_synth_examples.png`, `fig3i_synth_full.png` | 합성한 점의 실제 모양과 위치 | 3 | 복사: `runs/minyeop/06_synth_insert_v1/figures/` |
| `fig3h_group_samples.png` | 호기·해상도 그룹별 검출 예시 | 3 | 복사: `runs/minyeop/07_group_stats_v1/figures/` |
| `fig3j_synth_eval_bars.png` | 막대 안 합성 점 검출률(세 모델): 대비별, 호기별 | 3 | 복사: `runs/minyeop/08_synth_eval_v1/figures/` |
| `fig3k_compare6.png` | 6개 모델 합성 점 비교: val 임계값, 임계값과 무관한 AP, 호기별 | 3 | 복사: `runs/minyeop/09_extra_models_v1/figures/` |
| `fig3l_robust6.png` | 6개 모델 점 제거·가짜 정상 히트맵(오경보율, AUC) | 3 | 복사: `runs/minyeop/09_extra_models_v1/figures/` |
| `fig4b_final_dfine.png` | 최종 모델 D-FINE-N: 실제 test 신뢰도 분포와 임계값별 검출·오경보 | 4 | 복사: `runs/minyeop/09_extra_models_v1/figures/` |
| `fig3m_synth_bars.png` | 작대기 안 합성 점 미리보기(호기마다 1장) | 3 | `make_figures.py`(`fig3_bar_preview`, 임시 폴더에 시드 42로 다시 생성) |
| `fig5a_preprocess.png` | 전처리 전후(원본 BMP의 색 윤곽 → 제거 후), 호기 1·3 | 1, 발표 | `make_example_figures.py` (원본 BMP가 있는 PC에서만) |
| `fig5b_detection_examples.png` | 실제 test에서 6개 모델의 검출(호기별 3장 + IoU 미달 예시) | 2, 발표 | `make_example_figures.py` |
| `fig5c_fake_examples.png` | 가짜 정상 예시: 원본 / 평균 보간 / NS+노이즈, 모델별 반응 | 3, 발표 | `make_example_figures.py` |
| `fig5d_fn_by_model.png` | 모델별 못 찾은 개수: IoU 0.5 / 중심 2px / 중심 5px | 2, 발표 | `make_example_figures.py` |
| `fig5e_select_ci.png` | val mAP50-95와 95% 신뢰구간(모델 선정) | 2, 발표 | `make_example_figures.py` |
| `fig6a_label_whole.png` | 라벨 예시: 이미지 전체에 박스(공식 라벨 / 팀 라벨, 1·3호기) | 1, 발표 | `make_example_figures.py` |
| `fig6b_label_zoom.png` | 박스 확대: 공식은 6~21px 제각각, 팀은 10·13.4px 고정 | 1, 발표 | `make_example_figures.py` |
| `fig3n_synth_dots.png` | 합성 점 조건별 확대(진하기·크기·대비) | 3, 발표 | 복사: `data_synth/test/preview_dots.png` |
| `fig4a_threshold_tradeoff.png` | 임계값에 따른 옅은 점 검출 vs 오경보 → 재검사 구간 근거 | 4 | `make_figures.py` |

모델 색은 모든 그림에서 같다: R50 파랑, YOLOv3-tiny 주황, MobileNetV3 초록.

## 규칙
- 그림의 숫자는 `runs/minyeop/` 결과 파일에서 읽는다. 예외로 `fig2b`의 평가 v2 값과 `fig2c`의 속도·파라미터 값은 `docs/experiments.md`, `docs/analysis.md`의 표에서 스크립트에 옮겨 두었으니 그 표가 바뀌면 `make_figures.py`의 `V2`와 `fig2_tradeoff`도 고친다.
- 그림에는 **데이터 이미지가 그려진 것**(fig3a, fig3g~i)이 있다. 저장소가 비공개인 동안에만 Git에 둔다.
- 해석과 한계는 `docs/analysis.md`가 원문이고, `report_draft.md`는 그것을 보고서 문체로 옮긴 것이다. 두 문서가 어긋나면 `docs/analysis.md`를 기준으로 고친다.
- 새 그림은 `make_figures.py`에 함수를 추가하고 이 표에 한 줄 적는다.

- `presentation.pptx/pdf`에는 데이터 이미지가 들어간 그림이 있어 저장소가 비공개인 동안에만 Git에 둔다. 제출 전에 팀명을 넣고, 식별 정보(소속, 로고)가 없는지 PDF 속성까지 확인한다.
