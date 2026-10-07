"""발표자료(PPT) 생성: ppt/figures 의 그림과 docs/analysis.md 의 수치로 ppt/presentation.pptx 를 만든다.

  python ppt/make_pptx.py            # presentation.pptx 생성
  (선택) PowerPoint 가 있으면 PDF 변환: PowerShell 에서 ppt/export_pdf.ps1 참고

규칙: 소속(기업·학교명)·로고 등 식별 정보를 넣지 않는다(성명·팀명만). 팀명은 아직 정해지지 않아 [확정 필요]로 둔다.
숫자는 docs/experiments.md, docs/analysis.md 와 같은 값이다. 값이 바뀌면 이 파일의 문자열을 고친다.
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "ppt" / "figures"
FONT = "맑은 고딕"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x3B, 0x6F, 0xD4)
TEAL = RGBColor(0x2A, 0xA7, 0xB8)
RED = RGBColor(0xD6, 0x60, 0x4D)
GRAY = RGBColor(0x55, 0x5B, 0x66)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x22, 0x22, 0x22)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = 13.333, 7.5
page = [0]


def tb(slide, x, y, w, h, text="", size=16, bold=False, color=BLACK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size, r.font.bold, r.font.name = Pt(size), bold, FONT
    r.font.color.rgb = color
    return box


def bullets(slide, x, y, w, h, items, size=17, color=BLACK, gap=6):
    """items: 문자열 또는 (문자열, 단계) 튜플. 단계 1은 들여쓴 하위 항목. '**굵게**' 표기는 앞뒤 ** 로 감싼 부분을 굵게."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    first = True
    for it in items:
        text, lvl = (it, 0) if isinstance(it, str) else it
        text = text.replace("`", "")
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        mark = "• " if lvl == 0 else "– "
        parts = text.split("**")
        r0 = p.add_run()
        r0.text = ("    " * lvl) + mark
        r0.font.size, r0.font.name = Pt(size - 2 * lvl), FONT
        r0.font.color.rgb = TEAL if lvl == 0 else GRAY
        r0.font.bold = True
        for k, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            r.font.size, r.font.name = Pt(size - 2 * lvl), FONT
            r.font.bold = k % 2 == 1
            r.font.color.rgb = color if lvl == 0 else GRAY
    return box


def pic(slide, name, x, y, w, h, caption=None):
    """그림을 (x, y, w, h) 상자 안에 비율을 유지하며 가운데 맞춰 넣는다."""
    path = FIG / name
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    if caption:
        tb(slide, x, y + h - 0.05, w, 0.4, caption, size=11, color=GRAY, align=PP_ALIGN.CENTER)


def table(slide, x, y, w, rows, col_w, size=13, row_h=0.4, header=NAVY, bold_rows=(), hi_rows=()):
    n_r, n_c = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r))
    tbl = shape.table
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(int(Inches(w) * cw / tot))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = ""
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.06)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.size, r.font.name = Pt(size), FONT
            if i == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = header
                r.font.color.rgb = WHITE
                r.font.bold = True
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = RGBColor(0xDD, 0xF1, 0xF4) if i in hi_rows else (LIGHT if i % 2 == 0 else WHITE)
                r.font.color.rgb = BLACK
                r.font.bold = i in bold_rows
    return tbl


def new_slide(title, message=None, note=None):
    s = prs.slides.add_slide(BLANK)
    page[0] += 1
    bar = s.shapes.add_shape(1, 0, 0, Inches(0.18), Inches(SH))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tb(s, 0.55, 0.3, SW - 1.1, 0.8, title, size=28, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    if message:
        tb(s, 0.55, 1.1, SW - 1.1, 0.7, message, size=18, color=TEAL, bold=True)
    tb(s, SW - 1.2, SH - 0.5, 0.9, 0.35, str(page[0]), size=12, color=GRAY, align=PP_ALIGN.RIGHT)
    if note:
        tb(s, 0.55, SH - 0.55, SW - 2.2, 0.4, note, size=11, color=GRAY)
    return s


# ---------------------------------------------------------------- 1 표지
s = prs.slides.add_slide(BLANK)
page[0] += 1
bg = s.shapes.add_shape(1, 0, 0, Inches(SW), Inches(SH))
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
tb(s, 0.9, 1.5, 11.5, 0.6, "제6회 K-인공지능 제조데이터 분석 경진대회", size=20, color=RGBColor(0xBF, 0xD4, 0xF2))
tb(s, 0.9, 2.2, 11.5, 1.8, "점수가 포화된 X-ray 이물 검출,\n점수 뒤의 진짜 성능을 검증하다", size=40, bold=True, color=WHITE)
tb(s, 0.9, 4.3, 11.5, 0.6, "영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석", size=22, color=RGBColor(0x8F, 0xD8, 0xE2))
tb(s, 0.9, 5.6, 11.5, 0.5, "팀명: [확정 필요]", size=18, color=WHITE)
tb(s, 0.9, 6.1, 11.5, 0.5, "허민엽 · 정연창 · 이총", size=18, color=WHITE)

# ---------------------------------------------------------------- 2 문제와 데이터
s = new_slide("문제와 데이터", "약 10px의 작은 점(금속구)을 찾는 검출 문제, 이미지는 모두 NG 판정")
bullets(s, 0.55, 1.9, 5.2, 5.2, [
    "X-ray 이물 검출기 영상 **2,532장**, 박스 4,494개",
    "호기 3종, 해상도 4종 (316x332 ~ 576x444)",
    "train 1,767 / val 369 / test 396장 (연속 촬영 묶음 단위, 시드 42)",
    "**라벨이 두 종류**: 공식 500장은 크기가 제각각(5~21px), 팀 라벨 2,032장은 **고정 크기 네모(10·13px)** → 점수를 부풀림",
    "정상 이미지가 없다(빈 라벨 105장은 제품이 반만 찍힌 사진)",
    "장비의 색 표시는 지운 이미지를 사용",
], size=16)
pic(s, "fig1a_data_labels.png", 5.9, 1.9, 7.0, 4.9)

# ---------------------------------------------------------------- 3 접근
s = new_slide("접근: 점수를 믿기 전에 검증한다", "문제 → 의심 → 검증 → 발견 → 조치의 순서로 분석을 설계했다")
pic(s, "fig0_story_flow.png", 0.6, 1.75, 12.1, 4.6)
bullets(s, 0.8, 6.45, 11.8, 0.9, ["6개 모델(베이스라인 YOLOv3-tiny, Faster R-CNN 2종, 팀원 모델 3종)을 **같은 평가 코드**로 비교"], size=16)

# ---------------------------------------------------------------- 4 점수 포화
s = new_slide("기본 지표는 6개 모델이 구분되지 않는다", "F1(IoU 0.5)은 0.986~0.991, 차이는 박스 정밀도(mAP50-95)에서만 난다")
table(s, 0.6, 1.95, 8.0, [
    ["모델 (test)", "F1", "공식 라벨만 F1", "mAP50-95"],
    ["YOLOv3-tiny (베이스라인)", "0.989", "0.964", "0.558"],
    ["Faster R-CNN R50", "0.989", "0.964", "0.611"],
    ["Faster R-CNN MobileNetV3", "0.986", "0.959", "0.559"],
    ["YOLO26n", "0.991", "0.969", "0.633"],
    ["RT-DETR-l", "0.988", "0.961", "0.639"],
    ["D-FINE-N", "0.989", "0.964", "0.659"],
], [3.4, 1, 1.7, 1.4], size=16, row_h=0.66, bold_rows=(6,), hi_rows=(6,))
bullets(s, 8.9, 1.95, 4.1, 5.0, [
    "모델 임계값은 **val에서 F1 최대**로 정해 고정, test는 한 번만",
    "공식 라벨만 보면 F1은 0.96대로 내려간다",
    "mAP50-95 차이도 공식 라벨만 보면 0.335~0.365로 **좁아진다**",
    "→ 점수 차이는 \"찾는 능력\"이 아니라 박스 맞춤 정밀도",
], size=17, gap=10)

# ---------------------------------------------------------------- 5 중심 거리
s = new_slide("지표 재설계: 중심 거리 평가", "IoU 0.5는 사실상 \"중심이 2px 이내인가\"를 재고 있었다")
pic(s, "fig2b_center_distance.png", 0.6, 1.9, 7.8, 4.9)
bullets(s, 8.6, 1.95, 4.3, 4.8, [
    "약 10px 결함은 박스가 몇 px만 어긋나도 IoU가 급락",
    "\"검출 중심이 정답 중심에서 R(px) 이내\"로 매칭하는 평가 v2 추가",
    "**R ≥ 3px에서 6개 모델 모두 포화**(F1 0.998~1.000)",
    "구분은 R 1~2px의 위치 정밀도에서만 보인다",
], size=16)

# ---------------------------------------------------------------- 6 오류의 정체
s = new_slide("오류의 정체: 놓친 것이 아니라 박스 크기가 다른 것", "test 오류 7건은 전부 공식 라벨 이미지, 점은 모두 예측 박스 안에 있다")
pic(s, "fig3a_fn_cases.png", 0.6, 1.9, 7.9, 4.9)
bullets(s, 8.7, 1.95, 4.2, 4.8, [
    "모델 박스는 점을 정확히 감쌈(중심 거리 0.5~1.2px)",
    "정답 박스가 더 작거나(6x7, 9x5) 더 클 뿐(14x15)",
    "중심 5px 기준으로는 오류 **0건**",
    "→ 모델의 약점이 아니라 **라벨 크기와 채점 기준**의 문제",
], size=16)

# ---------------------------------------------------------------- 7 점 제거
s = new_slide("점 제거 시험: 모델은 점 자체를 본다", "결함의 점을 지우면 검출이 사라진다 → 위치만 보는 지름길이 아니다")
pic(s, "fig3b_dot_removal.png", 0.8, 1.7, 11.7, 3.6)
bullets(s, 0.8, 5.4, 11.8, 1.9, [
    "test 정답 663개의 점을 보간으로 지우고 같은 모델을 다시 적용: 노이즈 없는 평균 보간으로 13x13 이상 지우면 **6개 모델 모두 0~3개**만 검출",
    "YOLOv3-tiny 등에 남는 반응은 보간에 더한 **노이즈 무늬** 때문 → 다음 질문: 노이즈 무늬에 속아 오경보를 내는가?",
], size=16, gap=8)

# ---------------------------------------------------------------- 8 가짜 정상
s = new_slide("가짜 정상 이미지: 노이즈 보간에는 R50과 YOLO26n만 속지 않는다", "정상 이미지가 없어 결함의 점을 지운 이미지로 오경보를 쟀다", note="가짜 정상은 진짜 정상 제품이 아니다. 오경보율은 범위로 해석한다.")
pic(s, "fig3l_robust6.png", 0.5, 1.8, 12.3, 3.9)
bullets(s, 0.7, 5.6, 12.0, 1.7, [
    "NS+노이즈 6px 오경보율: **R50 0.5%, YOLO26n 4.1%**, RT-DETR-l 37%, YOLOv3-tiny 39%, D-FINE-N 49%, MobileNetV3 64%",
    "오경보는 대부분 **지운 자리**에서 발생, 분리도(AUC)는 MobileNet 0.842·D-FINE 0.900이 가장 나쁨",
], size=15, gap=4)

# ---------------------------------------------------------------- 9 합성 점 생성
s = new_slide("합성 점 평가 데이터: 작대기 안에만, 시드 42로 모두가 같게", "실제로는 놓침이 사실상 없어서, 미탐지 조건을 직접 만들어 시험했다")
pic(s, "fig3m_synth_bars.png", 0.5, 1.8, 6.7, 5.3)
bullets(s, 7.4, 1.95, 5.5, 5.0, [
    "실제 점을 지우고 영상에서 찾은 **작대기(막대) 안쪽**에만 val의 실제 점(투과율 조각)을 합성",
    "점의 **진하기(4단계) x 크기(3단계)**를 바꿔 옅은 점까지 시험",
    "test 바탕 386장, 점 1,436개. 실제 점 663개가 **전부 검출된 막대에 닿음**",
    "**시드 42 + 이미지 이름별 시드**로 고정, 기준본과 `--verify`로 팀원 PC에서 같은 데이터인지 확인",
    "합성은 실제보다 약 10%p 덜 현실적 → 절대 수치가 아닌 **상대 비교**",
], size=15)

# ---------------------------------------------------------------- 10 합성 결과
s = new_slide("합성 점 결과: 옅은 점이 미탐지 조건이다", "val 임계값에서는 D-FINE-N이 앞서지만 대부분 느슨한 임계값 효과, 맞추면 차이가 줄어든다", note="합성 점 시험은 test 이미지가 바탕이며 참고용이다. 합성 점 AP와 오경보 예산 비교는 합성 이미지 자체로 계산한 분석용 수치다.")
pic(s, "fig3k_compare6.png", 0.5, 1.8, 12.3, 3.7)
bullets(s, 0.7, 5.45, 12.0, 1.8, [
    "대비(둘레-점 밝기) 6 미만은 모든 모델이 거의 0%, 20 이상에서 약 77~89%",
    "임계값과 무관한 합성 점 AP: **D-FINE-N 0.574 ≈ R50 0.567** > YOLOv3-tiny 0.515 ≈ YOLO26n 0.501 > RT-DETR-l 0.451 >> **MobileNetV3 0.262**",
    "3호기(대비 14 이상): MobileNetV3 4%, YOLO26n·RT-DETR-l 약 61%, R50·YOLOv3-tiny·D-FINE-N은 호기 차이가 작음",
], size=15, gap=3)

# ---------------------------------------------------------------- 11 호기·해상도
s = new_slide("호기·해상도: 뚜렷한 효과는 확인되지 않았다", "가장 낮은 그룹은 공식 라벨 박스가 76%여서 IoU 점수가 낮을 뿐이다")
pic(s, "fig3f_groups_chart.png", 0.6, 1.7, 12.1, 3.7)
bullets(s, 0.8, 5.5, 11.8, 1.8, [
    "해상도별 공식 라벨 박스 비율이 10~76%로 달라 라벨 구성이 섞인다. 공식 라벨 이미지끼리만 비교하면 F1이 0.956~0.977로 좁아지고 구간이 겹친다",
    "그룹별 표본이 작다(공식 라벨 이미지 호기별 20~36장). 합성 점에서 보인 3호기 약점은 **모델에 따라** 다르다(10번)",
], size=16, gap=8)

# ---------------------------------------------------------------- 12 최종 모델
s = new_slide("최종 모델: D-FINE-N", "val 동률 선두 + 미탐지 강건성 최상 + 작다(3.7M)", note="선정 기준 ①②③은 팀원 모델 추가 전에 정했다. '미탐지 우선'은 결과를 본 뒤 정한 우선순위다. 상세는 docs/analysis.md 5절.")
table(s, 0.6, 1.9, 8.2, [
    ["모델", "val mAP50-95\n(전체 / 공식만)", "합성 점 AP", "가짜 정상 오경보\n(NS+노이즈 6)", "파라미터"],
    ["D-FINE-N (선정)", "0.655 / 0.343", "0.574", "49.3%", "3.7M"],
    ["RT-DETR-l", "0.662 / 0.357", "0.451", "36.6%", "32.8M"],
    ["R50", "0.615 / 0.337", "0.567", "0.5%", "41.4M"],
    ["YOLO26n", "0.642 / 0.314", "0.501", "4.1%", "2.5M"],
], [2.3, 2.1, 1.3, 2.0, 1.3], size=15, row_h=0.72, bold_rows=(1,), hi_rows=(1,))
bullets(s, 0.6, 5.45, 8.1, 1.9, [
    "**① 동률 선두**: 전체 val에서 RT-DETR-l과 차이 −0.008 (95% 구간 −0.016~0.000)",
    "**② 강건성**: 미탐지 쪽은 D-FINE-N, 오경보 쪽은 RT-DETR-l이 낫다 → 놓침이 더 비싸다고 보아 **미탐지 우선**",
    "**③ 크기**: RT-DETR-l의 약 1/9",
], size=15, gap=4)
bullets(s, 9.05, 1.95, 3.9, 5.0, [
    "R50·YOLO26n은 val에서 유의하게 낮음(공식 라벨만이면 R50은 동률)",
    "**약점**: 노이즈 보간 오경보 49%, 신뢰도가 0.9 근처에서 포화",
    "**대안**: 오경보 우선 → R50 / YOLO26n, CPU 전용 → YOLOv3-tiny",
    "MobileNetV3는 위치 의존·노이즈 오경보로 비권장",
], size=15)

# ---------------------------------------------------------------- 13 현장 활용
s = new_slide("현장 활용: 3단 판정과 재검사 구간", "실제 test에서는 결함 369장(≥0.748)과 빈 라벨 27장(≤0.04)이 완전히 갈린다", note="재검사 구간의 효과는 합성·가짜 정상에 기반한 참고 수치다. 재검사·놓침 비용은 현장 값을 몰라 가정하지 않았다.")
pic(s, "fig4b_final_dfine.png", 0.5, 1.7, 12.3, 3.9)
bullets(s, 0.8, 5.65, 11.8, 1.7, [
    "**불합격** 최고 신뢰도 ≥ 0.66(val 임계값)  |  **재검사** 0.3 ~ 0.66  |  **합격** < 0.3",
    "하한을 0.3으로 낮추면 옅은 점 검출 61.9% → 67.1%, 대신 가짜 정상 오경보 4.1% → 5.7%(평균 보간), 49% → 59%(노이즈 보간)",
    "오경보가 비싼 현장은 재검사 구간의 이득이 작다",
], size=16, gap=6)

# ---------------------------------------------------------------- 14 선택 가이드
s = new_slide("모델 선택 가이드와 처리 속도", "촬영 간격(중앙값 4초)보다 모든 모델이 빠르다")
table(s, 0.6, 1.9, 12.1, [
    ["상황", "권장", "이유"],
    ["기본(최종 모델)", "D-FINE-N", "val 동률 선두, 미탐지 강건성 최상, 3.7M. 노이즈 오경보 49%"],
    ["오경보가 중요한 현장", "Faster R-CNN R50 또는 YOLO26n", "노이즈 오경보 0.5%·4.1%. R50은 느리고 YOLO26n은 3호기 합성 점에 약함"],
    ["CPU 전용 / 최경량", "YOLOv3-tiny", "CPU 4스레드 43ms, 정확도 손실이 작음"],
    ["비권장", "MobileNetV3", "위치 의존, 합성 점 AP 0.262, 노이즈 오경보 34~64%"],
], [2.2, 3.4, 6.5], size=15, row_h=0.64)
table(s, 0.6, 5.25, 12.1, [
    ["처리 속도 (이미지 1장)", "YOLOv3-tiny", "MobileNetV3", "R50", "D-FINE-N", "YOLO26n", "RT-DETR-l"],
    ["GPU (ms)", "4.8", "11.3", "48.4", "26~92", "14~46", "33~92"],
    ["CPU 4스레드 (ms)", "43", "73", "756", "114~473", "139~400", "496~505"],
], [2.6, 1.5, 1.5, 1.2, 1.5, 1.5, 1.5], size=13, row_h=0.45)
tb(s, 0.6, 6.7, 12.1, 0.6, "앞 3개 모델과 뒤 3개 모델은 측정 방식이 달라(전처리 포함, 두 번 측정한 범위, 실행마다 변동) 직접 비교하지 않는다. 현장 장비 사양은 알지 못한다.", size=11, color=GRAY)

# ---------------------------------------------------------------- 15 차별점
s = new_slide("차별점: 새 알고리즘이 아니라 \"점수를 믿게 만드는 검증\"", "이 검증이 없었다면 \"6개 모델 모두 F1 0.99\"에서 끝났을 것이다")
bullets(s, 0.7, 1.95, 12.0, 5.0, [
    "**평가 재설계**: 공식·팀 라벨 분리, 중심 거리 평가, mAP50-95 → 포화된 점수 뒤의 차이를 드러냄",
    "**점 제거 시험**: 결함을 지워 모델이 점 자체를 보는지 확인 → 지름길 우려 해소",
    "**가짜 정상과 합성 점**: 정상 이미지가 없는 데이터에서 오경보·미탐지 조건(옅은 점, 위치, 호기)을 정량화",
    "**임계값과 무관한 비교**: AP·오경보 예산으로 임계값 효과를 걷어냄(D-FINE-N의 우위가 임계값 효과임을 확인)",
    "**공유 가능한 평가 데이터**: 시드 42 합성 데이터 + 기준본 검증으로 팀 모두가 같은 시험을 재현",
    "**분석을 조치로**: 취약점이 모델 선택 가이드와 재검사 구간으로 이어짐",
], size=20, gap=14)

# ---------------------------------------------------------------- 16 재현성
s = new_slide("코드 구성과 재현성", "전처리 · 학습 · 추론 · 평가 · 분석을 폴더별로 분리하고 실행 명령을 문서화했다", note="전처리 코드, 한 번에 실행하는 스크립트, 테스트 예측 결과 CSV는 제출 전에 정리할 항목이다 [작성 필요].")
bullets(s, 0.7, 1.95, 6.1, 5.0, [
    "환경: conda `KAMP`(Python 3.10), torch 2.6, 한글 경로용 `PYTHONUTF8=1`, `requirements.txt`",
    "`src/yolov3/`: 베이스라인(호환 수정본)",
    "`src/minyeop/`: Faster R-CNN, MobileNet, YOLOv3-tiny, 팀원 모델 평가, 분석(점 제거, 가짜 정상, 합성 점, 호기 통계)",
    "`scripts/synth_eval.py`: 시드 42 합성 평가 데이터 생성기",
    "`docs/`: 지표, 실험 결과 표, 분석 해석",
], size=17, gap=9)
bullets(s, 7.0, 1.95, 5.9, 5.0, [
    "평가 규칙: 임계값은 **val에서 고정**, test는 한 번만",
    "데이터 버전: `manifest.csv`의 sha256으로 확인",
    "합성 평가 데이터: 기준본과 `--verify`로 비교(위치·조건·픽셀 해시)",
    "모델 선정: val 부트스트랩 구간으로 \"동률\"을 객관적으로 판정",
    "모든 수치의 출처 파일을 `docs/`와 `runs/`에 보존(예측 JSON은 제외)",
], size=17, gap=9)

# ---------------------------------------------------------------- 17 한계
s = new_slide("한계와 향후 과제", "숨기지 않고 적는다")
bullets(s, 0.7, 1.95, 12.0, 5.0, [
    "합성 점과 가짜 정상은 **인위적인 이미지**다. 실제 현장 성능이 아니라 상대 비교와 취약점 탐색으로만 해석한다",
    "강건성 시험은 **test 이미지**로 먼저 수행했다. 모델 선정에는 val 지표만 썼지만, 선정 근거로 쓰려면 val로 반복해야 한다",
    "'미탐지 우선'이라는 우선순위는 결과를 본 뒤 정했다. 오경보 우선이면 선정이 바뀐다",
    "표시(색 사각형) 제거의 **흔적**이 단서가 되는지는 검증하지 못했다",
    "학습 시드가 하나이고, 팀원 모델은 학습 조건이 우리와 다르다(임계값은 우리 규칙으로 재선정)",
    "속도는 노트북에서 측정했고 실행마다 변동이 크다. 현장 장비 사양과 재검사 비용은 알지 못한다",
    "향후: 흔적 검증(원본 BMP 2x2 실험), 실제 정상 제품 이미지 확보, val 반복, 합성 점 대조군(s=0)",
], size=18, gap=11)

# ---------------------------------------------------------------- 18 결론
s = new_slide("결론", None)
bullets(s, 0.7, 1.5, 12.0, 5.5, [
    "6개 모델의 기본 점수(F1 0.99)는 구분되지 않았다 → **평가를 다시 설계**했다",
    "\"놓침\"으로 보이던 오류는 **박스 크기 차이**였고, 진짜 약점은 **옅은 결함, 노이즈 무늬 오경보, MobileNet의 위치 의존**이었다",
    "최종 모델은 **D-FINE-N**: val 동률 선두, 미탐지 강건성 최상, 작다. 약점(노이즈 오경보)은 임계값·재검사 구간과 대안 모델로 관리한다",
    "공유 가능한 합성 평가 데이터와 검증 코드로 **팀 모두가 같은 시험을 재현**한다",
], size=24, gap=22)

import datetime

cp = prs.core_properties            # 도구 기본값(작성자 등)을 비우고 제목만 둔다. 식별 정보를 넣지 않는다.
cp.title = "발표자료"
cp.author = ""
cp.last_modified_by = ""
cp.comments = ""
cp.subject = ""
cp.keywords = ""
cp.created = cp.modified = datetime.datetime.now()
out = ROOT / "ppt" / "presentation.pptx"
prs.save(str(out))
print("저장:", out, f"({page[0]}장)")
