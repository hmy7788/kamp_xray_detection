"""발표자료(PPT) 생성: ppt/figures 의 그림과 docs/analysis.md 의 수치로 ppt/presentation.pptx 를 만든다.

  python ppt/make_pptx.py            # presentation.pptx 생성
  (선택) PowerPoint 가 있으면 PDF 변환: powershell -File ppt/export_pdf.ps1

이야기 순서(실험한 순서): ① 학습·검증·평가 → ② 몇 개 못 찾음 → ③ 원인은 라벨 박스 크기, 중심 거리로 재채점하면 거의 다 찾음
→ ④ 일반화·robust 평가(점 제거, 가짜 정상, 합성 점, 호기별) → ⑤ 종합해 베스트 모델 선정 → ⑥ 현장 활용.
규칙: 소속(기업·학교명)·로고 등 식별 정보를 넣지 않는다(성명·팀명만). 팀명은 아직 정해지지 않아 [확정 필요]로 둔다.
숫자는 docs/experiments.md, docs/analysis.md 와 같은 값이다. 값이 바뀌면 이 파일의 문자열을 고친다.
"""
import datetime
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "ppt" / "figures"
FONT = "맑은 고딕"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x2A, 0xA7, 0xB8)
GRAY = RGBColor(0x55, 0x5B, 0x66)
LIGHT = RGBColor(0xF2, 0xF5, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x22, 0x22, 0x22)
HI = RGBColor(0xDD, 0xF1, 0xF4)

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
    """items: 문자열 또는 (문자열, 단계). 단계 1은 하위 항목. **굵게** 표기 지원. 백틱은 제거한다."""
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
        r0 = p.add_run()
        r0.text = ("    " * lvl) + ("• " if lvl == 0 else "– ")
        r0.font.size, r0.font.name, r0.font.bold = Pt(size - 2 * lvl), FONT, True
        r0.font.color.rgb = TEAL if lvl == 0 else GRAY
        for k, part in enumerate(text.split("**")):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            r.font.size, r.font.name = Pt(size - 2 * lvl), FONT
            r.font.bold = k % 2 == 1
            r.font.color.rgb = color if lvl == 0 else GRAY
    return box


def pic(slide, name, x, y, w, h):
    """그림을 (x, y, w, h) 상자 안에 비율을 유지하며 가운데 맞춰 넣는다."""
    iw, ih = Image.open(FIG / name).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    slide.shapes.add_picture(str(FIG / name), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def table(slide, x, y, w, rows, col_w, size=13, row_h=0.4, bold_rows=(), hi_rows=()):
    n_r, n_c = len(rows), len(rows[0])
    tbl = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(row_h * n_r)).table
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
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = NAVY
                r.font.color.rgb = WHITE
                r.font.bold = True
            else:
                c.fill.fore_color.rgb = HI if i in hi_rows else (LIGHT if i % 2 == 0 else WHITE)
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


# ================================================================ 1 표지
s = prs.slides.add_slide(BLANK)
page[0] += 1
bg = s.shapes.add_shape(1, 0, 0, Inches(SW), Inches(SH))
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
tb(s, 0.9, 1.5, 11.5, 0.6, "제6회 K-인공지능 제조데이터 분석 경진대회", size=20, color=RGBColor(0xBF, 0xD4, 0xF2))
tb(s, 0.9, 2.2, 11.5, 1.8, "X-ray 이물 검출 모델 6종을 비교하고\n일반화·강건성까지 검증해 베스트 모델을 선정하다", size=36, bold=True, color=WHITE)
tb(s, 0.9, 4.3, 11.5, 0.6, "영상 기반 완제품 이물질 탐지 및 AI 미탐지 조건 분석", size=22, color=RGBColor(0x8F, 0xD8, 0xE2))
tb(s, 0.9, 5.6, 11.5, 0.5, "팀명: [확정 필요]", size=18, color=WHITE)
tb(s, 0.9, 6.1, 11.5, 0.5, "허민엽 · 정연창 · 이총", size=18, color=WHITE)

# ================================================================ 2 문제와 데이터
s = new_slide("문제와 데이터", "약 10px의 작은 점(금속구)을 찾는 검출 문제, 이미지는 모두 NG 판정")
bullets(s, 0.55, 1.9, 5.2, 5.2, [
    "X-ray 이물 검출기 영상 **2,532장**, 박스 4,494개",
    "호기 3종, 해상도 4종 (316x332 ~ 576x444)",
    "train 1,767 / val 369 / test 396장 (연속 촬영 묶음 단위, 시드 42)",
    "**라벨이 두 종류**: 공식 500장은 사람이 그려 크기가 제각각(5~21px), 팀 라벨 2,032장은 클릭 중심에 **고정 크기 네모(10·13px)** 를 자동 생성",
    "정상 이미지가 없다(빈 라벨 105장은 제품이 반만 찍힌 사진)",
    "**전처리**: 원본 BMP → 중복 제거(277쌍) → 장비의 색 표시 제거(주변 회색 5x5 평균으로 메움) → 회색조 PNG. 원본에 다시 적용하면 2,532장이 **픽셀 단위로 모두 일치**",
], size=15)
pic(s, "fig1a_data_labels.png", 5.9, 1.9, 7.0, 4.9)

# ================================================================ 3 실험 흐름
s = new_slide("실험 흐름", "학습·평가 → 못 찾은 것의 원인 → 일반화·robust 평가 → 베스트 모델")
FLOW = [("① 학습·검증\n·평가", "6개 모델\n임계값 val 고정\ntest는 한 번", RGBColor(0xDD, 0xE6, 0xF7)),
        ("② 못 찾은\n것이 있다", "정답 663개 중\n6~9개 놓침\nF1 0.99 안팎", RGBColor(0xFB, 0xE7, 0xCC)),
        ("③ 원인은\n라벨 문제", "공식 라벨\n박스 크기 문제\n중심 거리로\n재채점\n→ 거의 다 찾음", RGBColor(0xD6, 0xEC, 0xDC)),
        ("④ 일반화\n·robust 평가", "점 제거\n가짜 정상\n합성 점\n호기별", RGBColor(0xF5, 0xD9, 0xD9)),
        ("⑤ 종합 →\n베스트 모델", "val 동률 선두\n미탐지 강건성\n최상\n→ D-FINE-N", RGBColor(0xE6, 0xDC, 0xF0)),
        ("⑥ 현장 활용", "3단 판정\n재검사 구간\n모델 선택 가이드", RGBColor(0xF1, 0xEC, 0xD2))]
for i, (ft, fb, fc) in enumerate(FLOW):
    fx = 0.5 + i * (1.93 + 0.16)
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(fx), Inches(2.0), Inches(1.93), Inches(3.3))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fc
    shp.line.color.rgb = GRAY
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.08)
    first_para = True
    t_lines = ft.split("\n")
    for k, line in enumerate(t_lines):
        pp = tf.paragraphs[0] if first_para else tf.add_paragraph()
        first_para = False
        pp.alignment = PP_ALIGN.CENTER
        pp.space_after = Pt(10 if k == len(t_lines) - 1 else 0)
        rr = pp.add_run()
        rr.text = line
        rr.font.size, rr.font.bold, rr.font.name = Pt(16), True, FONT
        rr.font.color.rgb = NAVY
    for line in fb.split("\n"):
        pp = tf.add_paragraph()
        pp.alignment = PP_ALIGN.CENTER
        pp.space_after = Pt(3)
        rr = pp.add_run()
        rr.text = line
        rr.font.size, rr.font.name = Pt(13), FONT
        rr.font.color.rgb = BLACK
    if i < len(FLOW) - 1:
        tb(s, fx + 1.93 - 0.04, 3.45, 0.24, 0.4, "▶", size=10, color=GRAY, align=PP_ALIGN.CENTER)
bullets(s, 0.8, 5.6, 11.8, 1.7, [
    "모델 6개: 베이스라인 YOLOv3-tiny, Faster R-CNN(ResNet-50, MobileNetV3), 팀원 모델 YOLO26n · RT-DETR-l · D-FINE-N",
    "모든 모델을 **같은 평가 코드**로 채점하고, 임계값은 val에서 정해 고정, test는 한 번만 본다",
], size=16, gap=6)

# ================================================================ 4 학습·검증·평가 설정
s = new_slide("① 학습 · 검증 · 평가", "지금 있는 데이터로 6개 모델을 같은 조건에서 학습하고 평가했다")
table(s, 0.6, 1.95, 8.3, [
    ["모델", "입력", "학습 설정", "val 임계값"],
    ["YOLOv3-tiny (베이스라인)", "640", "COCO 사전학습, 100 epoch", "0.06"],
    ["Faster R-CNN ResNet-50", "640", "COCO 사전학습, 20 epoch(선택 7)", "0.95"],
    ["Faster R-CNN MobileNetV3", "640", "COCO 사전학습, 20 epoch(선택 18)", "0.93"],
    ["YOLO26n (팀원)", "1024", "COCO 사전학습, 40 epoch", "0.42"],
    ["RT-DETR-l (팀원)", "640", "COCO 사전학습, 최대 30 epoch(계획)", "0.74"],
    ["D-FINE-N (팀원)", "640", "학습 설정 기록 [확인 필요]", "0.66"],
], [3.0, 0.9, 3.6, 1.4], size=14, row_h=0.62)
bullets(s, 9.2, 1.95, 3.8, 5.0, [
    "같은 데이터·분할, 같은 평가 코드",
    "임계값은 모델마다 **val에서 F1 최대**로 정해 고정",
    "**test는 한 번만** 평가",
    "팀원 모델은 가중치를 받아 같은 코드로 다시 평가(임계값도 다시 선정)",
], size=16, gap=10)

# ================================================================ 5 1차 결과
s = new_slide("결과: 몇 개를 못 찾았다", "F1은 0.986~0.991, 정답 663개 중 6~9개를 놓쳤고 거의 전부 공식 라벨 이미지다")
table(s, 0.6, 1.95, 8.0, [
    ["모델 (test, IoU 0.5)", "F1", "못 찾음(FN)", "그중 공식 라벨 이미지"],
    ["YOLOv3-tiny (베이스라인)", "0.989", "7", "7"],
    ["Faster R-CNN R50", "0.989", "7", "7"],
    ["Faster R-CNN MobileNetV3", "0.986", "9", "8"],
    ["YOLO26n", "0.991", "6", "6"],
    ["RT-DETR-l", "0.988", "8", "7"],
    ["D-FINE-N", "0.989", "7", "7"],
], [3.2, 1.0, 1.5, 2.3], size=15, row_h=0.64)
bullets(s, 8.9, 1.95, 4.1, 5.0, [
    "공식 라벨만 보면 F1은 0.96대로 내려간다",
    "FP도 같은 수 → 같은 점을 **어긋난 박스**로 낸 것",
    "→ 정말 놓친 것일까? 놓친 자리를 직접 확인했다",
], size=17, gap=12)

# ================================================================ 6 원인
s = new_slide("원인: 공식 라벨의 박스 크기", "놓친 것이 아니라 정답 박스의 크기가 달라서 IoU가 0.5에 못 미친 것이다")
pic(s, "fig3a_fn_cases.png", 0.6, 1.9, 7.6, 4.9)
bullets(s, 8.4, 1.95, 4.5, 5.0, [
    "오류가 난 이미지는 거의 전부 **공식 라벨**(사람이 그려 크기가 제각각)",
    "확대하면 모델 박스는 점을 정확히 감쌈(중심 거리 0.5~1.2px)",
    "정답 박스가 더 작거나(6x7, 9x5) 더 큼(14x15)",
    "팀 라벨은 고정 크기라 IoU가 잘 맞아 점수가 부풀려진다",
], size=16, gap=9)

# ================================================================ 7 중심 거리
s = new_slide("해결: 중심 거리로 재채점", "중심이 정답에서 R(px) 이내면 찾은 것으로 보면 거의 다 찾는다")
pic(s, "fig2b_center_distance.png", 0.5, 1.9, 7.3, 4.9)
bullets(s, 8.0, 1.95, 5.0, 5.0, [
    "R = 5px로 재채점하면 못 찾은 것이 4개 모델은 **0개**, MobileNetV3·RT-DETR-l은 **1개**",
    "R ≥ 3px에서 6개 모델의 F1은 **0.998~1.000**(포화)",
    "IoU 0.5는 사실상 \"중심이 2px 이내인가\"를 재고 있었다",
    "남는 차이는 위치 정밀도: mAP50-95는 D-FINE-N 0.659 > RT-DETR-l 0.639 > YOLO26n 0.633 > R50 0.611 > MobileNetV3 0.559 ≈ YOLOv3-tiny 0.558 (공식 라벨만 보면 0.335~0.365로 좁아짐)",
], size=15, gap=8)

# ================================================================ 8 일반화·robust 개요
s = new_slide("② 일반화와 robust 평가", "실제 데이터에서는 다 찾으니, 조건을 인위적으로 바꿔 모델을 시험한다")
table(s, 0.6, 1.95, 12.1, [
    ["시험", "묻는 것", "방법"],
    ["점 제거", "모델이 결함의 점 자체를 보는가, 위치나 주변만 보는가", "정답의 점을 보간으로 지운 이미지에 다시 적용"],
    ["가짜 정상", "결함이 없는 이미지에서 오경보를 내는가", "점을 지운 이미지(결함 369장)를 가짜 정상으로 사용"],
    ["합성 점", "옅은 점·다른 호기에서도 찾는가", "작대기 안에만 점을 합성(진하기 4 x 크기 3), 시드 42"],
    ["호기·해상도", "특정 호기나 해상도에서 약한가", "호기·해상도별로 나눠 평가"],
], [1.8, 5.2, 5.1], size=15, row_h=0.8)
bullets(s, 0.6, 6.0, 12.0, 1.2, [
    "합성·가짜 정상은 **인위적인 이미지**라 실제 현장 성능이 아니라 **모델 간 상대 비교와 취약점 탐색**으로 해석한다",
], size=16)

# ================================================================ 9 점 제거
s = new_slide("점 제거: 모델은 점 자체를 본다", "결함의 점을 지우면 검출이 사라진다 → 위치만 보는 지름길이 아니다")
pic(s, "fig3b_dot_removal.png", 0.8, 1.7, 11.7, 3.6)
bullets(s, 0.8, 5.4, 11.8, 1.9, [
    "test 정답 663개의 점을 보간으로 지우고 같은 모델을 다시 적용: 노이즈 없는 평균 보간으로 13x13 이상 지우면 **6개 모델 모두 0~3개**만 검출",
    "YOLOv3-tiny 등에 남는 반응은 보간에 더한 **노이즈 무늬** 때문 → 다음 질문: 노이즈 무늬에 속아 오경보를 내는가?",
], size=16, gap=8)

# ================================================================ 10 가짜 정상
s = new_slide("가짜 정상: 노이즈 보간에는 R50과 YOLO26n만 속지 않는다", "정상 이미지가 없어 결함의 점을 지운 이미지로 오경보를 쟀다", note="가짜 정상은 진짜 정상 제품이 아니다. 오경보율은 범위로 해석한다.")
pic(s, "fig3l_robust6.png", 0.5, 1.8, 12.3, 3.9)
bullets(s, 0.7, 5.6, 12.0, 1.7, [
    "NS+노이즈 6px 오경보율: **R50 0.5%, YOLO26n 4.1%**, RT-DETR-l 37%, YOLOv3-tiny 39%, D-FINE-N 49%, MobileNetV3 64%",
    "오경보는 대부분 **지운 자리**에서 발생, 분리도(AUC)는 MobileNet 0.842·D-FINE 0.900이 가장 나쁨",
], size=15, gap=4)

# ================================================================ 11 합성 점 생성
s = new_slide("합성 점: 작대기 안에만, 시드 42로 모두가 같게", "실제로는 놓침이 사실상 없어서, 미탐지 조건을 직접 만들어 시험했다")
pic(s, "fig3m_synth_bars.png", 0.5, 1.8, 6.7, 5.3)
bullets(s, 7.4, 1.95, 5.5, 5.0, [
    "실제 점을 지우고 영상에서 찾은 **작대기(막대) 안쪽**에만 val의 실제 점(투과율 조각)을 합성",
    "점의 **진하기(4단계) x 크기(3단계)**를 바꿔 옅은 점까지 시험",
    "test 바탕 386장, 점 1,436개. 실제 점 663개가 **전부 검출된 막대에 닿음**",
    "**시드 42 + 이미지 이름별 시드**로 고정, 기준본과 --verify로 팀원 PC에서 같은 데이터인지 확인",
    "합성은 실제보다 약 10%p 덜 현실적 → 절대 수치가 아닌 **상대 비교**",
], size=15)

# ================================================================ 12 합성 점 결과
s = new_slide("합성 점 결과: 옅은 점이 미탐지 조건이다", "val 임계값에서는 D-FINE-N이 앞서지만 대부분 느슨한 임계값 효과, 맞추면 차이가 줄어든다", note="합성 점 시험은 test 이미지가 바탕이며 참고용이다. 합성 점 AP와 오경보 예산 비교는 합성 이미지 자체로 계산한 분석용 수치다.")
pic(s, "fig3k_compare6.png", 0.5, 1.8, 12.3, 3.7)
bullets(s, 0.7, 5.45, 12.0, 1.8, [
    "대비(둘레-점 밝기) 6 미만은 모든 모델이 거의 0%, 20 이상에서 약 77~89%",
    "임계값과 무관한 합성 점 AP: **D-FINE-N 0.574 ≈ R50 0.567** > YOLOv3-tiny 0.515 ≈ YOLO26n 0.501 > RT-DETR-l 0.451 >> **MobileNetV3 0.262**",
    "3호기(대비 14 이상): MobileNetV3 4%, YOLO26n·RT-DETR-l 약 61%, R50·YOLOv3-tiny·D-FINE-N은 호기 차이가 작음",
], size=15, gap=3)

# ================================================================ 13 호기·해상도
s = new_slide("호기·해상도: 뚜렷한 효과는 확인되지 않았다", "가장 낮은 그룹은 공식 라벨 박스가 76%여서 IoU 점수가 낮을 뿐이다")
pic(s, "fig3f_groups_chart.png", 0.6, 1.7, 12.1, 3.7)
bullets(s, 0.8, 5.5, 11.8, 1.8, [
    "해상도별 공식 라벨 박스 비율이 10~76%로 달라 라벨 구성이 섞인다. 공식 라벨 이미지끼리만 비교하면 F1이 0.956~0.977로 좁아지고 구간이 겹친다",
    "그룹별 표본이 작다(공식 라벨 이미지 호기별 20~36장). 합성 점에서 보인 3호기 약점은 **모델에 따라** 다르다(앞 슬라이드)",
], size=16, gap=8)

# ================================================================ 14 종합 성적표
s = new_slide("③ 종합: 모델별 성적표", "한 모델이 모든 항목에서 1등은 아니다", note="val mAP50-95는 val, 나머지는 test 이미지 기준(가짜 정상·합성 점은 인위 이미지). 상세 수치는 docs/experiments.md.")
table(s, 0.5, 1.95, 12.3, [
    ["모델", "test F1\n(IoU 0.5)", "val mAP50-95\n(전체 / 공식만)", "가짜 정상 오경보\n(NS+노이즈 6)", "합성 점 AP", "3호기 합성 점\n(대비 14 이상)", "파라미터"],
    ["D-FINE-N", "0.989", "0.655 / 0.343", "49.3%", "0.574", "89%", "3.7M"],
    ["RT-DETR-l", "0.988", "0.662 / 0.357", "36.6%", "0.451", "61%", "32.8M"],
    ["Faster R-CNN R50", "0.989", "0.615 / 0.337", "0.5%", "0.567", "75%", "41.4M"],
    ["YOLO26n", "0.991", "0.642 / 0.314", "4.1%", "0.501", "62%", "2.5M"],
    ["YOLOv3-tiny", "0.989", "0.579 / 0.311", "38.5%", "0.515", "70%", "8.7M"],
    ["Faster R-CNN MobileNetV3", "0.986", "0.583 / 0.322", "64.0%", "0.262", "4%", "19.0M"],
], [2.6, 1.3, 2.0, 2.0, 1.4, 1.8, 1.2], size=14, row_h=0.62, hi_rows=(1,), bold_rows=(1,))
bullets(s, 0.6, 6.35, 12.0, 0.9, [
    "점 제거는 6개 모델 모두 0~3개만 남아 차이가 없다. test F1은 모두 포화다.",
], size=14)

# ================================================================ 15 베스트 모델
s = new_slide("베스트 모델: D-FINE-N", "val 동률 선두 + 미탐지 강건성 최상 + 작다(3.7M)", note="선정 기준 ①②③은 팀원 모델 추가 전에 정했다. '미탐지 우선'은 결과를 본 뒤 정한 우선순위다. 상세는 docs/analysis.md 5절.")
bullets(s, 0.6, 1.95, 7.6, 5.0, [
    "**① val mAP50-95**: val 이미지 복원추출 1,000회로 비교하니 **RT-DETR-l 0.662와 D-FINE-N 0.655가 동률 선두**(차이 −0.008, 95% 구간 −0.016~0.000). R50(0.615)·YOLO26n(0.642)은 유의하게 낮음. 공식 라벨만 보면 R50도 동률",
    "**② 동률이면 강건성**: 합성 점 AP와 3호기 검출은 D-FINE-N이 우세(0.574 대 0.451, 89% 대 61%), 가짜 정상 오경보는 RT-DETR-l이 우세(37% 대 49%)",
    "→ 과제가 \"AI 미탐지 조건 분석\"이고 놓침이 재검사보다 비싸다고 보아 **미탐지를 우선**했다",
    "**③ 크기**: 3.7M(15MB)로 RT-DETR-l(32.8M)의 약 1/9",
], size=15, gap=10)
bullets(s, 8.5, 1.95, 4.5, 5.0, [
    "**약점**: 노이즈 보간 오경보 49%, 신뢰도가 0.9 근처에서 포화",
    "**대안**: 오경보 우선이면 R50 / YOLO26n, CPU 전용이면 YOLOv3-tiny",
    "MobileNetV3는 위치 의존·노이즈 오경보로 비권장",
    "이 우선순위는 **결과를 본 뒤 정한 것**이라 보고서에 그대로 밝힌다",
], size=15, gap=10)

# ================================================================ 16 현장 활용
s = new_slide("현장 활용: 3단 판정과 재검사 구간", "실제 test에서는 결함 369장(≥0.748)과 빈 라벨 27장(≤0.04)이 완전히 갈린다", note="재검사 구간의 효과는 합성·가짜 정상에 기반한 참고 수치다. 재검사·놓침 비용은 현장 값을 몰라 가정하지 않았다.")
pic(s, "fig4b_final_dfine.png", 0.5, 1.65, 12.3, 3.55)
bullets(s, 0.8, 5.3, 11.8, 1.6, [
    "**불합격** 최고 신뢰도 ≥ 0.66(val 임계값)  |  **재검사** 0.3 ~ 0.66  |  **합격** < 0.3",
    "하한을 0.3으로 낮추면 옅은 점 검출 61.9% → 67.1%, 대신 가짜 정상 오경보 4.1% → 5.7%(평균 보간), 49% → 59%(노이즈 보간)",
    "오경보가 비싼 현장은 재검사 구간의 이득이 작다",
], size=16, gap=6)

# ================================================================ 17 선택 가이드
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

# ================================================================ 18 차별점
s = new_slide("차별점: 점수를 믿게 만드는 검증", "못 찾은 것의 원인을 확인하고, 일반화까지 시험해 모델을 골랐다")
bullets(s, 0.7, 1.95, 12.0, 5.0, [
    "**오류의 원인 규명**: 놓친 것이 아니라 라벨 박스 크기 차이임을 확대 확인하고, **중심 거리 평가**로 재채점",
    "**점 제거 시험**: 결함을 지워 모델이 점 자체를 보는지 확인 → 지름길 우려 해소",
    "**가짜 정상과 합성 점**: 정상 이미지가 없는 데이터에서 오경보·미탐지 조건(옅은 점, 위치, 호기)을 정량화",
    "**임계값과 무관한 비교**: AP·오경보 예산으로 임계값 효과를 걷어냄(D-FINE-N의 우위가 임계값 효과임을 확인)",
    "**공유 가능한 평가 데이터**: 시드 42 합성 데이터 + 기준본 검증으로 팀 모두가 같은 시험을 재현",
    "**분석을 조치로**: 취약점이 모델 선택 가이드와 재검사 구간으로 이어짐",
], size=20, gap=14)

# ================================================================ 19 재현성
s = new_slide("코드 구성과 재현성", "전처리 · 학습 · 추론 · 평가 · 분석을 폴더별로 분리하고 실행 명령을 문서화했다", note="전처리 코드는 이총의 브랜치에 있어 main 병합이 필요하다. 한 번에 실행하는 스크립트와 테스트 예측 결과 CSV는 제출 전에 정리할 항목이다 [작성 필요].")
bullets(s, 0.7, 1.95, 6.1, 5.0, [
    "환경: conda KAMP(Python 3.10), torch 2.6, 한글 경로용 PYTHONUTF8=1, requirements.txt",
    "src/yolov3/: 베이스라인(호환 수정본)",
    "src/minyeop/: Faster R-CNN, MobileNet, YOLOv3-tiny, 팀원 모델 평가, 분석(점 제거, 가짜 정상, 합성 점, 호기 통계)",
    "scripts/synth_eval.py: 시드 42 합성 평가 데이터 생성기",
    "docs/: 지표, 실험 결과 표, 분석 해석",
], size=17, gap=9)
bullets(s, 7.0, 1.95, 5.9, 5.0, [
    "평가 규칙: 임계값은 **val에서 고정**, test는 한 번만",
    "데이터 버전: manifest.csv의 sha256으로 확인",
    "합성 평가 데이터: 기준본과 --verify로 비교(위치·조건·픽셀 해시)",
    "모델 선정: val 부트스트랩 구간으로 \"동률\"을 객관적으로 판정",
    "모든 수치의 출처 파일을 docs/와 runs/에 보존(예측 JSON은 제외)",
], size=17, gap=9)

# ================================================================ 20 한계
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

# ================================================================ 21 결론
s = new_slide("결론", None)
bullets(s, 0.7, 1.5, 12.0, 5.5, [
    "6개 모델을 같은 조건으로 학습·평가했고 F1은 0.99 안팎이었다. 몇 개 못 찾은 것은 **공식 라벨의 박스 크기** 때문이었고 **중심 거리**로 재채점하면 거의 다 찾았다",
    "일반화·robust 평가로 진짜 차이가 드러났다: **옅은 결함, 노이즈 무늬 오경보, MobileNetV3의 위치 의존**",
    "종합해서 **D-FINE-N**을 베스트 모델로 선정: val 동률 선두, 미탐지 강건성 최상, 작다. 약점(노이즈 오경보)은 임계값·재검사 구간과 대안 모델로 관리한다",
    "공유 가능한 합성 평가 데이터와 검증 코드로 **팀 모두가 같은 시험을 재현**한다",
], size=22, gap=20)

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
