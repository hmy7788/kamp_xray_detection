"""합성 점 평가 데이터 생성: 작대기(막대) 안에 옅은 점을 합성한 test 이미지를 시드 42 로 똑같이 만든다.

  python scripts/synth_eval.py --split test --out data_synth/test --preview     # 생성 + 미리보기 그림
  python scripts/synth_eval.py --split test --out data_synth/test --verify      # 협업자 PC 에서: 같은 데이터인지 검증

무엇을 만드는가
  1. 원본 이미지의 **실제 결함 점은 평균 보간으로 지운다**(가짜 정상). 합성 점만 정답이 되도록 하기 위해서다.
  2. 영상에서 **작대기(어두운 직사각형)** 를 찾는다: 제품 영역 안에서 주변 평균보다 어두운 곳(적분 영상 기반 지역 임계값)을 잡고, 가늘고 긴 덩어리만 남긴다.
  3. 작대기 **안쪽**(가장자리에서 3px 이상 떨어진 곳)에서만 위치를 뽑아 점을 합성한다. 점 핵(3x3)이 막대 밖으로 나가지 않는다.
  4. 점은 val(test 일 때) 또는 train(val 일 때)의 **실제 점**을 투과율 조각(9x9)으로 떠서 쓴다. 진하기 s 와 크기 f 로 칸을 나눈다.
     X선은 겹친 물체의 투과율이 곱해지므로 바탕 밝기에 T^s 를 곱한다. 같은 호기의 조각만 쓴다.
  5. 정답 라벨은 합성 점마다 박스 하나(1·2호기 10px, 3호기 13px, 팀 라벨 규칙과 같음). 채점은 중심 거리(R=5px 권장)로 한다.

재현성 (모두가 같은 평가 데이터를 쓰도록)
  - 시드 42 고정. 이미지마다 default_rng([42, crc32(이미지 이름)]) 를 따로 만들어, 일부 이미지만 돌려도 그 이미지의 점은 같다.
  - 난수는 정수(integers)만 쓰고 배열 순서는 정렬된 파일명·행 우선으로 고정한다. 막대 검출은 정수 연산(8bit 가우시안, 적분 영상, 형태학) 위주다.
  - 점의 위치·조건은 dots.csv 에, 이미지의 픽셀 해시는 images.sha256 에 남긴다. 기준본은 scripts/synth_eval_reference/ 에 커밋되어 있고
    --verify 가 그것과 비교한다. 위치·조건(dots.csv)이 같으면 같은 문제이고, 픽셀 해시가 다르면 라이브러리 버전 차이(±1 밝기)일 수 있다.
  - 이미지 파일의 해시가 아니라 **픽셀 배열의 해시**를 쓴다(PNG 압축 구현이 달라도 영향이 없음).

원본 data/ 는 바꾸지 않는다. 생성물(data_synth/)은 용량이 커서 Git 에 올리지 않는다(.gitignore).
의존성: numpy, opencv-python, Pillow (requirements.txt). 개인 폴더(src/<이름>/) 코드에 의존하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REF = Path(__file__).resolve().parent / "synth_eval_reference"

SEED = 42
S_LIST = [0.15, 0.3, 0.5, 1.0]          # 깊이 배율 (1.0 = 실제 점의 세기)
F_LIST = [0.67, 1.0, 1.5]               # 크기 배율
F_SIZE = {0.67: 7, 1.0: 9, 1.5: 13}     # 조각 한 변(px)
CELLS = [(s, f) for s in S_LIST for f in F_LIST]
HALF = 4                                # 점은행 조각 9x9
BOX = {"1": 10, "2": 10, "3": 13}       # 합성 점 정답 박스 한 변 (팀 라벨 규칙)
MARGIN = 3                              # 막대 가장자리에서 점 중심까지 최소 거리
GAP_DOTS = 24                           # 합성 점끼리 최소 거리
GAP_REAL = 20                           # 지운 실제 점 자리와의 최소 거리
BAR_WIN = 20                            # 국소 평균 창 반지름 (41x41)
BAR_DELTA = 4                           # 막대 = 국소 평균보다 이만큼 이상 어두운 곳
MIN_BAR_AREA = 120
MIN_BAR_ASPECT = 2.5


# ------------------------------------------------------------------ 입출력
def read_gray(p):
    return np.array(Image.open(p).convert("L"))


def read_boxes(label_path, w, h):
    out = []
    if label_path.exists():
        for ln in label_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                _, cx, cy, bw, bh = (float(v) for v in ln.split()[:5])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def pixel_sha(img):
    return hashlib.sha256(img.tobytes() + str(img.shape).encode()).hexdigest()


# ------------------------------------------------------------------ 실제 점 찾기·지우기
def locate_dot(blur, box, win=5):
    """정답 박스 중심 ±win px 안에서 가장 어두운 점의 (x, y). blur 는 uint8 가우시안(σ=1)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = int(round(cx)), int(round(cy))
    ys, xs = max(y0 - win, 0), max(x0 - win, 0)
    sub = blur[ys:y0 + win + 1, xs:x0 + win + 1]
    j = np.unravel_index(int(np.argmin(sub)), sub.shape)
    return xs + int(j[1]), ys + int(j[0])


def remove_real_dots(img, spots, half=6):
    """점 주변 (2*half+1) 정사각형을 바로 바깥 3px 링의 평균 밝기(정수 반올림)로 채운다."""
    out = img.copy()
    h, w = img.shape
    mask = np.zeros(img.shape, bool)
    for x, y in spots:
        mask[max(y - half, 0):min(y + half + 1, h), max(x - half, 0):min(x + half + 1, w)] = True
    for x, y in spots:
        y0, y1, x0, x1 = max(y - half, 0), min(y + half + 1, h), max(x - half, 0), min(x + half + 1, w)
        Y0, Y1, X0, X1 = max(y - half - 3, 0), min(y + half + 4, h), max(x - half - 3, 0), min(x + half + 4, w)
        ring = np.ones((Y1 - Y0, X1 - X0), bool)
        ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
        ring &= ~mask[Y0:Y1, X0:X1]
        vals = img[Y0:Y1, X0:X1][ring].astype(np.int64)
        out[y0:y1, x0:x1] = int((vals.sum() + len(vals) // 2) // max(len(vals), 1))
    return out


# ------------------------------------------------------------------ 작대기(막대) 찾기
def box_sum(a, r):
    """(2r+1)x(2r+1) 창의 합(정수, 바깥은 0). 적분 영상으로 계산."""
    p = cv2.copyMakeBorder(a, r, r, r, r, cv2.BORDER_CONSTANT, value=0)
    ii = cv2.integral(p, sdepth=cv2.CV_32S)
    k = 2 * r + 1
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]


def find_bars(img):
    """(제품 마스크, 막대 마스크). 막대 = 제품 안에서 주변보다 어두운 가늘고 긴 덩어리. 정수 연산 위주."""
    b3 = cv2.GaussianBlur(img, (0, 0), 3)
    _, th = cv2.threshold(b3, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    if n < 2:
        return None, None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    prod = lab == k
    inner = cv2.erode(prod.astype(np.uint8), np.ones((15, 15), np.uint8))   # 제품 테두리의 그늘 제외 (uint8 0/1)
    if int(inner.sum()) < 500:
        return prod, None
    # 지역 적응형 임계값: 주변(41x41, 제품 안쪽만) 평균보다 BAR_DELTA 이상 어두운 픽셀. 적분 영상(정수)으로 계산해 PC 가 달라도 같다.
    # 제품 전체에 Otsu 를 쓰면 저대비 이미지에서 막대가 안 잡혀(실제 점 663개 중 20% 놓침) 이 방식으로 바꿨다.
    b1 = cv2.GaussianBlur(img, (0, 0), 1.0)
    sv = box_sum(b1 * inner, BAR_WIN).astype(np.int64)
    cnt = box_sum(inner, BAR_WIN).astype(np.int64)
    bar = ((b1.astype(np.int64) * cnt + BAR_DELTA * cnt <= sv) & (inner > 0) & (cnt > 0)).astype(np.uint8)
    bar = cv2.morphologyEx(bar, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bar = cv2.morphologyEx(bar, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(bar)
    keep = np.zeros(img.shape, np.uint8)
    for i in range(1, n2):
        if st2[i, cv2.CC_STAT_AREA] < MIN_BAR_AREA:
            continue
        comp = (lab2 == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (rw, rh), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) < MIN_BAR_ASPECT:
            continue
        keep[lab2 == i] = 1
    return prod, (keep if keep.any() else None)


# ------------------------------------------------------------------ 점 은행·합성
def build_bank(man, split):
    """split 의 실제 점 -> {호기: [투과율 조각 9x9]}. 정렬된 파일 순서."""
    bank = {"1": [], "2": [], "3": []}
    ring_mask = np.maximum(*np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1].__abs__()) == HALF
    for lab in sorted((DATA / split / "labels").glob("*.txt")):
        img = read_gray(DATA / split / "images" / (lab.stem + ".png"))
        h, w = img.shape
        boxes = read_boxes(lab, w, h)
        if not boxes:
            continue
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        for b in boxes:
            x, y = locate_dot(blur, b)
            if x < HALF or y < HALF or x > w - HALF - 1 or y > h - HALF - 1:
                continue
            win = img[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1].astype(np.float64)
            base = max(float(np.median(win[ring_mask])), 1.0)
            T = np.clip(win / base, 0.0, 1.0)
            if T.min() < 0.95:
                bank[man[lab.stem + ".png"]["machine"]].append(T)
    return bank


def resize_bilinear(T, k):
    n = T.shape[0]
    src = np.clip((np.arange(k) + 0.5) * (n / k) - 0.5, 0, n - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w1 = src - i0
    rows = T[i0] * (1 - w1)[:, None] + T[i1] * w1[:, None]
    return rows[:, i0] * (1 - w1)[None, :] + rows[:, i1] * w1[None, :]


def make_patch(T, f, flip, s):
    if flip & 1:
        T = T[:, ::-1]
    if flip & 2:
        T = T[::-1, :]
    if flip & 4:
        T = T.T
    Tr = np.clip(resize_bilinear(np.ascontiguousarray(T), F_SIZE[f]), 1e-3, 1.0)
    return np.exp(s * np.log(Tr))


def measure_contrast(img, x, y):
    """둘레(체비쇼프 거리 6~9) 밝기 중앙값 - 점 주변 3x3 의 가장 어두운 값(σ=1 흐림)."""
    h, w = img.shape
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    yy, xx = np.mgrid[max(y - 9, 0):min(y + 10, h), max(x - 9, 0):min(x + 10, w)]
    d = np.maximum(abs(xx - x), abs(yy - y))
    ring = img[max(y - 9, 0):min(y + 10, h), max(x - 9, 0):min(x + 10, w)][(d >= 6) & (d <= 9)]
    core = blur[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].min()
    return float(np.median(ring)) - float(core)


def place_dots(rng, bar, real_spots, n_dots):
    """막대 안쪽에서 서로·실제 점 자리와 떨어진 위치를 뽑는다. 정수 난수만 사용."""
    inside = cv2.erode(bar, np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8))
    cand = np.argwhere(inside > 0)            # (y, x), 행 우선 정렬
    out = []
    if len(cand) == 0:
        return out
    for _ in range(n_dots):
        for _try in range(300):
            y, x = cand[int(rng.integers(len(cand)))]
            if all(np.hypot(x - px, y - py) >= GAP_DOTS for px, py in out) and all(np.hypot(x - rx, y - ry) >= GAP_REAL for rx, ry in real_spots):
                out.append((int(x), int(y)))
                break
    return out


def generate(split, out_dir, n_dots, seed, only=None):
    man = {r["image_id"] + ".png": r for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8"))}
    bank = build_bank(man, "val" if split == "test" else "train")
    print("점 은행:", {k: len(v) for k, v in bank.items()}, flush=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    dots, imgs, skipped, extra = [], [], [], {}
    names = sorted(p.name for p in (DATA / split / "images").glob("*.png"))
    for name in names:
        if only and name not in only:
            continue
        img0 = read_gray(DATA / split / "images" / name)
        h, w = img0.shape
        m = man[name]["machine"]
        boxes = read_boxes(DATA / split / "labels" / (Path(name).stem + ".txt"), w, h)
        blur = cv2.GaussianBlur(img0, (0, 0), 1.0)
        real = [locate_dot(blur, b) for b in boxes]
        base = remove_real_dots(img0, real) if real else img0
        prod, bar = find_bars(base)
        extra[name] = {"real": real, "real_boxes": boxes, "bar": bar}
        if bar is None:
            skipped.append((name, m, "작대기 없음"))
            continue
        rng = np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])
        sites = place_dots(rng, bar, real, n_dots)
        img = base.astype(np.float64)
        placed = []
        for j, (x, y) in enumerate(sites):
            s, f = CELLS[int(rng.integers(len(CELLS)))]
            bi = int(rng.integers(len(bank[m])))
            flip = int(rng.integers(8))
            T = make_patch(bank[m][bi], f, flip, s)
            k = T.shape[0] // 2
            y0, y1, x0, x1 = y - k, y + k + 1, x - k, x + k + 1
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                continue
            img[y0:y1, x0:x1] *= T
            placed.append((j, x, y, s, f, bi, flip))
        out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        if not placed:
            skipped.append((name, m, "점을 놓을 자리 없음"))
            continue
        Image.fromarray(out, "L").save(out_dir / "images" / name)
        side = BOX[m]
        with open(out_dir / "labels" / (Path(name).stem + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
            for _, x, y, *_r in placed:
                fh.write(f"0 {x / w:.6f} {y / h:.6f} {side / w:.6f} {side / h:.6f}\n")
        for j, x, y, s, f, bi, flip in placed:
            dots.append({"image": name, "machine": m, "width": w, "height": h, "dot_id": j, "x": x, "y": y, "s": s, "f": f,
                         "bank_idx": bi, "flip": flip, "box_side": side, "contrast": round(measure_contrast(out, x, y), 1)})
        imgs.append({"image": name, "machine": m, "width": w, "height": h, "n_dots": len(placed), "removed_real_dots": len(real), "pixel_sha256": pixel_sha(out)})
        extra[name]["placed"] = placed
    return dots, imgs, skipped, extra, man


# ------------------------------------------------------------------ 저장·검증
DOT_COLS = ["image", "machine", "width", "height", "dot_id", "x", "y", "s", "f", "bank_idx", "flip", "box_side", "contrast"]
IMG_COLS = ["image", "machine", "width", "height", "n_dots", "removed_real_dots", "pixel_sha256"]


def write_csv(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def dataset_sha(imgs):
    lines = "\n".join(f"{r['image']} {r['pixel_sha256']}" for r in sorted(imgs, key=lambda r: r["image"]))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def save_outputs(out_dir, dots, imgs, skipped):
    write_csv(out_dir / "dots.csv", dots, DOT_COLS)
    write_csv(out_dir / "images.csv", imgs, IMG_COLS)
    (out_dir / "images.sha256").write_text("".join(f"{r['pixel_sha256']}  {r['image']}\n" for r in sorted(imgs, key=lambda r: r["image"])), encoding="utf-8", newline="\n")
    (out_dir / "dataset.sha256").write_text(dataset_sha(imgs) + "\n", encoding="utf-8", newline="\n")
    write_csv(out_dir / "skipped.csv", [{"image": a, "machine": b, "reason": c} for a, b, c in skipped], ["image", "machine", "reason"])


def verify(out_dir, split):
    ref = REF / split
    if not ref.exists():
        sys.exit(f"기준본이 없습니다: {ref}")
    a = list(csv.DictReader(open(ref / "dots.csv", encoding="utf-8")))
    b = list(csv.DictReader(open(out_dir / "dots.csv", encoding="utf-8")))
    keys = ("image", "dot_id", "x", "y", "s", "f", "bank_idx", "flip", "box_side")
    ka = [tuple(r[k] for k in keys) for r in a]
    kb = [tuple(r[k] for k in keys) for r in b]
    same_dots = ka == kb
    ra = dict(l.split("  ", 1)[::-1] for l in (ref / "images.sha256").read_text(encoding="utf-8").splitlines())
    rb = dict(l.split("  ", 1)[::-1] for l in (out_dir / "images.sha256").read_text(encoding="utf-8").splitlines())
    match = sum(1 for k in ra if rb.get(k) == ra[k])
    print(f"점 위치·조건(dots.csv): {'일치' if same_dots else '불일치'} (기준 {len(ka)}개, 생성 {len(kb)}개)")
    print(f"이미지 픽셀 해시: {match}/{len(ra)}장 일치, 데이터셋 해시 {'일치' if (ref / 'dataset.sha256').read_text().strip() == (out_dir / 'dataset.sha256').read_text().strip() else '불일치'}")
    if not same_dots:
        print("→ 위치·조건이 다르면 다른 평가 데이터입니다. 시드, 입력 데이터(data/manifest.csv의 sha256), numpy/opencv 버전을 확인하세요.")
    elif match != len(ra):
        print("→ 위치·조건은 같고 픽셀만 다릅니다. numpy/opencv/Pillow 버전 차이로 밝기가 ±1 달라졌을 수 있습니다(채점에는 영향이 작음).")
    else:
        print("→ 기준본과 완전히 같은 평가 데이터입니다.")
    return same_dots


# ------------------------------------------------------------------ 미리보기
def get_font(size):
    for f in ("C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


COLOR_S = {0.15: (170, 80, 220), 0.3: (60, 130, 255), 0.5: (255, 170, 30), 1.0: (230, 50, 50)}


def preview(out_dir, split, extra, man, n_per_machine=2, scale=1.4):
    pick = []
    for mc in ("1", "2", "3"):
        got = [n for n in sorted(extra) if "placed" in extra[n] and man[n]["machine"] == mc and len(extra[n]["placed"]) >= 3][:n_per_machine]
        pick += got
    tiles = []
    for n in pick:
        e = extra[n]
        orig = np.array(Image.open(DATA / split / "images" / n).convert("RGB"))
        syn = np.array(Image.open(out_dir / "images" / n).convert("RGB"))
        a = orig.copy()
        for (x1, y1, x2, y2) in e["real_boxes"]:
            cv2.rectangle(a, (int(x1) - 3, int(y1) - 3), (int(x2) + 3, int(y2) + 3), (40, 200, 60), 1)
        b = syn.copy()
        cnts, _ = cv2.findContours(e["bar"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(b, cnts, -1, (0, 220, 220), 1)
        for _j, x, y, s, f, _bi, _fl in e["placed"]:
            cv2.rectangle(b, (x - 9, y - 9), (x + 9, y + 9), COLOR_S[s], 1)
        row = []
        for im in (a, b):
            row.append(Image.fromarray(im).resize((int(im.shape[1] * scale), int(im.shape[0] * scale)), Image.NEAREST))
        tiles.append((n, man[n]["machine"], row))
    pad, cap = 8, 20
    cw = [max(t[2][j].width for t in tiles) for j in range(2)]
    rh = [max(im.height for im in t[2]) for t in tiles]
    canvas = Image.new("RGB", (sum(cw) + pad * 3, 52 + sum(rh) + (cap + pad) * len(tiles)), (30, 30, 30))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 6), f"합성 점 평가 데이터 미리보기 ({split}, 시드 {SEED}): 왼쪽 = 원본(초록 = 실제 점), 오른쪽 = 실제 점을 지우고 작대기(하늘색 윤곽) 안에만 합성한 점", fill=(255, 255, 255), font=get_font(14))
    lx = pad
    for s, c in COLOR_S.items():
        d.rectangle([lx, 30, lx + 14, 44], outline=c, width=2)
        d.text((lx + 20, 29), f"s={s}", fill=(230, 230, 230), font=get_font(12))
        lx += 70
    d.text((lx + 10, 29), "(s = 깊이 배율, 1.0이 실제 세기. 상자는 표시용 18px, 실제 점은 약 3px)", fill=(200, 200, 200), font=get_font(12))
    y = 52
    for (n, mc, row), h in zip(tiles, rh):
        x = pad
        for j, im in enumerate(row):
            d.text((x, y), f"{mc}호기 | {n[:-4][:26]} | " + ("원본" if j == 0 else f"합성 {len(extra[n]['placed'])}개"), fill=(255, 255, 0), font=get_font(12))
            canvas.paste(im, (x, y + cap))
            x += cw[j] + pad
        y += h + cap + pad
    canvas.save(out_dir / "preview_overview.png")

    # 조건 칸마다 점 하나씩 크게 (같은 칸에서 가장 앞선 이미지의 점)
    first = {}
    for n in sorted(extra):
        for j, x, y, s, f, bi, fl in extra[n].get("placed", []):
            first.setdefault((s, f), (n, x, y))
    Wd, S = 40, 7
    cols = 6
    rows_n = (len(CELLS) + cols - 1) // cols
    z = Image.new("RGB", (cols * (Wd * S + 8) + 8, 30 + rows_n * (Wd * S + 46)), (30, 30, 30))
    zd = ImageDraw.Draw(z)
    zd.text((8, 6), f"조건 칸별 합성 점 확대 ({split}, 40x40px 를 7배): s = 깊이 배율, f = 크기 배율, 대비 = 둘레 밝기 - 점 밝기", fill=(255, 255, 255), font=get_font(14))
    for i, cell in enumerate(CELLS):
        if cell not in first:
            continue
        n, x, y = first[cell]
        im = Image.open(out_dir / "images" / n).convert("RGB")
        x1 = min(max(x - Wd // 2, 0), im.width - Wd)
        y1 = min(max(y - Wd // 2, 0), im.height - Wd)
        crop = im.crop((x1, y1, x1 + Wd, y1 + Wd)).resize((Wd * S, Wd * S), Image.NEAREST)
        cx, cy = 8 + (i % cols) * (Wd * S + 8), 30 + (i // cols) * (Wd * S + 46)
        z.paste(crop, (cx, cy + 20))
        c = measure_contrast(np.array(Image.open(out_dir / "images" / n).convert("L")), x, y)
        zd.text((cx, cy), f"s={cell[0]}  f={cell[1]}  대비 {c:.0f}", fill=(255, 255, 0), font=get_font(13))
        zd.text((cx, cy + 20 + Wd * S + 3), f"{man[n]['machine']}호기 {n[:-4][:20]}", fill=(210, 210, 210), font=get_font(11))
    z.save(out_dir / "preview_dots.png")
    return [out_dir / "preview_overview.png", out_dir / "preview_dots.png"]


def qa_report(extra, imgs, skipped, man):
    """막대 검출이 맞는지: 실제 점(막대 끝에 있음)이 검출된 막대에 닿아 있는 비율."""
    tot = hit = 0
    for n, e in extra.items():
        if e["bar"] is None:
            tot += len(e["real"])
            continue
        near = cv2.dilate(e["bar"], np.ones((9, 9), np.uint8))
        for x, y in e["real"]:
            tot += 1
            hit += int(near[y, x] > 0)
    return {"실제 점 수": tot, "막대(9px 확장)에 닿은 실제 점": hit, "비율": round(hit / max(tot, 1), 3)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="작대기 안에 합성 점을 넣은 평가 데이터를 시드 42 로 만든다.")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="data_synth/test")
    ap.add_argument("--dots-per-image", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED, help="협업에서는 42 를 바꾸지 말 것")
    ap.add_argument("--preview", action="store_true", help="미리보기 그림 2장을 만든다")
    ap.add_argument("--verify", action="store_true", help="생성 후 scripts/synth_eval_reference/ 의 기준본과 비교")
    ap.add_argument("--save-reference", action="store_true", help="(기준본 갱신용) dots.csv 등을 scripts/synth_eval_reference/<split>/ 에 복사")
    a = ap.parse_args()
    out_dir = ROOT / a.out
    dots, imgs, skipped, extra, man = generate(a.split, out_dir, a.dots_per_image, a.seed)
    save_outputs(out_dir, dots, imgs, skipped)
    made = {r["image"] for r in imgs}
    for sub in ("images", "labels"):      # 이전 실행(다른 옵션)이 남긴 파일이 평가에 섞이지 않게 지운다
        for f in (out_dir / sub).iterdir():
            if f.stem + ".png" not in made:
                f.unlink()
    qa = qa_report(extra, imgs, skipped, man)
    by_m = {m: sum(1 for r in dots if r["machine"] == m) for m in ("1", "2", "3")}
    by_cell = {f"s{s}_f{f}": sum(1 for r in dots if r["s"] == s and r["f"] == f) for s, f in CELLS}
    summary = {"split": a.split, "seed": a.seed, "dots_per_image": a.dots_per_image, "이미지(생성)": len(imgs), "이미지(제외)": len(skipped), "합성 점 수": len(dots),
               "호기별 점 수": by_m, "칸별 점 수": by_cell, "막대 검출 검증(실제 점 기준)": qa, "dataset_sha256": dataset_sha(imgs)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "칸별 점 수"}, ensure_ascii=False, indent=2))
    print("칸별 점 수:", by_cell)
    if skipped:
        print("제외된 이미지:", len(skipped), "(", {r: sum(1 for x in skipped if x[2] == r) for r in {x[2] for x in skipped}}, ")")
    if a.preview:
        for p in preview(out_dir, a.split, extra, man):
            print("미리보기:", p)
    if a.save_reference:
        import shutil
        dst = REF / a.split
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("dots.csv", "images.sha256", "dataset.sha256", "summary.json"):
            shutil.copy(out_dir / f, dst / f)
        print("기준본 저장:", dst)
    if a.verify:
        verify(out_dir, a.split)


if __name__ == "__main__":
    main()
