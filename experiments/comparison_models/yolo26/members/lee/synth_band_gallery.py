"""띠 안 합성 결과 갤러리: runs/synth_band/gallery/index.html (+ 공유용 zip).

배경 사진마다 (1) 전체 사진에 자리 표시 — 색 = 기준 설정(lee003_gpu640)이 잡은 가장 옅은 진하기,
(2) 자리별 조각 띠: 넣기 전 | 진하기 1.0 | 0.7 | 0.5 | 0.35 | 0.25, 조각 아래 설정별 확신도.
사용: python members/lee/synth_band_gallery.py [--zip]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import kx  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_band import OUT, SCALES, imread, inject  # noqa: E402

G = OUT / "gallery"
F = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 12)
F2 = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 11)
UP, R = 4, 16
BASE = "lee003_gpu640"
SCALE_COLOR = {1.0: (235, 80, 80), 0.7: (240, 150, 60), 0.5: (240, 210, 60), 0.35: (140, 220, 90), 0.25: (60, 200, 120), None: (120, 120, 130)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zip", action="store_true"); a = ap.parse_args()
    if G.exists(): shutil.rmtree(G)
    (G / "marked").mkdir(parents=True); (G / "crops").mkdir()
    for p in ["qa.png", "placement_m1.png", "placement_m2.png", "placement_m3.png"]:
        shutil.copy2(OUT / p, G / p)
    tags = [p.parent.name for p in sorted(OUT.glob("*/summary.json"))]
    summ = {t: kx.read_json(OUT / t / "summary.json") for t in tags}
    preds = {t: {r["id"]: r["after_conf"] for r in kx.read_json(OUT / t / "predictions.json")} for t in tags}
    samples = [json.loads(l) for l in (OUT / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    templates = {p.stem: np.load(p)["signal"].astype(np.float32) for p in (OUT / "templates").glob("*.npz")}
    bgs = kx.read_json(OUT / "backgrounds.json")
    split = kx.load_split(); imgs, _ = kx.path_maps(split["val"])
    base = BASE if BASE in tags else tags[0]

    # 자리별 묶기: (background, x, y, template) → {scale: sample}
    pos = {}
    for s in samples:
        pos.setdefault((s["background_id"], s["x"], s["y"], s["template"]), {})[s["scale"]] = s

    def faintest_hit(bg, x, y, t, tag, thr=0.42):
        hit = [sc for sc in SCALES if preds[tag].get(pos[(bg, x, y, t)][sc]["id"], 0) >= thr]
        return min(hit) if hit else None

    sections = []
    for b in bgs:
        iid = b["image_id"]; im = imread(imgs[iid])
        marked = Image.fromarray(im).resize((im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST).convert("RGB")
        d = ImageDraw.Draw(marked)
        for bx, by, bw, bh in b["band_boxes"]:
            d.rectangle([bx * 2, by * 2, (bx + bw) * 2, (by + bh) * 2], outline=(80, 140, 220), width=1)
        for lx, ly in b["labels_xy"]:
            d.rectangle([lx * 2 - 10, ly * 2 - 10, lx * 2 + 10, ly * 2 + 10], outline=(60, 220, 90), width=2)
        keys = sorted(k for k in pos if k[0] == iid)
        t_first = sorted({k[3] for k in keys})[0]
        strips = []
        for (bg, x, y, t) in keys:
            fh = faintest_hit(bg, x, y, t, base)
            if t == t_first:
                c = SCALE_COLOR[fh]
                d.ellipse([x * 2 - 5, y * 2 - 5, x * 2 + 5, y * 2 + 5], outline=c, width=2)
            row = []
            for sc in [None] + SCALES:
                a2 = im if sc is None else inject(im, templates[t], x, y, sc)
                cr = a2[max(0, y - R):y + R, max(0, x - R):x + R]
                pil = Image.fromarray(cr).resize((cr.shape[1] * UP, cr.shape[0] * UP), Image.NEAREST).convert("RGB")
                dd = ImageDraw.Draw(pil); cx, cy = pil.width // 2, pil.height // 2
                col = (90, 170, 255)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    dd.line([cx + dx * 4, cy + dy * 4, cx + dx * 8, cy + dy * 8], fill=col)
                lines = ["넣기 전" if sc is None else f"진하기 {sc}"]
                if sc is not None:
                    sid = pos[(bg, x, y, t)][sc]["id"]
                    for tag in tags:
                        lines.append(f"{tag.replace('lee001_', '').replace('lee003_', '003_')}: {preds[tag].get(sid, 0):.2f}")
                lab = Image.new("RGB", (pil.width + 4, pil.height + 14 * len(lines) + 6), (28, 32, 40))
                lab.paste(pil, (2, 2))
                if sc is not None:
                    ImageDraw.Draw(lab).rectangle([1, 1, pil.width + 2, pil.height + 2], outline=SCALE_COLOR[sc] if (preds[base].get(sid, 0) >= 0.42) else (90, 90, 100), width=2)
                dl = ImageDraw.Draw(lab)
                for i, ln in enumerate(lines):
                    dl.text((3, pil.height + 4 + 14 * i), ln, fill=(230, 235, 245), font=F2)
                row.append(lab)
            W = sum(p.width for p in row) + 4 * (len(row) + 1) + 150; H = row[0].height
            strip = Image.new("RGB", (W, H), (20, 23, 30))
            ImageDraw.Draw(strip).text((4, 4), f"({x},{y})\n{t}\n띠밝기 {pos[(bg, x, y, t)][1.0]['band_median']:.0f}\n기준이 잡은\n가장 옅은: {fh}", fill=(200, 208, 220), font=F)
            xx = 150
            for p in row:
                strip.paste(p, (xx, 0)); xx += p.width + 4
            strips.append(strip)
        W = max(s.width for s in strips); H = sum(s.height for s in strips)
        sheet = Image.new("RGB", (W, H), (20, 23, 30)); yy = 0
        for s in strips:
            sheet.paste(s, (0, yy)); yy += s.height
        sheet.save(G / "crops" / f"{iid}.png"); marked.save(G / "marked" / f"{iid}.png")
        sections.append((b, len(keys)))

    def pct(v): return f"{v*100:.0f}%"
    H = [f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>띠 안 합성 이물질</title>
<style>body{{background:#15191f;color:#e8ecf3;font:15px/1.6 'Malgun Gothic',system-ui;max-width:1600px;margin:24px auto;padding:0 16px}}
img{{max-width:100%;image-rendering:pixelated}} table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #3a4150;padding:4px 10px;text-align:right}} th{{background:#232a35}} td:first-child,th:first-child{{text-align:left}}
.box{{background:#1d222b;border:1px solid #2e3542;border-radius:8px;padding:12px 16px;margin:14px 0}} details{{margin:10px 0}} summary{{cursor:pointer;font-weight:600}} .small{{color:#9aa4b5;font-size:13px}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:6px;margin-right:4px;vertical-align:middle}}</style>
<h1>검은 띠 안 합성 이물질 — 미탐 조건과 경량화 비교</h1>
<div class="box"><b>만든 방법.</b> 검증(val) 사진 33장(호기별 10~12장, 묶음별 1장)에서 어두운 띠를 찾고, 띠 안을 10px 간격으로 따라가며 자리 412곳을 골랐다(기존 이물질에서 12px 이상 떨어진 곳).
각 자리에 그 호기의 실제 이물질에서 떼어낸 점 신호 2개를, 진하기 1.0 / 0.7 / 0.5 / 0.35 / 0.25 로 <b>한 번에 하나씩</b> 넣었다. 총 4,120장. 넣기 전 사진에서 그 자리에 검출된 것은 0건.</div>
<div class="box"><b>색 뜻 (전체 사진의 동그라미, 조각 테두리).</b> 기준 설정 {base} 이 확신도 0.42 이상으로 잡은 <u>가장 옅은 진하기</u>:
<span class="sw" style="background:rgb(60,200,120)"></span>0.25 까지 잡음 &nbsp; <span class="sw" style="background:rgb(140,220,90)"></span>0.35 &nbsp; <span class="sw" style="background:rgb(240,210,60)"></span>0.5 &nbsp;
<span class="sw" style="background:rgb(240,150,60)"></span>0.7 &nbsp; <span class="sw" style="background:rgb(235,80,80)"></span>1.0 만 잡음 &nbsp; <span class="sw" style="background:rgb(120,120,130)"></span>1.0 도 못 잡음.
초록 네모 = 실제 이물질(라벨), 파란 네모 = 찾은 띠. 조각 아래 숫자 = 설정별 확신도.</div>
<h2>1. 설정별 결과</h2>
<table><tr><th>설정</th><th>입력</th><th>실행</th><th>원본 val F1@0.42</th><th>중심 재현율</th><th>한 장 p50 ms</th><th>p95</th><th>FPS</th>{"".join(f"<th>진하기 {s}</th>" for s in SCALES)}<th>전체</th></tr>"""]
    for t in tags:
        s = summ[t]; v = s["val"][0]; run = f"{s['backend']} {s['device']}" + (f" {s['threads']}스레드" if s.get("threads") else "")
        H.append(f"<tr><td>{t}</td><td>{s['imgsz']}</td><td>{run}</td><td>{v['f1']:.4f}</td><td>{v['center_recall']:.3f}</td><td>{s['speed']['ms_p50']}</td><td>{s['speed']['ms_p95']}</td><td>{s['speed']['fps']}</td>"
                 + "".join(f"<td>{pct(s['hit@0.42']['by_scale'][str(sc)])}</td>" for sc in SCALES) + f"<td>{pct(s['hit@0.42']['overall'])}</td></tr>")
    H.append('</table><p class="small">합성 검출률은 확신도 0.42 기준. 속도는 한 장씩(batch 1) 전처리·추론·후처리 포함. GPU 속도는 같은 GPU 에서 lee_003 학습이 돌던 중 잰 값이라 실제보다 느리다(학습 끝난 뒤 다시 잼).</p>')
    H.append("<h3>호기별 검출률 (0.42) — 진하기별</h3><table><tr><th>설정</th>" + "".join(f"<th>{m}호기 · {sc}</th>" for m in "123" for sc in SCALES) + "</tr>")
    for t in tags:
        H.append(f"<tr><td>{t}</td>" + "".join(f"<td>{pct(summ[t]['hit@0.42']['by_scale_machine'][f'{sc}|{m}'])}</td>" for m in "123" for sc in SCALES) + "</tr>")
    H.append('</table><h2>2. 합성이 어떻게 보이나</h2><p class="small">호기별 4곳, 넣기 전 / 1.0 / 0.5 / 0.25. 가운데 십자가 삽입 자리.</p><img src="qa.png" style="max-width:760px">'
             '<h2>3. 자리 배치 (호기별 예)</h2><p class="small">주황 = 합성 자리, 초록 = 실제 이물질, 파랑 = 띠.</p><img src="placement_m1.png" style="max-width:600px"> <img src="placement_m2.png" style="max-width:600px"><br><img src="placement_m3.png" style="max-width:900px">'
             '<h2>4. 배경 사진별 상세</h2>')
    for mc in "123":
        H.append(f"<h3>{mc}호기</h3>")
        for b, n in sections:
            if b["machine"] != mc: continue
            iid = b["image_id"]
            fhs = [faintest_hit(iid, k[1], k[2], k[3], base) for k in pos if k[0] == iid]
            n_full = sum(1 for f in fhs if f == 1.0); n_none = sum(1 for f in fhs if f is None); n_faint = sum(1 for f in fhs if f is not None and f <= 0.35)
            H.append(f'<details><summary>{iid} · 띠 {b["n_bands"]}개 · 자리 {b["n_positions"]} (신호 2개 = {n}) · 0.35 이하까지 잡음 {n_faint} / 1.0 만 {n_full} / 못 잡음 {n_none}</summary>'
                     f'<img src="marked/{iid}.png"><p class="small">자리별 조각 (넣기 전 | 1.0 | 0.7 | 0.5 | 0.35 | 0.25)</p><img src="crops/{iid}.png"></details>')
    H.append('<h2>5. 파일</h2><ul><li>../samples.jsonl — 4,120개 자리·진하기·신호</li><li>../&lt;설정&gt;/predictions.json, summary.json</li><li>../compare.md — 비교표</li></ul><p class="small">KAMP 데이터에서 만든 사진. 팀 안에서만 공유.</p></html>')
    (G / "index.html").write_text("\n".join(H), encoding="utf-8")
    print("HTML:", G / "index.html")
    if a.zip:
        share = Path(os.environ.get("KAMP_SHARE_DIR", str(kx.ROOT.parents[1] / "share")))  # 하네스 옆 share/ 폴더
        zp = share / "synth_band_2026-10-07.zip"
        if zp.exists(): zp.unlink()
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in G.rglob("*"):
                if f.is_file(): z.write(f, f"synth_band_2026-10-07/gallery/{f.relative_to(G).as_posix()}")
            for name in ["samples.jsonl", "build_info.json", "backgrounds.json", "compare.md"]:
                if (OUT / name).exists(): z.write(OUT / name, f"synth_band_2026-10-07/{name}")
            for t in tags:
                for name in ["summary.json", "predictions.json"]:
                    z.write(OUT / t / name, f"synth_band_2026-10-07/{t}/{name}")
            for f in (OUT / "templates").glob("*.npz"): z.write(f, f"synth_band_2026-10-07/templates/{f.name}")
        print("ZIP:", zp, round(zp.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
