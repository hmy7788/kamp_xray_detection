"""Build a deterministic, paired position experiment before model evaluation.

python src/chong/detectability/build_probes.py --config ... --out ...
No test access; no changes to the source images; no detector-driven sampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from core import (Dataset, EXPECTED_MANIFEST, digest, extract_probe, extract_sham,
                  inject, intersects_labels, json_write, png, product_mask)


def ordered(rows, seed):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}/{r.image_id}".encode()).hexdigest())


def panel(array, title, size=180):
    im = Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8)).convert("RGB")
    im.thumbnail((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size+12, size+38), "#20242d")
    canvas.paste(im, ((size+12-im.width)//2, 30+(size-im.height)//2))
    ImageDraw.Draw(canvas).text((6, 7), title, fill="white")
    return canvas


def enlarged(array, factor=6):
    return np.repeat(np.repeat(array, factor, 0), factor, 1)


def montage(probe, sham_patch, before, sham, after, x, y):
    # An enlargement exposes the estimated fill and possible carrier remnants.
    r = 14
    crops = [im[y-r:y+r+1, x-r:x+r+1] for im in [before, sham, after]]
    panels = [panel(enlarged(probe["patch"]), "Donor (native)"),
              panel(enlarged(probe["background"]), "Estimated background"),
              panel(enlarged(probe["signal"]+128), "Residual +128"),
              panel(enlarged(sham_patch), "Sham source"),
              *[panel(enlarged(a), label) for a, label in zip(crops, ["Before", "Sham", "Inserted"])]]
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "#20242d")
    offset = 0
    for p in panels:
        canvas.paste(p, (offset, 0))
        offset += p.width
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    if cfg["split"] not in {"train", "val"}:
        raise ValueError("Only train/val can be synthesized")
    if (a.out / "samples.jsonl").exists():
        raise FileExistsError("Use a new output directory; frozen samples will not be overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset()
    rejection = Counter()
    sources, samples, templates, gallery = [], [], [], []
    all_rows = [r for r in dataset.rows.values() if r.split == cfg["split"]]
    for machine in ["1", "2", "3"]:
        candidates = ordered([r for r in all_rows if r.machine == machine and r.source == "official"], cfg["seed"])
        donors, used_bursts = [], set()
        for row in candidates:
            if len(donors) == cfg["donors_per_machine"]:
                break
            if row.burst_id in used_bursts:
                continue
            im, boxes = dataset.image(row.image_id), dataset.boxes(row.image_id)
            # One template per source burst, first geometrically valid label.
            for j, box in enumerate(boxes):
                try:
                    probe = extract_probe(im, box)
                    sham_signal, sham_center, sham_patch = extract_sham(im, probe, boxes)
                    if probe["bright_mass"] > .25 * probe["dark_mass"]:
                        raise ValueError("Residual has excessive bright component")
                except ValueError as e:
                    rejection[f"donor: {e}"] += 1
                    continue
                tid = f"m{machine}_t{len(donors)+1}"
                path = a.out / "templates" / f"{tid}.npz"
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, signal=probe["signal"], sham=sham_signal)
                item = {"id": tid, "machine": machine, "image_id": row.image_id, "burst_id": row.burst_id,
                        "label_index": j, "file": f"templates/{tid}.npz", "sha256": digest(path),
                        "signal_shape": list(probe["signal"].shape), "peak": probe["peak"],
                        "dark_mass": probe["dark_mass"], "bright_mass": probe["bright_mass"],
                        "sham_abs_mass": float(np.abs(sham_signal).sum()), "sham_center": sham_center,
                        "center_offset": probe["center_offset"]}
                templates.append(item)
                donors.append((row, probe, sham_signal, sham_patch, tid))
                sources.append({"role": "donor", "image_id": row.image_id, "burst_id": row.burst_id,
                                "image_sha256": digest(dataset.images[row.image_id])})
                used_bursts.add(row.burst_id)
                break
        if len(donors) < cfg["donors_per_machine"]:
            raise RuntimeError(f"Insufficient donor bursts for machine {machine}")

        backgrounds = []
        for row in ordered([r for r in all_rows if r.machine == machine], cfg["seed"]+1):
            if len(backgrounds) == cfg["backgrounds_per_machine"]:
                break
            if row.burst_id in used_bursts:
                continue
            im, boxes = dataset.image(row.image_id), dataset.boxes(row.image_id)
            if not boxes:
                rejection["background: empty label/partial product"] += 1
                continue
            try:
                foreground, rect = product_mask(im)
            except ValueError as e:
                rejection[f"background: {e}"] += 1
                continue
            backgrounds.append((row, im, boxes, foreground, rect))
            used_bursts.add(row.burst_id)
            sources.append({"role": "background", "image_id": row.image_id, "burst_id": row.burst_id,
                            "image_sha256": digest(dataset.images[row.image_id])})
        if len(backgrounds) < cfg["backgrounds_per_machine"]:
            raise RuntimeError(f"Insufficient background bursts for machine {machine}")

        for donor_row, probe, sham_signal, sham_patch, tid in donors:
            ph, pw = probe["signal"].shape
            for bi, (row, im, boxes, fg, rect) in enumerate(backgrounds):
                if donor_row.burst_id == row.burst_id:
                    raise AssertionError("Donor/background burst overlap")
                rx, ry, rw, rh = rect
                shown = False
                for gy in range(cfg["grid_size"]):
                    for gx in range(cfg["grid_size"]):
                        x = round(rx + (gx+.5)*rw/cfg["grid_size"])
                        y = round(ry + (gy+.5)*rh/cfg["grid_size"])
                        x0, y0 = x-pw//2, y-ph//2
                        if (x0 < 0 or y0 < 0 or x0+pw > im.shape[1] or y0+ph > im.shape[0]
                                or not fg[y0:y0+ph, x0:x0+pw].all()):
                            rejection["position: outside estimated product"] += 1
                            continue
                        if intersects_labels(x, y, pw, ph, boxes, im.shape):
                            rejection["position: near existing target"] += 1
                            continue
                        try:
                            after, local = inject(im, probe["signal"], x, y)
                            sham, _ = inject(im, sham_signal, x, y)
                        except ValueError as e:
                            rejection[f"position: {e}"] += 1
                            continue
                        sid = f"{tid}_b{bi+1}_r{gy}_c{gx}"
                        # Use the original donor annotation width/height, translated
                        # to the extracted target's peak; do not shrink matching to 3px.
                        bw, bh = probe["annotation_size"]
                        center = [local[0]+probe["center_offset"][0], local[1]+probe["center_offset"][1]]
                        target = [center[0]-bw/2, center[1]-bh/2, center[0]+bw/2, center[1]+bh/2]
                        samples.append({"id": sid, "machine": machine, "background_id": row.image_id,
                                        "background_burst": row.burst_id, "template_id": tid,
                                        "x": x, "y": y, "grid_y": gy, "grid_x": gx,
                                        "support_box": list(local), "target_box": target,
                                        "foreground_rect": list(rect)})
                        if not shown and x >= 14 and y >= 14 and x+14 < im.shape[1] and y+14 < im.shape[0]:
                            gallery_path = f"images/qa_{tid}_b{bi+1}.png"
                            (a.out / "images").mkdir(exist_ok=True)
                            montage(probe, sham_patch, im, sham, after, x, y).save(a.out / gallery_path)
                            png(a.out / f"images/full_{tid}_b{bi+1}.png", after)
                            gallery.append({"sample_id": sid, "path": gallery_path})
                            shown = True
    with open(a.out / "samples.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    record = {"stage": "requires_visual_QA", "config": cfg, "manifest_sha256": EXPECTED_MANIFEST,
              "config_sha256": digest(a.config), "samples_sha256": digest(a.out / "samples.jsonl"),
              "n_samples": len(samples), "n_templates": len(templates), "sources": sources,
              "templates": templates, "exclusions": dict(rejection), "gallery": gallery,
              "limitations": ["display-domain residual, not calibrated X-ray physics",
                              "original labelled targets retained; only inserted location is scored",
                              "source residual extraction may be imperfect; inspect all templates",
                              "positions share source bursts and are not independent observations",
                              "no material identity, physical sensitivity, production FPR, or equipment miss claim"]}
    json_write(a.out / "record.json", record)
    html = ['<!doctype html><meta charset="utf-8"><title>Probe QA</title>',
            '<style>body{background:#15191f;color:#eef;font:16px system-ui;max-width:1400px;margin:30px auto}img{max-width:100%}section{margin:24px 0}p{line-height:1.65}</style>',
            '<h1>가상 시험편 합성 검수</h1><p>왼쪽부터 원래 이물 주변 / 추정 배경 / 이물 신호 / 대조 신호 출처 / 삽입 전 / 대조 처리 / 삽입 후. 확대 영상은 픽셀을 그대로 확대했습니다. 이물 재질이나 실제 크기는 확인되지 않았습니다.</p>',
            f'<p>{len(templates)}개 신호 · {len(samples)}개 위치. 모델 검출 결과를 보기 전에 합성 품질을 확인합니다.</p>']
    for item in gallery:
        html.append(f'<section><p>{item["sample_id"]}</p><img src="{item["path"]}"></section>')
    (a.out / "qa.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps({"output": str(a.out), "samples": len(samples), "exclusions": dict(rejection)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
