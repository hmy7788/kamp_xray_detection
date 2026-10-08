"""하네스 실험(runs/<exp_id>)을 팀 저장소(hmy7788/kamp_xray_detection) 형식으로 내보낸다.

사용: python members/lee/export_to_team.py lee_003 08_yolo26n_img640 [--team <팀 저장소 경로>]  (기본: 하네스 옆의 KAMP/)

팀 저장소 규칙 (CONTRIBUTING.md, docs/README.md "실험 지표 기록 가이드"):
- runs/chong/<번호>_<모델>_<설정>/ 에 실험 하나. 가중치·work/ 제외. 로컬 경로는 <KAMP>/<HOME> 로 치환.
- preds_<split>.json: [{"name","w","h","gt":[[x1,y1,x2,y2]],"dets":[[x1,y1,x2,y2,score]]}] (픽셀 좌표, 정답 포함이라 Git 제외).
  이 파일로 팀 계산기(src/minyeop/faster_rcnn/report.py)를 돌려 report_<split>.json 과 표 줄을 만든다.
- config.json: 팀원들과 같은 모양 (exp, model, init_weights, data, manifest_sha256, hyp, env, started, finished, train_minutes).
여기에 더해 하네스 산출물(metrics.json: 조건별 재현율, synth_normal.*: 합성 정상 검사)을 그대로 넣는다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import kx  # noqa: E402

TEAM_DEFAULT = kx.ROOT.parents[1] / "KAMP"   # 하네스 저장소 옆의 팀 저장소 복제본 (--team 으로 바꿀 수 있다)
HYP_KEYS = ["imgsz", "epochs", "patience", "batch", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
            "cos_lr", "amp", "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "degrees", "translate", "scale", "mosaic",
            "mixup", "copy_paste", "close_mosaic", "cache", "workers", "seed", "deterministic"]


def sanitize(text: str) -> str:
    for p, tag in ((kx.ROOT, "<KAMP-harness>"), (Path.home(), "<HOME>")):
        for form in {str(p), p.as_posix(), str(p).replace("\\", "\\\\"), str(p).replace("\\", "/")}:
            text = re.sub(re.escape(form), tag, text, flags=re.IGNORECASE)
    return text


def collapse_progress(text: str) -> str:
    """\r 로 덮어쓰던 진행 막대는 줄마다 마지막 상태만 남긴다."""
    out = []
    for line in text.split("\n"):
        line = line.split("\r")[-1] if "\r" in line else line
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)  # 색 코드 제거
        if re.search(r"\d+(\.\d+)?(it/s|s/it)", line) and "100%" not in line:
            continue  # 진행 막대 중간 상태는 버리고 에폭 마지막 줄만 남긴다
        out.append(line)
    return "\n".join(l for l in out if l.strip())


def preds_json(exp_dir: Path, split_ids: list[str], seed: int) -> list[dict]:
    man = kx.load_manifest().set_index("image_id")
    imgs, labs = kx.path_maps(split_ids)
    dets_by = {i: [] for i in split_ids}
    with open(exp_dir / "preds_val.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["seed"]) != seed or r["conf"] in ("", None):
                continue
            w, h = int(man.loc[r["image_id"], "width"]), int(man.loc[r["image_id"], "height"])
            cx, cy, bw, bh = (float(r[k]) for k in ("cx", "cy", "w", "h"))
            dets_by[r["image_id"]].append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h, float(r["conf"])])
    recs = []
    for i in split_ids:
        w, h = int(man.loc[i, "width"]), int(man.loc[i, "height"])
        gt = [[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h] for cx, cy, bw, bh in kx.read_yolo_labels(labs[i])]
        recs.append({"name": f"{i}.png", "w": w, "h": h, "gt": gt, "dets": sorted(dets_by[i], key=lambda d: -d[4])})
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("team_name", help="예: 08_yolo26n_img640")
    ap.add_argument("--team", default=str(TEAM_DEFAULT))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src = kx.RUNS / a.exp_id
    cfg = kx.read_yaml(src / "config.yaml")
    rec = kx.read_json(src / "record.json")
    met = kx.read_json(src / "metrics.json")
    seed = a.seed if a.seed is not None else int(cfg["seeds"][0])
    dst = Path(a.team) / "runs" / "chong" / a.team_name
    if dst.exists() and not a.force:
        sys.exit(f"{dst} 가 이미 있다. 팀 규칙: 이미 있는 이름은 덮어쓰지 않는다 (--force 로 재생성)")
    dst.mkdir(parents=True, exist_ok=True)

    # 1. preds_val.json (팀 계산기 입력, Git 제외)
    split = kx.load_split()
    recs = preds_json(src, split["val"], seed)
    (dst / "preds_val.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")

    # 2. config.json (팀원들과 같은 모양) + 하네스 설정 원본
    work = src / "work" / f"seed{seed}" / "train"
    args = kx.read_yaml(work / "args.yaml") if (work / "args.yaml").exists() else {}
    hyp = {k: args[k] for k in HYP_KEYS if k in args} if args else dict(cfg.get("model", {}))
    train_minutes = None
    if (work / "results.csv").exists():
        rows = list(csv.DictReader(open(work / "results.csv", encoding="utf-8")))
        if rows and "time" in rows[-1]:
            train_minutes = round(float(rows[-1]["time"]) / 60, 1)
        shutil.copy2(work / "results.csv", dst / "results.csv")
    m = cfg.get("model", {})
    config = {
        "exp": a.team_name, "harness_exp_id": cfg["exp_id"], "family": cfg["family"],
        "model": m.get("arch", cfg["family"]),
        "init_weights": ("COCO 사전학습 (ultralytics 배포본 " + re.sub(r"-p2\.yaml$", "", str(m.get("arch", ""))) + ".pt)")
        if cfg["family"] != "blob" else "없음 (학습 없는 LoG 점 검출)",
        "data": "data/ v2 (train 학습, val 임계값·채점). test 미채점",
        "manifest_sha256": rec["manifest_sha256"],
        "hyp": hyp, "seeds": cfg["seeds"], "seed_reported": seed,
        "parent_exp": cfg.get("parent_exp"), "change": cfg["change"], "reason": cfg["reason"],
        "env": rec["env"], "started": rec["started_at"], "finished": rec["finished_at"],
        "train_minutes": train_minutes,
        "harness": "https://github.com/2chong/kamp-xray (common/run.py 로 실행, 코드 사본은 src/chong/harness)",
    }
    (dst / "config.json").write_text(sanitize(json.dumps(config, ensure_ascii=False, indent=2)), encoding="utf-8")
    shutil.copy2(src / "config.yaml", dst / "harness_config.yaml")

    # 3. 하네스 채점·검사 산출물
    shutil.copy2(src / "metrics.json", dst / "harness_metrics.json")
    for f in ("synth_normal.md", "synth_normal.json", "synth_normal_margin.json"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    if (src / "errors" / "false_positives.csv").exists():
        shutil.copy2(src / "errors" / "false_positives.csv", dst / "false_positives.csv")  # 예측 좌표만 (정답 좌표 없음)

    # 4. 로그 (경로 치환, 진행 막대 정리)
    for name in ("train", "score", "synth"):
        p = kx.RUNS / f"{a.exp_id}_{name}.log"
        if p.exists():
            (dst / f"{name}.log").write_text(sanitize(collapse_progress(p.read_text(encoding="utf-8", errors="replace"))), encoding="utf-8")

    o = met["overall"]
    print(f"내보냄: {dst}")
    print(f"  하네스 채점 (val): map50 {o['map50']['mean']:.3f} box_recall {o['box_recall']['mean']:.3f} "
          f"center_hit_recall {o['center_hit_recall']['mean']:.3f} thr {met['threshold']}")
    print("  다음: 팀 계산기로 report_val.json 과 표 줄 만들기 (docs/README.md 가이드 3단계)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
