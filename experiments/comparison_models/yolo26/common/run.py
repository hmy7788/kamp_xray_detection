"""실험의 유일한 진입점. 설정 파일 하나를 받아 검사 → (시드별) 학습 → 검증 예측 검사 → 채점 → 기록 → 집계.

사용: python common/run.py configs/<멤버>/<exp_id>.yaml [--dry-run] [--force]

검사 (docs/contracts/experiment.md):
- KAMP_MEMBER 가 설정의 member 와 같다
- 데이터 검사 통과 표시(data/PASS)가 있다
- manifest 잠금 해시가 맞다 (분할은 manifest 의 split 열)
- reason(고른 이유)이 비어 있지 않다
- parent_exp 대비 바뀐 항목이 하나다
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402

META_KEYS = {"exp_id", "parent_exp", "change", "reason", "rubric_items", "member"}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = repr(v)
    return out


def diff_keys(cfg: dict, parent: dict) -> list[str]:
    a = {k: v for k, v in flatten(cfg).items() if k.split(".")[0] not in META_KEYS}
    b = {k: v for k, v in flatten(parent).items() if k.split(".")[0] not in META_KEYS}
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def load_entry(path: Path):
    spec = importlib.util.spec_from_file_location("entry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run_split"):
        raise AttributeError(f"{path} 에 run_split 가 없다")
    return mod


def make_data(name: str, train_ids, val_ids) -> dict:
    """멤버 진입점에 넘기는 데이터 묶음. 사진·라벨 경로는 image_id 별 딕셔너리로 준다."""
    imgs, labs = kx.path_maps(set(train_ids) | set(val_ids))
    return {"name": name, "train_ids": sorted(train_ids), "val_ids": sorted(val_ids), "images": imgs, "labels": labs}


def validate(cfg: dict, cfg_path: Path, force: bool) -> Path:
    m = kx.member()
    if cfg.get("member") != m:
        sys.exit(f"설정의 member({cfg.get('member')})가 KAMP_MEMBER({m})와 다르다")
    for k in ["exp_id", "family", "seeds", "change", "rubric_items", "entry", "model"]:
        if k not in cfg:
            sys.exit(f"설정에 {k} 가 없다")
    if cfg_path.stem != cfg["exp_id"]:
        sys.exit(f"파일 이름({cfg_path.stem})과 exp_id({cfg['exp_id']})가 다르다")
    if not str(cfg.get("reason", "")).strip():
        sys.exit("reason(이 실험을 고른 이유)이 비어 있다. 사람이 적어야 실행된다")
    if not kx.data_passed():
        sys.exit("data/PASS 가 없다. python common/check_data.py 를 먼저 통과시켜라")
    kx.load_split()  # manifest 잠금 해시 검사
    entry = kx.ROOT / cfg["entry"]
    if not entry.exists():
        sys.exit(f"진입점이 없다: {entry}")
    if cfg.get("parent_exp"):
        pp = kx.RUNS / cfg["parent_exp"] / "config.yaml"
        if not pp.exists():
            sys.exit(f"parent_exp {cfg['parent_exp']} 의 config.yaml 이 없다")
        d = diff_keys(cfg, kx.read_yaml(pp))
        if len(d) != 1:
            sys.exit(f"parent 대비 바뀐 항목이 {len(d)}개다 (하나여야 한다): {d}")
    rd = kx.RUNS / cfg["exp_id"]
    if rd.exists() and (rd / "record.json").exists() and not force:
        sys.exit(f"{rd} 가 이미 있다. 새 exp_id 를 써라 (--force 는 재실행)")
    return rd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg_path = Path(a.config)
    cfg = kx.read_yaml(cfg_path)
    rd = validate(cfg, cfg_path, a.force)
    print(f"검사 통과: {cfg['exp_id']} (seeds {cfg['seeds']})")
    if a.dry_run:
        return 0

    rd.mkdir(parents=True, exist_ok=True)
    kx.write_yaml(rd / "config.yaml", cfg)
    split = kx.load_split()
    data = make_data("val", split["train"], split["val"])
    record = {
        "exp_id": cfg["exp_id"], "member": cfg["member"], "family": cfg["family"],
        "manifest_sha256": kx.manifest_sha(),
        "config_sha256": kx.sha256_file(cfg_path), "git_commit": kx.git_commit(),
        "parent_exp": cfg.get("parent_exp"), "change": cfg["change"], "reason": cfg["reason"],
        "rubric_items": cfg["rubric_items"], "env": kx.env_info(),
        "n_train": len(data["train_ids"]), "n_val": len(data["val_ids"]),
        "started_at": kx.now_iso(), "finished_at": None, "runs": {}, "summary": {},
    }
    kx.write_json(rd / "record.json", record)
    entry = load_entry(kx.ROOT / cfg["entry"])

    frames = []
    for seed in cfg["seeds"]:
        seed = int(seed)
        out_dir = rd / "work" / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        status = "ok"
        try:
            csv = entry.run_split(cfg, data, seed, out_dir)
            df = pd.read_csv(csv, dtype={"image_id": str})
            df["seed"] = seed
            frames.append(df)
        except Exception:
            status = "error"
            (rd / f"error_seed{seed}.txt").write_text(traceback.format_exc())
            print(f"[실패] seed {seed}: {rd / f'error_seed{seed}.txt'}")
        record["runs"][str(seed)] = {"status": status, "minutes": round((time.time() - t0) / 60, 2)}
        kx.write_json(rd / "record.json", record)
        print(f"seed {seed} {status} ({record['runs'][str(seed)]['minutes']} 분)")
    if frames:
        out = pd.concat(frames, ignore_index=True)[["image_id", "seed", "cx", "cy", "w", "h", "conf"]]
        out.to_csv(rd / "preds_val.csv", index=False, lineterminator="\n")

    import subprocess

    py = sys.executable
    ok = subprocess.call([py, str(kx.COMMON / "check_preds.py"), str(rd)]) == 0
    if ok:
        ok = subprocess.call([py, str(kx.COMMON / "evaluate.py"), str(rd)]) == 0
    if ok:
        m = kx.read_json(rd / "metrics.json")
        record["summary"] = {"map50_mean": m["overall"]["map50"]["mean"], "map50_std": m["overall"]["map50"]["std"],
                             "box_recall_mean": m["overall"]["box_recall"]["mean"],
                             "img_f1_mean": m["overall"]["img_f1"]["mean"], "threshold": m["threshold"]}
    record["finished_at"] = kx.now_iso()
    kx.write_json(rd / "record.json", record)
    subprocess.call([py, str(kx.COMMON / "aggregate.py")])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
