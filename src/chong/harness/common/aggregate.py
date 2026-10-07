"""모든 실험 기록을 모아 reports/experiments.csv 를 다시 만든다. 손으로 고치지 않는다.

검사(check_record)를 통과한 실험만 들어간다. 조건별 재현율은 rec_<축>_<구간> 열로 펼친다.
사용: python common/aggregate.py [--print]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402
from check_record import check  # noqa: E402

BASE_COLS = ["exp_id", "member", "family", "parent_exp", "change", "reason",
             "rubric_items", "n_train", "n_val", "n_seeds", "threshold",
             "map50_mean", "map50_std", "map50_95_mean", "box_recall_mean", "box_precision_mean", "box_f1_mean",
             "center_hit_recall_mean", "img_f1_mean", "safety_thr", "safety_recheck", "minutes", "finished_at", "git_commit"]


def main() -> None:
    rows, skipped = [], []
    for rd in sorted(kx.RUNS.glob("*/")):
        if rd.name.startswith("_") or not (rd / "record.json").exists():
            continue
        errs = check(rd)
        if errs:
            skipped.append((rd.name, errs[0]))
            continue
        r = kx.read_json(rd / "record.json")
        m = kx.read_json(rd / "metrics.json")
        o = m["overall"]
        row = {
            "exp_id": r["exp_id"], "member": r["member"], "family": r["family"],
            "parent_exp": r.get("parent_exp") or "", "change": r["change"], "reason": r["reason"],
            "rubric_items": ",".join(str(x) for x in r["rubric_items"]),
            "n_train": r.get("n_train"), "n_val": r.get("n_val"), "n_seeds": m.get("n_seeds", 1), "threshold": m["threshold"],
            "map50_mean": o["map50"]["mean"], "map50_std": o["map50"]["std"],
            "map50_95_mean": o["map50_95"]["mean"],
            "box_recall_mean": o["box_recall"]["mean"], "box_precision_mean": o["box_precision"]["mean"],
            "box_f1_mean": o["box_f1"]["mean"], "center_hit_recall_mean": o["center_hit_recall"]["mean"],
            "img_f1_mean": o["img_f1"]["mean"],
            "safety_thr": m["safety"].get("threshold"), "safety_recheck": m["safety"].get("recheck_rate"),
            "minutes": round(sum(v.get("minutes", 0) for v in r["runs"].values()), 1),
            "finished_at": r["finished_at"], "git_commit": r.get("git_commit", ""),
        }
        for axis, bins in m.get("by_condition", {}).items():
            for b, v in bins.items():
                row[f"rec_{axis}_{b}"] = round(v["box_recall"], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        extra = [c for c in df.columns if c not in BASE_COLS]
        df = df[BASE_COLS + sorted(extra)].sort_values("finished_at")
        for c in df.columns:
            if df[c].dtype == float:
                df[c] = df[c].round(4)
    kx.REPORTS.mkdir(exist_ok=True)
    out = kx.REPORTS / "experiments.csv"
    df.to_csv(out, index=False, lineterminator="\n")
    print(f"공유 표: {out} ({len(df)}줄)")
    for name, why in skipped:
        print(f"  제외: {name} — {why}")
    if "--print" in sys.argv and len(df):
        cols = ["exp_id", "member", "change", "map50_mean", "map50_std", "box_recall_mean", "center_hit_recall_mean"]
        print(df[cols].tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
