"""확정 데이터에서 보고서 숫자까지 한 번에 다시 만든다 (평가표 6. 재현성).

사용:
  python reproduce.py                 데이터 검사 → 데이터 요약(README.txt) → 조건 → 환경 기록
  python reproduce.py --runs A B      설정 파일들을 순서대로 실행하고 공유 표를 만든다
  python reproduce.py --final <exp>   최종 후보를 학습+검증 사진 전부로 학습해 테스트(test) 를 한 번 채점한다
  python reproduce.py --final <exp> --existing-weights
                                      다시 학습하지 않고 runs/<exp>/work/seed*/ 의 학습 완료 가중치로 test 를 채점한다
                                      (val 분석에 쓴 가중치와 test 점수를 같은 모델로 묶을 때)

테스트 채점은 이 명령으로만 한다 (KAMP_FINAL=1). 결과는 runs/<exp>/metrics_test.json.
데이터 계보(원본 → 확정본)는 archive/legacy_v1_fill/README.md 에 있다. 다시 만들지 않는다.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "common"))
import kx  # noqa: E402


def sh(*args) -> None:
    print("$", " ".join(str(a) for a in args))
    if subprocess.call([sys.executable, *[str(a) for a in args]], cwd=ROOT) != 0:
        sys.exit("실패")


def prepare() -> None:
    sh("common/check_data.py")
    sh("common/data_summary.py")
    sh("common/compute_conditions.py")
    sh("common/env_info.py", "--lock")


def final(exp_id: str, existing_weights: bool = False) -> None:
    import shutil

    import pandas as pd
    from run import load_entry, make_data

    rd = kx.RUNS / exp_id
    cfg = kx.read_yaml(rd / "config.yaml")
    split = kx.load_split(allow_test=True)
    train_ids = sorted(split["train"]) if existing_weights else sorted(set(split["train"]) | set(split["val"]))
    data = make_data("test", train_ids, sorted(split["test"]))
    entry = load_entry(kx.ROOT / cfg["entry"])
    frames = []
    for seed in cfg["seeds"]:
        out_dir = rd / "work" / f"test_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if existing_weights:
            # 학습 완료 폴더(best.pt + 에폭 수가 찬 results.csv)를 복사해 두면 진입점이 학습을 건너뛰고 예측만 한다
            src = rd / "work" / f"seed{seed}" / "train"
            if not (src / "weights" / "best.pt").exists():
                sys.exit(f"학습 완료 가중치가 없다: {src / 'weights' / 'best.pt'}")
            (out_dir / "train" / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "weights" / "best.pt", out_dir / "train" / "weights" / "best.pt")
            shutil.copy2(src / "results.csv", out_dir / "train" / "results.csv")
        df = pd.read_csv(entry.run_split(cfg, data, int(seed), out_dir), dtype={"image_id": str})
        df["seed"] = int(seed)
        frames.append(df)
    pd.concat(frames)[["image_id", "seed", "cx", "cy", "w", "h", "conf"]].to_csv(
        rd / "preds_test.csv", index=False, lineterminator="\n")
    sh("common/check_preds.py", rd, "--final")
    sh("common/evaluate.py", rd, "--final")
    kx.write_json(rd / "final_mode.json", {
        "mode": "existing_weights" if existing_weights else "retrain_train+val",
        "weights": [str(rd / "work" / (f"seed{s}" if existing_weights else f"test_seed{s}") / "train" / "weights" / "best.pt") for s in cfg["seeds"]],
        "note": "val 분석에 쓴 학습 가중치 그대로 test 채점" if existing_weights else "train+val 전체로 다시 학습한 뒤 test 채점",
        "scored_at": kx.now_iso()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="설정 파일 경로들")
    ap.add_argument("--final", help="테스트를 채점할 exp_id")
    ap.add_argument("--existing-weights", action="store_true", help="--final 에서 다시 학습하지 않고 기존 학습 가중치로 채점")
    a = ap.parse_args()
    if a.final:
        final(a.final, a.existing_weights)
        return
    prepare()
    for c in a.runs or []:
        sh("common/run.py", c)
    sh("common/aggregate.py", "--print")
    # 평가표 산출물 점검은 안내용이다. 실험이 아직 없으면 "미완" 이 나오는 게 정상이라 실패로 치지 않는다
    subprocess.call([sys.executable, "common/check_rubric.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
