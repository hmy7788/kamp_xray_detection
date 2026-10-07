"""분할 파일을 만든다 (docs/contracts/data.md 4절, docs/decisions/002).

라벨링이 끝난 뒤 한 번 돌린다. 2,532장 전부를 묶음(burst) 단위로 학습 / 검증 / 테스트 로 나눈다.
  test  = 최종 테스트. 최종 보고 때 reproduce.py --final 로만 쓴다. 그 전에는 읽지도 않는다 (기본 15%)
  val   = 평소 실험의 채점용 (기본 15%)
  train = 나머지
같은 묶음의 사진은 같은 쪽에만 들어간다 (연속 사진이 거의 같아서). 장비 비율을 출력으로 확인한다.
만든 뒤 sha256 으로 잠근다. 이미 잠긴 버전이 있으면 덮어쓰지 않는다.

사용: python common/make_splits.py [--version v2] [--test-frac 0.15] [--val-frac 0.15] [--seed 42]
      python common/make_splits.py --version smoke --partial   # 라벨 있는 사진만으로 시험 분할 (파이프라인 점검용)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402


def _summary(name: str, g) -> str:
    return f"{name:5s}: {len(g):4d}장, 묶음 {g.burst_id.nunique():3d}, 장비 {g.groupby('machine').size().to_dict()}, 월 {g.groupby('month').size().to_dict()}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partial", action="store_true", help="팀 라벨이 덜 끝나도 라벨 있는 사진만으로 만든다 (시험용)")
    a = ap.parse_args()

    out = kx.SPLITS / f"{a.version}.json"
    lock = kx.SPLITS / f"{a.version}.sha256"
    if lock.exists():
        sys.exit(f"{lock} 가 이미 있다. 분할은 수정하지 않는다. 바꾸려면 새 버전 이름을 써라.")
    if not kx.data_passed():
        sys.exit("data/PASS 가 없다. python common/check_data.py 를 먼저 통과시켜라")

    df = kx.load_manifest()
    has = df[[(kx.DATA / p).exists() for p in df.label_path]].reset_index(drop=True)
    if len(has) < len(df):
        if not a.partial:
            sys.exit(f"라벨이 {len(has)}/{len(df)}장뿐이다. 라벨링이 끝난 뒤 만든다 (시험이면 --partial)")
        print(f"[시험 분할] 라벨 있는 {len(has)}장만 사용")

    rest_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=a.test_frac, random_state=a.seed)
                              .split(has, groups=has["burst_id"]))
    test = has.iloc[test_idx]
    rest = has.iloc[rest_idx].reset_index(drop=True)
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=a.val_frac / (1 - a.test_frac), random_state=a.seed)
                          .split(rest, groups=rest["burst_id"]))
    train, val = rest.iloc[tr_idx], rest.iloc[va_idx]

    split = {
        "version": a.version,
        "group_key": "burst_id",
        "rule": f"burst-level random: test {a.test_frac:.2f}, val {a.val_frac:.2f}, train rest",
        "seed": a.seed,
        "partial": bool(a.partial),
        "manifest_sha256": kx.manifest_sha(),
        "created_at": kx.now_iso(),
        "train": sorted(train["image_id"].tolist()),
        "val": sorted(val["image_id"].tolist()),
        "test": sorted(test["image_id"].tolist()),
    }
    assert not (set(train.burst_id) & set(val.burst_id)) and not (set(rest.burst_id) & set(test.burst_id)), "묶음이 두 쪽에 걸침"
    kx.write_json(out, split)
    lock.write_text(kx.sha256_file(out) + f"  {out.name}\n")
    for name, g in (("train", train), ("val", val), ("test", test)):
        print(_summary(name, g))
    print(f"저장: {out} / 잠금: {lock}")


if __name__ == "__main__":
    main()
