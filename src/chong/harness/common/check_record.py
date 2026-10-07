"""실험 기록(record.json) 검사. 집계는 통과한 실험만 표에 넣는다.

사용: python common/check_record.py runs/<exp_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kx  # noqa: E402

REQUIRED = ["exp_id", "member", "family", "manifest_sha256",
            "config_sha256", "change", "reason", "rubric_items", "env", "started_at", "finished_at", "runs", "summary"]


def check(rd: Path) -> list[str]:
    errs = []
    rp = rd / "record.json"
    if not rp.exists():
        return ["record.json 없음"]
    r = kx.read_json(rp)
    for k in REQUIRED:
        if k not in r:
            errs.append(f"record.json: {k} 없음")
    if r.get("exp_id") != rd.name:
        errs.append("exp_id 가 폴더 이름과 다름")
    if not str(r.get("reason", "")).strip():
        errs.append("reason(고른 이유)이 비어 있음")
    if not str(r.get("change", "")).strip():
        errs.append("change(변경점)가 비어 있음")
    if not r.get("rubric_items"):
        errs.append("rubric_items 가 비어 있음")
    if not (rd / "metrics.json").exists():
        errs.append("metrics.json 없음 (채점 안 됨)")
    if r.get("manifest_sha256") != kx.manifest_sha():
        errs.append("manifest_sha256 가 현재 데이터 잠금 해시와 다름 (다른 데이터로 돌린 실험)")
    bad = [k for k, v in r.get("runs", {}).items() if v.get("status") != "ok"]
    if bad:
        errs.append(f"실패한 시드: {bad}")
    return errs


def main() -> int:
    rd = Path(sys.argv[1])
    errs = check(rd)
    if errs:
        print("FAIL:")
        for e in errs:
            print("  -", e)
        return 1
    print(f"PASS: {rd.name} 기록")
    return 0


if __name__ == "__main__":
    sys.exit(main())
