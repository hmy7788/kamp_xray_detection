"""실험 진행 상황 보기 (run_experiment.py 가 쓰는 status.json 을 읽는다).

  python scripts/watch_experiment.py                 # 가장 최근 실험
  python scripts/watch_experiment.py 01_yolov3spp_coco
  python scripts/watch_experiment.py --follow        # 10초마다 갱신 (Ctrl+C 로 종료)

status.json 이 STALE_MIN 분 넘게 갱신되지 않았는데 단계가 진행 중이면 "멈춘 것 같음" 경고를 낸다
(PC 절전, 프로세스 종료 등). 이때 프로세스가 살아 있는지도 같이 확인한다.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FINISHED = {"완료", "실패"}


def fmt_time(sec):
    sec = int(max(sec, 0))
    h, m = divmod(sec // 60, 60)
    return f"{h}시간 {m:02d}분" if h else f"{m}분 {sec % 60:02d}초"


def pid_alive(pid):
    if sys.platform == "win32":
        # tasklist 의 한글 메시지(cp949)를 디코딩하지 않고 바이트로 PID 만 찾는다 (PYTHONUTF8=1 에서도 안전)
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True).stdout or b""
        return f" {pid} ".encode() in out
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def latest_run():
    cands = [p for p in RUNS.rglob("status.json") if "_invalid" not in p.parts]  # runs/<이름>/<실험>/status.json
    return max(cands, key=lambda p: p.stat().st_mtime).parent if cands else None


def show(run, stale_min):
    st = json.loads((run / "status.json").read_text(encoding="utf-8"))
    now = datetime.now()
    age = (now - datetime.fromisoformat(st["updated_at"])).total_seconds()
    stage, ep, total = st["stage"], st["epoch"], st["epochs"]
    alive = pid_alive(st["pid"])

    print(f"\n=== {st['name']}  ({now:%H:%M:%S} 기준) ===")
    print(f"단계     : {stage}" + (f"  - {st['message']}" if st.get("message") else ""))
    bar = "#" * int(30 * ep / total) + "-" * (30 - int(30 * ep / total))
    print(f"epoch    : {ep}/{total}  [{bar}] {100 * ep / total:.0f}%")
    print(f"경과     : {fmt_time(st['elapsed_sec'])}" + (f" | 남은 시간 약 {fmt_time(st['eta_sec'])}" if st.get("eta_sec") and stage.startswith("학습") else ""))
    if st.get("last_val"):
        v, b = st["last_val"], st["best_val"]
        print(f"최근 val : P {v['P']:.3f} R {v['R']:.3f} mAP {v['mAP']:.3f} F1 {v['F1']:.3f}  (epoch {v['epoch']})")
        print(f"최고 F1  : {b['F1']:.3f}  (epoch {b['epoch']}, P {b['P']:.3f} R {b['R']:.3f} mAP {b['mAP']:.3f})")
    log = run / "train.log"
    log_age = (now - datetime.fromtimestamp(log.stat().st_mtime)).total_seconds() if log.exists() else None
    print(f"마지막 갱신: {fmt_time(age)} 전 | 프로세스(pid {st['pid']}): {'실행 중' if alive else '없음'}"
          + (f" | train.log 갱신 {fmt_time(log_age)} 전" if log_age is not None else ""))

    if stage not in FINISHED:
        if not alive and log_age is not None and log_age < 120:
            print("⚠ 기록용 프로세스(run_experiment)는 종료됐지만 train.log 가 계속 갱신되고 있어 학습은 진행 중입니다. "
                  "위의 epoch/지표는 오래된 값일 수 있습니다. 학습이 끝나면 evaluate.py 로 평가하세요.")
        elif not alive:
            print("⚠ 프로세스가 종료됐는데 완료/실패로 기록되지 않았습니다. 강제 종료됐을 수 있습니다. train.log 를 확인하세요.")
        elif age > stale_min * 60:
            print(f"⚠ {stale_min}분 넘게 갱신이 없습니다. 멈춘 것 같습니다 (절전/GPU 문제 의심). train.log 갱신 시각: {st.get('log_mtime')}")
    if stage == "완료" and (run / "metrics.json").exists():
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for sp in ("val", "test"):
            if m.get(sp):
                x = m[sp]
                print(f"{sp:5s}: P {x['precision']:.3f} R {x['recall']:.3f} mAP@0.5 {x['mAP@0.5']:.3f} F1 {x['F1']:.3f}")
    print(f"진행 로그 : runs/{st['name']}/progress.log")
    return stage in FINISHED


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="실험 이름 (생략하면 가장 최근 실험)")
    ap.add_argument("--follow", "-f", action="store_true", help="10초마다 갱신")
    ap.add_argument("--stale-min", type=int, default=5, help="이 분 수 넘게 갱신이 없으면 경고 (기본 5)")
    a = ap.parse_args()

    run = RUNS / a.name if a.name else latest_run()
    if not run or not (run / "status.json").exists():
        sys.exit("status.json 이 있는 실험이 없습니다. run_experiment.py 로 실행한 실험인지 확인하세요.")
    try:
        while True:
            done = show(run, a.stale_min)
            if not a.follow or done:
                break
            time.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
