"""학습 진행 상태를 보여준다 (학습 프로세스와 별개로 실행하며, 파일만 읽으므로 학습에 영향이 없다).

  python src/minyeop/faster_rcnn/watch.py --name 01_frcnn_r50fpn_min640            # 한 번 출력
  python src/minyeop/faster_rcnn/watch.py --name 01_frcnn_r50fpn_min640 --follow    # 5초마다 갱신 (Ctrl+C 로 종료)

읽는 파일: runs/<owner>/<name>/ 의 status.json(현재 상태), results.csv(epoch별 지표), progress.log(최근 로그)
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def fmt(sec):
    sec = int(max(sec, 0))
    h, m = divmod(sec // 60, 60)
    return f"{h}시간 {m:02d}분" if h else f"{m}분 {sec % 60:02d}초"


def alive(pid):
    if not pid:
        return None
    try:
        if sys.platform == "win32":
            # tasklist 는 한글 Windows 에서 cp949 로 출력하므로 바이트로 받아 숫자(PID)만 찾는다
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True).stdout
            return f'"{pid}"'.encode() in out
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_json(p):
    try:  # 학습이 파일을 교체하는 순간과 겹치면 읽기에 실패할 수 있어 다음 갱신에 맡긴다
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def bar(frac, width=30):
    n = int(max(0.0, min(frac, 1.0)) * width)
    return "█" * n + "░" * (width - n)


def render(run, tail):
    st = load_json(run / "status.json") or {}
    cfg = load_json(run / "config.json") or {}
    rows = []
    try:
        with open(run / "results.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        pass

    ep, eps = st.get("epoch", 0), st.get("epochs") or cfg.get("epochs") or 0
    it, its = st.get("iter"), st.get("iters")
    # 진행률: 끝낸 epoch + 현재 epoch 안의 배치 진행
    done = len(rows)
    frac = (done + ((it or 0) / its if its and ep > done else 0)) / eps if eps else 0
    pid, run_alive = st.get("pid"), alive(st.get("pid"))
    stage = st.get("stage", "?")
    if stage == "학습" and run_alive is False:
        stage = "중단됨(프로세스 없음)"

    L = [f"실험  : {run.parent.name}/{run.name}", f"상태  : {stage}"
         + (f" (pid {pid} {'실행 중' if run_alive else '없음'})" if pid else "")]
    if st.get("message"):
        L.append(f"오류  : {st['message']}")
    L.append(f"진행  : {bar(frac)} {frac * 100:5.1f}%   epoch {min(ep, eps)}/{eps}"
             + (f"   (현재 epoch 배치 {it}/{its})" if its and stage == "학습" else ""))
    L.append(f"경과  : {fmt(st.get('elapsed_sec', 0))}" + (f"   남은 시간 약 {fmt(st['eta_sec'])}" if st.get("eta_sec") is not None and stage == "학습" else ""))
    if st.get("updated_at"):
        L.append(f"갱신  : {st['updated_at']}")
    if rows:
        best = max(rows, key=lambda r: float(r["val_AP50"]))
        L.append(f"최고  : val AP@0.5 {float(best['val_AP50']):.3f} (epoch {best['epoch']})")
        L += ["", "epoch  loss    AP@0.5  임계값  P      R      F1     TP/FP/FN       추론ms"]
        for r in rows[-12:]:
            mark = " ←최고" if r is best else ""
            L.append(f"{int(r['epoch']):>5}  {float(r['train_loss']):<6.3f}  {float(r['val_AP50']):<6.3f}  {float(r['val_thr']):<6.2f}  "
                     f"{float(r['val_P']):<5.3f}  {float(r['val_R']):<5.3f}  {float(r['val_F1']):<5.3f}  "
                     f"{r['val_TP']}/{r['val_FP']}/{r['val_FN']}".ljust(60)[:60] + f"  {float(r['infer_ms']):.0f}{mark}")
        if len(rows) > 12:
            L.append(f"  (앞 {len(rows) - 12}개 epoch 생략, 전체는 results.csv)")
    else:
        L.append("(첫 epoch 가 끝나면 지표가 표시됩니다)")
    try:
        lines = (run / "progress.log").read_text(encoding="utf-8").splitlines()
        L += ["", f"최근 로그 {min(tail, len(lines))}줄:"] + ["  " + s[:150] for s in lines[-tail:]]
    except OSError:
        pass
    return "\n".join(L)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="실험 이름 (생략하면 가장 최근 실험)")
    ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--follow", action="store_true", help="주기적으로 갱신")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--tail", type=int, default=3, help="보여줄 최근 로그 줄 수")
    a = ap.parse_args()

    base = ROOT / "runs" / a.owner
    if a.name:
        run = base / a.name
    else:
        cands = [p for p in base.glob("*") if (p / "config.json").exists()]
        if not cands:
            sys.exit(f"실험 폴더가 없습니다: {base}")
        run = max(cands, key=lambda p: (p / "config.json").stat().st_mtime)
    if not run.exists():
        sys.exit(f"실험 폴더가 없습니다: {run}")

    try:
        while True:
            text = render(run, a.tail)
            if a.follow:
                os.system("cls" if os.name == "nt" else "clear")
            print(text, flush=True)
            if not a.follow:
                break
            if (load_json(run / "status.json") or {}).get("stage") in ("완료", "실패"):
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
