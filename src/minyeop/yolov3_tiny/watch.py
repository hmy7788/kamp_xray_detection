"""YOLOv3-tiny 학습 진행 상태를 보여준다 (파일만 읽으므로 학습에 영향이 없다).

  python src/minyeop/yolov3_tiny/watch.py --name 02_yolov3tiny_img640_v1             # 한 번 출력
  python src/minyeop/yolov3_tiny/watch.py --name 02_yolov3tiny_img640_v1 --follow     # 5초마다 갱신

읽는 파일: runs/<owner>/<name>/ 의 config.json(설정), results.txt(epoch별 지표), train.log(최근 로그)
results.txt 열: epoch, gpu_mem, GIoU, obj, cls, total, targets, img_size, P, R, mAP@0.5, F1, val GIoU, val obj, val cls
(P/R/F1 은 test.py 의 신뢰도 0.001 기준 값이라 참고용. 최종 지표는 predict.py + report.py 로 낸다)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
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
        if sys.platform == "win32":  # 한글 Windows 의 tasklist 는 cp949 출력이라 바이트로 받아 PID 만 찾는다
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"], capture_output=True).stdout
            return f'"{pid}"'.encode() in out
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_results(run):
    rows = []
    try:
        for line in (run / "results.txt").read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) >= 15 and "/" in t[0]:
                ep, total = t[0].split("/")
                v = [float(x.rstrip("G")) for x in t[1:15]]  # gpu_mem 열은 0.499G 처럼 단위가 붙는다
                rows.append({"epoch": int(ep) + 1, "epochs": int(total) + 1, "loss": v[4], "P": v[7], "R": v[8], "mAP": v[9], "F1": v[10]})
    except OSError:
        pass
    return rows


def render(run, tail):
    cfg = {}
    try:
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    rows = read_results(run)
    eps = cfg.get("epochs") or (rows[-1]["epochs"] if rows else 0)
    done = len(rows)
    frac = done / eps if eps else 0
    log = run / "train.log"
    started = datetime.fromisoformat(cfg["started_at"]) if cfg.get("started_at") else None
    last_touch = max([p.stat().st_mtime for p in (run / "results.txt", log) if p.exists()], default=None)
    pid_alive = alive(cfg.get("pid"))
    finished = done >= eps > 0
    # run.py(pid) 가 살아 있으면 학습 중, 아니면 종료. 종료됐는데 epoch 가 모자라면 중단된 것
    stage = "완료" if finished else ("학습 중" if pid_alive else "중단됨(프로세스 없음)")
    elapsed = (last_touch - started.timestamp()) if (started and last_touch and not pid_alive) else ((time.time() - started.timestamp()) if started else 0)

    L = [f"실험  : {run.parent.name}/{run.name}", f"상태  : {stage}" + (f" (pid {cfg['pid']} {'실행 중' if pid_alive else '없음'})" if cfg.get("pid") else ""),
         f"진행  : {'█' * int(frac * 30)}{'░' * (30 - int(frac * 30))} {frac * 100:5.1f}%   epoch {done}/{eps}"]
    eta = elapsed / done * (eps - done) if done and stage == "학습 중" else None
    L.append(f"경과  : {fmt(elapsed)}" + (f"   남은 시간 약 {fmt(eta)}" if eta is not None else ""))
    if rows:
        best = max(rows, key=lambda r: 0.99 * r["mAP"] + 0.01 * r["R"])  # train.py 가 best.pt 를 고르는 기준(fitness)
        L.append(f"최고  : val mAP@0.5 {best['mAP']:.3f} (epoch {best['epoch']}, best.pt 선택 기준: 0.99·mAP+0.01·R)")
        L += ["", "epoch  loss    mAP@0.5  P      R      F1   (val, 신뢰도 0.001 기준)"]
        for r in rows[-12:]:
            L.append(f"{r['epoch']:>5}  {r['loss']:<6.3f}  {r['mAP']:<7.3f}  {r['P']:<5.3f}  {r['R']:<5.3f}  {r['F1']:<5.3f}" + (" ←최고" if r is best else ""))
        if len(rows) > 12:
            L.append(f"  (앞 {len(rows) - 12}개 epoch 생략, 전체는 results.txt)")
    else:
        L.append("(첫 epoch 가 끝나면 지표가 표시됩니다)")
    try:  # 진행 막대는 \r 로 갱신되므로 마지막 줄만 의미가 있다
        txt = log.read_text(encoding="utf-8", errors="replace")
        lines = [s for s in re.split(r"[\r\n]+", txt) if s.strip()]
        L += ["", f"최근 로그 {min(tail, len(lines))}줄:"] + ["  " + s[:150] for s in lines[-tail:]]
        err = [s for s in lines if "Traceback" in s or "Error" in s]
        if err and stage != "완료":
            L.append("  ※ 오류가 기록되어 있습니다: " + err[-1][:120])
    except OSError:
        pass
    return "\n".join(L), stage


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name"); ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--follow", action="store_true"); ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--tail", type=int, default=2)
    a = ap.parse_args()
    base = ROOT / "runs" / a.owner
    run = base / a.name if a.name else max((p for p in base.glob("*") if (p / "kamp.data").exists()), key=lambda p: p.stat().st_mtime)
    try:
        while True:
            text, stage = render(run, a.tail)
            if a.follow:
                os.system("cls" if os.name == "nt" else "clear")
            print(text, flush=True)
            if not a.follow or stage.startswith(("완료", "중단")):
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
