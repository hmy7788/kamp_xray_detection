"""실험 1건 실행: 학습 -> val/test 평가 -> runs/<name>/ 에 결과 저장.

예)
  python scripts/run_experiment.py --name 01_yolov3spp_coco --cfg yolov3-spp.cfg \
      --weights weights/pretrained/yolov3-spp-ultralytics.pt --epochs 100 --batch-size 3

진행 상황 (백그라운드로 돌려도 확인 가능)
  - 콘솔과 runs/<name>/progress.log 에 단계와 epoch 마다 한 줄씩 출력한다.
  - runs/<name>/status.json 은 10초마다 갱신된다 (단계, epoch, 경과/남은 시간, 최근 val 지표).
  - 요약 보기:  python scripts/watch_experiment.py [<name>] [--follow]

산출물(runs/<name>/): weights/{last,best}.pt, results.txt, results.png,
  train.log, val_eval.log, test_eval.log, metrics.json, config.json, progress.log, status.json
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO = ROOT / "src" / "yolov3"
POLL_SEC = 10  # 진행 상황 갱신 주기


def keep_awake(on):
    """실험이 도는 동안 Windows 가 절전/대기로 들어가지 않게 한다(시스템 설정은 바꾸지 않음)."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def fmt_time(sec):
    sec = int(max(sec, 0))
    h, m = divmod(sec // 60, 60)
    return f"{h}시간 {m:02d}분" if h else f"{m}분 {sec % 60:02d}초"


class Progress:
    """콘솔 + progress.log 에 한 줄씩 기록하고 status.json 을 원자적으로 갱신한다."""

    def __init__(self, out, name, epochs):
        self.out, self.name, self.epochs = out, name, epochs
        self.t0 = time.time()
        self.state = {"name": name, "pid": os.getpid(), "stage": "시작", "epoch": 0, "epochs": epochs,
                      "started_at": datetime.now().isoformat(timespec="seconds"), "last_val": None,
                      "best_val": None, "message": ""}
        self.log = open(out / "progress.log", "a", encoding="utf-8")

    def say(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n")
        self.log.flush()

    def update(self, **kw):
        self.state.update(kw)
        self.state["elapsed_sec"] = int(time.time() - self.t0)
        self.state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        # 상태 파일은 보조 정보다. OneDrive 동기화나 다른 프로세스가 파일을 잡고 있어 교체가 거부(WinError 5)돼도
        # 실험이 죽으면 안 되므로 재시도하고, 끝내 실패하면 이번 갱신만 건너뛴다.
        text = json.dumps(self.state, ensure_ascii=False, indent=2)
        tmp = self.out / "status.json.tmp"
        for _ in range(5):
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, self.out / "status.json")
                return
            except OSError:
                time.sleep(0.5)
        try:
            (self.out / "status.json").write_text(text, encoding="utf-8")  # 교체가 안 되면 직접 덮어쓰기
        except OSError:
            pass


def read_epochs(results_txt):
    """results.txt 의 epoch 별 행을 읽어 [{epoch, P, R, mAP, F1}] 로 반환한다."""
    rows = []
    if not results_txt.exists():
        return rows
    for i, line in enumerate(results_txt.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        c = line.split()
        if len(c) >= 12:
            try:
                rows.append({"epoch": i, "P": float(c[8]), "R": float(c[9]), "mAP": float(c[10]), "F1": float(c[11])})
            except ValueError:
                pass
    return rows


def run_train(cmd, out, env, pg, epochs):
    """학습 프로세스를 띄우고 POLL_SEC 마다 results.txt 를 읽어 epoch 진행을 출력한다."""
    log_path = out / "train.log"
    seen, t_start = 0, time.time()
    with open(log_path, "wb") as f:
        p = subprocess.Popen(cmd, cwd=YOLO, env=env, stdout=f, stderr=subprocess.STDOUT)
        while True:
            alive = p.poll() is None
            rows = read_epochs(out / "results.txt")
            if rows:
                best = max(rows, key=lambda r: r["F1"])
                last = rows[-1]
                pg.update(stage="학습", epoch=len(rows), last_val=last, best_val=best,
                          log_mtime=datetime.fromtimestamp(log_path.stat().st_mtime).isoformat(timespec="seconds"),
                          eta_sec=int((time.time() - t_start) / len(rows) * (epochs - len(rows))))
            else:
                pg.update(stage="학습 준비(모델 로드/캐시)")
            if len(rows) > seen:
                for r in rows[seen:]:
                    el = time.time() - t_start
                    eta = el / len(rows) * (epochs - len(rows))
                    best = max(rows[:r["epoch"]], key=lambda x: x["F1"])
                    pg.say(f"epoch {r['epoch']:>3}/{epochs}  val P {r['P']:.3f} R {r['R']:.3f} "
                           f"mAP {r['mAP']:.3f} F1 {r['F1']:.3f} | 최고 F1 {best['F1']:.3f}(epoch {best['epoch']}) "
                           f"| 경과 {fmt_time(el)} | 남은 시간 약 {fmt_time(eta)}")
                seen = len(rows)
            if not alive:
                break
            time.sleep(POLL_SEC)
        return p.returncode


def run_eval(cmd, log, env):
    with open(log, "wb") as f:
        return subprocess.run(cmd, cwd=YOLO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def parse_metrics(log):
    """test.py 로그의 'all  N  targets  P  R  mAP  F1' 행을 파싱한다."""
    text = Path(log).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    m = None
    for m in re.finditer(r"\ball\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", text):
        pass
    if not m:
        return None
    n_img, n_tgt, p, r, ap, f1 = m.groups()
    return {"images": int(n_img), "targets": int(n_tgt), "precision": float(p), "recall": float(r),
            "mAP@0.5": float(ap), "F1": float(f1)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # 리다이렉트 시 cp949 로 깨지는 것을 방지
    keep_awake(True)
    try:
        _main()
    finally:
        keep_awake(False)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="yolov3-spp.cfg")
    ap.add_argument("--weights", required=True, help="KAMP 루트 기준 상대경로 또는 절대경로")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, nargs="+", default=[320, 640])
    ap.add_argument("--dataset", default="n500", choices=["n500", "nomark"],
                    help="n500: 장비 표시(색 박스)가 있는 원본, nomark: 표시를 지운 데이터(data/nomark)")
    ap.add_argument("--device", default="")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="train.py 에 그대로 전달할 추가 인자")
    a = ap.parse_args()

    out = ROOT / "runs" / a.name
    if out.exists():
        sys.exit(f"이미 존재하는 실험 폴더입니다(덮어쓰기 방지): runs/{a.name}")
    (out / "weights").mkdir(parents=True)
    weights = Path(a.weights) if Path(a.weights).is_absolute() else ROOT / a.weights
    env = {**os.environ, "PYTHONUTF8": "1", "YOLO_SAVE_DIR": str(out)}
    dev = ["--device", a.device] if a.device else []
    (out / "config.json").write_text(json.dumps({**vars(a), "weights_abs": str(weights)}, ensure_ascii=False, indent=2), encoding="utf-8")

    pg = Progress(out, a.name, a.epochs)
    try:
        pg.say(f"실험 시작: {a.name} | 데이터 {a.dataset} | cfg {a.cfg} | batch {a.batch_size} | epochs {a.epochs} | img-size {a.img_size}")
        pg.say("진행 확인: python scripts/watch_experiment.py " + a.name)

        pg.update(stage="분할 생성")
        pg.say("[1/4] train/val/test 분할 생성")
        if a.dataset == "nomark":
            splits = ROOT / "data" / "splits_nomark"
            split_args = ["--src", "data/nomark", "--out", "data/splits_nomark"]
        else:
            splits = ROOT / "data" / "splits"
            split_args = []
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_split.py"), *split_args], check=True, capture_output=True)

        pg.say(f"[2/4] 학습 시작 (최대 {a.epochs} epoch, val 지표는 epoch 마다 출력)")
        t0 = time.time()
        rc = run_train([sys.executable, "-u", "train.py", "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                        "--weights", str(weights), "--cfg", a.cfg, "--data", str(splits / "val.data"),
                        "--img-size", *map(str, a.img_size), *dev, *a.extra], out, env, pg, a.epochs)
        hours = (time.time() - t0) / 3600
        if rc != 0:
            sys.exit(f"학습 실패(rc={rc}). runs/{a.name}/train.log 확인")
        if (YOLO / "results.png").exists():
            shutil.move(str(YOLO / "results.png"), out / "results.png")
        for f in YOLO.glob("test_batch*.jpg"):
            f.unlink()

        n_done = len(read_epochs(out / "results.txt"))
        if n_done < a.epochs:  # 오류 없이 중간에 끝난 경우도 걸러낸다
            sys.exit(f"학습이 {n_done}/{a.epochs} epoch에서 끝났습니다(불완전). runs/{a.name}/train.log 확인")
        pg.say(f"학습 완료: {n_done} epoch, {fmt_time(hours * 3600)}")

        ckpt = out / "weights" / ("best.pt" if (out / "weights" / "best.pt").exists() else "last.pt")
        metrics = {"checkpoint": ckpt.name, "train_hours": round(hours, 3), "eval_img_size": a.img_size[-1]}
        for i, split in enumerate(("val", "test"), 3):
            pg.update(stage=f"평가({split})")
            pg.say(f"[{i}/4] {split} 평가 ({ckpt.name})")
            # 평가 해상도는 학습 중 검증과 같게(= --img-size 의 마지막 값) 맞춘다. test.py 기본값(512)을 쓰면
            # best.pt 를 고른 조건과 달라져 지표가 낮게 나온다.
            rc = run_eval([sys.executable, "-u", "test.py", "--cfg", a.cfg, "--batch-size", str(a.batch_size),
                           "--img-size", str(a.img_size[-1]),
                           "--data", str(splits / f"{split}.data"), "--weights", str(ckpt), *dev], out / f"{split}_eval.log", env)
            metrics[split] = parse_metrics(out / f"{split}_eval.log")
            m = metrics[split]
            pg.say(f"  {split}: " + (f"P {m['precision']:.3f} R {m['recall']:.3f} mAP@0.5 {m['mAP@0.5']:.3f} F1 {m['F1']:.3f}"
                                    if m else f"지표를 읽지 못함 (rc={rc}, {split}_eval.log 확인)"))
        for f in YOLO.glob("test_batch*.jpg"):
            f.unlink()
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        pg.update(stage="완료", message="metrics.json 저장됨")
        pg.say("완료: 결과는 runs/" + a.name + "/metrics.json")
    except SystemExit as e:
        pg.update(stage="실패", message=str(e))
        pg.say(f"실패: {e}")
        raise
    except BaseException as e:  # 예기치 못한 오류도 status 에 남긴다
        pg.update(stage="실패", message=f"{type(e).__name__}: {e}")
        pg.say(f"실패: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
