"""Faster R-CNN (ResNet-50 FPN) 학습. 결과는 runs/minyeop/<name>/ 에 저장된다.

예)
  python src/minyeop/faster_rcnn/train.py --name 01_frcnn_r50fpn_min640_v1 --epochs 20 --batch-size 4

진행 상황은 콘솔과 runs/minyeop/<name>/progress.log 에 epoch 마다 한 줄씩 남고, status.json 이 갱신된다.
epoch 마다 val 로 AP@0.5 를 재고 가장 높은 epoch 의 가중치를 best.pt 로 저장한다. **test 는 쓰지 않는다**
(최종 후보를 한 번만 평가해야 하므로 predict.py 로 따로 한다).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from dataset import ROOT, YoloDetDataset, collate  # noqa: E402
from model import build_model  # noqa: E402


def keep_awake(on):
    """학습 중 Windows 절전을 막는다 (시스템 설정은 바꾸지 않음)."""
    if sys.platform != "win32":
        return
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def fmt(sec):
    sec = int(max(sec, 0))
    h, m = divmod(sec // 60, 60)
    return f"{h}시간 {m:02d}분" if h else f"{m}분 {sec % 60:02d}초"


class Reporter:
    def __init__(self, out):
        self.out, self.t0 = out, time.time()
        self.f = open(out / "progress.log", "a", encoding="utf-8")
        self.state = {"stage": "시작", "epoch": 0, "pid": os.getpid()}

    def say(self, msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()

    def update(self, **kw):
        """상태 파일은 보조 정보라, OneDrive 등이 파일을 잡고 있어 교체가 거부돼도 학습이 죽지 않게 한다."""
        self.state.update(kw, elapsed_sec=int(time.time() - self.t0), updated_at=datetime.now().isoformat(timespec="seconds"))
        text = json.dumps(self.state, ensure_ascii=False, indent=2)
        tmp = self.out / "status.json.tmp"
        for _ in range(5):
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, self.out / "status.json")
                return
            except OSError:
                time.sleep(0.5)


@torch.no_grad()
def infer(model, ds, device, batch_size, amp, workers=2):
    """데이터셋 전체를 추론해 공통 형식의 기록과 이미지당 추론 시간(ms)을 돌려준다."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
    recs, total_t, n = [], 0.0, 0
    for images, targets in dl:
        images = [im.to(device) for im in images]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_t += time.time() - t; n += len(images)
        for im, tg, o in zip(images, targets, outs):
            dets = torch.cat([o["boxes"], o["scores"][:, None]], 1).float().cpu().tolist()
            recs.append({"name": ds.names[int(tg["image_id"])], "w": im.shape[2], "h": im.shape[1],
                         "gt": tg["boxes"].tolist(), "dets": dets})
    return recs, 1000 * total_t / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="실험 이름 (결과: runs/<owner>/<name>/)")
    ap.add_argument("--owner", default="minyeop")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--min-size", type=int, default=640, help="입력의 짧은 변을 이 크기로 확대 (작은 결함을 키우는 효과)")
    ap.add_argument("--max-size", type=int, default=1000)
    ap.add_argument("--anchor-sizes", default=None, help="예: 16,32,64,128,256 (기본 32,64,128,256,512)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", help="혼합 정밀도 끄기")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=None, help="점검용: 학습 이미지 수 제한")
    ap.add_argument("--limit-val", type=int, default=None, help="점검용: 검증 이미지 수 제한")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = ROOT / "runs" / a.owner / a.name
    if out.exists():
        sys.exit(f"이미 존재하는 실험 폴더입니다(덮어쓰기 방지): runs/{a.owner}/{a.name}")
    (out / "weights").mkdir(parents=True)
    anchors = [int(s) for s in a.anchor_sizes.split(",")] if a.anchor_sizes else None
    (out / "config.json").write_text(json.dumps({**vars(a), "anchor_sizes_list": anchors, "model": "fasterrcnn_resnet50_fpn(COCO_V1)"},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not a.no_amp) and device.type == "cuda"
    rp = Reporter(out)
    keep_awake(True)
    try:
        train_ds = YoloDetDataset("train", train=True, limit=a.limit_train)
        val_ds = YoloDetDataset("val", train=False, limit=a.limit_val)
        dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate,
                        persistent_workers=a.workers > 0, pin_memory=device.type == "cuda", drop_last=True)
        model = build_model(a.min_size, a.max_size, anchors).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=5e-4)
        total, warm = a.epochs * len(dl), min(300, max(len(dl) - 1, 1))
        lr_at = lambda it: (0.001 + 0.999 * it / warm) if it < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (it - warm) / max(total - warm, 1)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

        rp.say(f"실험 시작: {a.owner}/{a.name} | Faster R-CNN R50-FPN | 입력 {a.min_size} | 앵커 {anchors or '기본'} | "
               f"batch {a.batch_size} | epochs {a.epochs} | AMP {amp} | 학습 {len(train_ds)}장 / 검증 {len(val_ds)}장 | {device}")
        best_ap, rows = -1.0, []
        t_train = time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            t_ep, loss_sum, k, skipped = time.time(), 0.0, 0, 0
            for it, (images, targets) in enumerate(dl):
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{n: v.to(device) for n, v in t.items()} for t in targets]
                with torch.autocast(device_type=device.type, enabled=amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):  # 드물게 발산하면 그 배치만 건너뜀
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                loss_sum += float(loss); k += 1
                if it % 50 == 0:
                    rp.update(stage="학습", epoch=ep, epochs=a.epochs, iter=it, iters=len(dl), loss=round(loss_sum / max(k, 1), 4))
            tr_time = time.time() - t_ep
            recs, ms = infer(model, val_ds, device, max(a.batch_size, 4), amp)
            ap50, bf = metrics.ap(recs), metrics.best_f1(recs)
            row = {"epoch": ep, "train_loss": round(loss_sum / max(k, 1), 4), "val_AP50": round(ap50, 4), "val_thr": bf["thr"],
                   "val_P": round(bf["P"], 4), "val_R": round(bf["R"], 4), "val_F1": round(bf["F1"], 4),
                   "val_TP": bf["TP"], "val_FP": bf["FP"], "val_FN": bf["FN"], "infer_ms": round(ms, 1), "lr": opt.param_groups[0]["lr"]}
            rows.append(row)
            ck = {"model": model.state_dict(), "epoch": ep, "val_AP50": ap50, "config": vars(a)}
            torch.save(ck, out / "weights" / "last.pt")
            if ap50 > best_ap:
                best_ap = ap50
                torch.save(ck, out / "weights" / "best.pt")
            with open(out / "results.csv", "w", encoding="utf-8") as f:
                f.write(",".join(rows[0]) + "\n")
                for r in rows:
                    f.write(",".join(str(v) for v in r.values()) + "\n")
            eta = (time.time() - t_train) / ep * (a.epochs - ep)
            rp.say(f"epoch {ep:>3}/{a.epochs} loss {row['train_loss']:.4f} | val AP@0.5 {ap50:.3f} | 임계값 {bf['thr']} "
                   f"P {bf['P']:.3f} R {bf['R']:.3f} F1 {bf['F1']:.3f} (TP {bf['TP']} FP {bf['FP']} FN {bf['FN']}) | "
                   f"최고 AP {best_ap:.3f} | 학습 {fmt(tr_time)} | 추론 {ms:.0f}ms/장 | 남은 시간 약 {fmt(eta)}"
                   + (f" | 건너뛴 배치 {skipped}" if skipped else ""))
            rp.update(stage="학습", epoch=ep, epochs=a.epochs, val_AP50=round(ap50, 4), best_AP50=round(best_ap, 4), eta_sec=int(eta))
        rp.update(stage="완료")
        rp.say(f"완료: 최고 val AP@0.5 {best_ap:.3f} (best.pt). test 는 predict.py 로 최종 후보만 한 번 평가")
    except BaseException as e:
        rp.update(stage="실패", message=f"{type(e).__name__}: {e}")
        rp.say(f"실패: {type(e).__name__}: {e}")
        raise
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
