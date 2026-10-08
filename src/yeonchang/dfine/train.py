"""D-FINE-N 학습 (HuggingFace transformers 구현, COCO 사전학습 ustc-community/dfine-nano-coco에서 시작).

    $env:PYTHONUTF8="1"
    python src/yeonchang/dfine/train.py --exp 03_dfinen_img640_e50

결과: runs/yeonchang/<exp>/ (config.json, train.log, results.csv; weights/는 Git 제외)
이미 있는 실험 폴더는 덮어쓰지 않고 중단한다. 평가는 predict.py → ../rtdetr/evaluate.py --reuse-preds.
RT-DETR(02)과 같은 분할·증강 방향·선택 기준(0.1*AP50 + 0.9*AP50-95, val)을 쓴다.
"""
import argparse
import copy
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rtdetr"))
import evaluate as E  # noqa: E402
from common import DATA, RUNS, check_data_version, env_info, load_gt, read_manifest, sanitize_paths  # noqa: E402

CKPT = "ustc-community/dfine-nano-coco"
HYP = dict(
    imgsz=640,          # 전처리 기본값(640x640으로 늘림, 비율 유지 안 함)
    epochs=50,          # 작은 모델이라 RT-DETR-l(30)보다 길게, patience로 조기 종료
    patience=10,
    batch=16,           # 3.7M 모델이라 6GB에서 여유
    lr=2e-4,            # 인코더·디코더·헤드
    backbone_lr=1e-4,   # 사전학습 백본은 절반
    weight_decay=1e-4,  # bias·정규화 층 제외
    warmup_epochs=1,
    lrf=0.1,            # 코사인으로 lr*lrf까지
    clip=0.1,           # DETR 계열 표준 기울기 클리핑
    ema_decay=0.9999,   # ultralytics RT-DETR과 같은 EMA(초반 tau 2000 반복 동안 완만)
    amp=True,
    # 증강: RT-DETR(02)과 같은 방향. 회색조라 색상 없음, mosaic·큰 축소 없음
    hsv_v=0.3, fliplr=0.5, flipud=0.5, translate=0.1, scale=0.2,
    workers=4,
    seed=0,
)


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def to_tensor(img, size):
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(x[..., ::-1].transpose(2, 0, 1))).float() / 255.0


class XrayDataset(Dataset):
    def __init__(self, ids, manifest, gt, hyp, augment):
        self.ids, self.manifest, self.gt, self.hyp, self.augment = ids, manifest, gt, hyp, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        img = imread(DATA / self.manifest[i]["image_path"])
        h, w = img.shape[:2]
        boxes = np.array([b[1:] for b in self.gt[i]], np.float32).reshape(-1, 4)
        if self.augment:
            hp = self.hyp
            img = np.clip(img.astype(np.float32) * random.uniform(1 - hp["hsv_v"], 1 + hp["hsv_v"]), 0, 255).astype(np.uint8)
            s = random.uniform(1 - hp["scale"], 1 + hp["scale"])
            tx, ty = random.uniform(-hp["translate"], hp["translate"]) * w, random.uniform(-hp["translate"], hp["translate"]) * h
            M = np.array([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]], np.float32)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if len(boxes):
                area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes * s + np.array([M[0, 2], M[1, 2], M[0, 2], M[1, 2]], np.float32)
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes[(area > 0.6 * area0 * s * s) & (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)]
            if random.random() < hp["fliplr"]:
                img = img[:, ::-1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < hp["flipud"]:
                img = img[::-1]
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        cxcywh = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h,
                           (boxes[:, 2] - boxes[:, 0]) / w, (boxes[:, 3] - boxes[:, 1]) / h], 1) if len(boxes) else np.zeros((0, 4), np.float32)
        return to_tensor(img, self.hyp["imgsz"]), {"class_labels": torch.zeros(len(cxcywh), dtype=torch.long),
                                                   "boxes": torch.from_numpy(cxcywh.astype(np.float32))}


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


class ModelEMA:
    def __init__(self, model, decay, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


@torch.no_grad()
def predict_images(model, processor, imgs, size, device, batch=16):
    """원본 BGR 이미지 목록 → [(conf, x1, y1, x2, y2), ...] 목록 (원본 픽셀 좌표)."""
    out = []
    for k in range(0, len(imgs), batch):
        chunk = imgs[k:k + batch]
        x = torch.stack([to_tensor(im, size) for im in chunk]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            o = model(pixel_values=x)
        o.logits, o.pred_boxes = o.logits.float(), o.pred_boxes.float()
        res = processor.post_process_object_detection(o, threshold=0.001, target_sizes=[im.shape[:2] for im in chunk])
        for r in res:
            out.append([(float(c), *map(float, b)) for c, b in zip(r["scores"].tolist(), r["boxes"].tolist())])
    return out


def val_metrics(preds_list, ids, gt):
    preds = dict(zip(ids, preds_list))
    aps = []
    for t in np.round(np.arange(0.5, 0.96, 0.05), 2):
        E.IOU_THR = float(t)
        aps.append(E.evaluate(preds, gt, 1.1, "iou")[0]["ap50"])
    E.IOU_THR = 0.5
    return aps[0], float(np.mean(aps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-iters", type=int, default=0, help="점검용: epoch당 반복 수 제한 (0=전체)")
    for k, v in HYP.items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else lambda s: s.lower() in ("1", "true"), default=v)
    args = ap.parse_args()
    hyp = {k: getattr(args, k) for k in HYP}

    run_dir = RUNS / args.exp
    if run_dir.exists():
        sys.exit(f"이미 있는 실험 폴더입니다: {run_dir} (덮어쓰지 않음, 새 번호를 쓰세요)")
    digest = check_data_version()
    run_dir.mkdir(parents=True)
    (run_dir / "weights").mkdir()

    log = logging.getLogger("dfine")
    log.setLevel(logging.INFO)
    for hd in (logging.FileHandler(run_dir / "train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        hd.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(hd)

    random.seed(hyp["seed"]); np.random.seed(hyp["seed"]); torch.manual_seed(hyp["seed"])
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    import transformers
    from transformers import AutoImageProcessor, DFineForObjectDetection
    processor = AutoImageProcessor.from_pretrained(args.ckpt)
    model = DFineForObjectDetection.from_pretrained(args.ckpt, num_labels=1, id2label={0: "defect"},
                                                    label2id={"defect": 0}, ignore_mismatched_sizes=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    manifest = read_manifest()
    gt_tr, gt_va = load_gt("train", manifest), load_gt("val", manifest)
    tr_ids, va_ids = sorted(gt_tr), sorted(gt_va)
    va_imgs = [imread(DATA / manifest[i]["image_path"]) for i in va_ids]
    dl = DataLoader(XrayDataset(tr_ids, manifest, gt_tr, hyp, True), batch_size=hyp["batch"], shuffle=True,
                    num_workers=hyp["workers"], collate_fn=collate, drop_last=True,
                    persistent_workers=hyp["workers"] > 0, pin_memory=True)

    decay, no_decay, bb = [], [], []
    for n, p in model.named_parameters():
        if "backbone" in n:
            bb.append(p)
        elif p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([{"params": bb, "lr": hyp["backbone_lr"], "weight_decay": hyp["weight_decay"]},
                             {"params": decay, "lr": hyp["lr"], "weight_decay": hyp["weight_decay"]},
                             {"params": no_decay, "lr": hyp["lr"], "weight_decay": 0.0}])
    base_lrs = [g["lr"] for g in opt.param_groups]
    iters = args.max_iters or len(dl)
    total, warm = hyp["epochs"] * iters, max(1, int(hyp["warmup_epochs"] * iters))

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return hyp["lrf"] + (1 - hyp["lrf"]) * 0.5 * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler("cuda", enabled=hyp["amp"] and device.type == "cuda")
    ema = ModelEMA(model, hyp["ema_decay"])

    config = {"exp": args.exp, "model": "D-FINE-N", "params": n_params,
              "init_weights": f"{args.ckpt} (COCO 사전학습, HuggingFace, Apache-2.0)",
              "data": "data/ v2 (train 학습, val 학습 중 선택·임계값, test 최종 1회)", "manifest_sha256": digest,
              "preprocess": f"{hyp['imgsz']}x{hyp['imgsz']}로 늘림(비율 유지 안 함), /255, 정규화 없음 (체크포인트 기본값)",
              "select": "val 0.1*AP50 + 0.9*AP50-95 (VOC 보간, EMA 가중치)", "hyp": hyp,
              "env": {**env_info(), "transformers": transformers.__version__}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"D-FINE-N {n_params:,} params | train {len(tr_ids)} val {len(va_ids)} | {iters} iters/epoch | device {device}")

    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "time_min", "train_loss", "lr", "val_ap50", "val_ap50_95", "fitness"])
    best, best_epoch, step, t0 = -1.0, 0, 0, time.time()
    for epoch in range(1, hyp["epochs"] + 1):
        model.train()
        losses = []
        for it, (x, labels) in enumerate(dl):
            if it >= iters:
                break
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            x = x.to(device, non_blocking=True)
            labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = model(pixel_values=x, labels=labels).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp["clip"])
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            losses.append(float(loss))
            step += 1
        ap50, ap5095 = val_metrics(predict_images(ema.ema, processor, va_imgs, hyp["imgsz"], device), va_ids, gt_va)
        fit = 0.1 * ap50 + 0.9 * ap5095
        row = [epoch, round((time.time() - t0) / 60, 2), round(float(np.mean(losses)), 4),
               f"{opt.param_groups[1]['lr']:.2e}", round(ap50, 4), round(ap5095, 4), round(fit, 4)]
        with open(run_dir / "results.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        ema.ema.save_pretrained(run_dir / "weights" / "last")
        if fit > best:
            best, best_epoch = fit, epoch
            ema.ema.save_pretrained(run_dir / "weights" / "best")
            processor.save_pretrained(run_dir / "weights" / "best")
        log.info(f"epoch {epoch}/{hyp['epochs']} loss {row[2]} val AP50 {ap50:.4f} AP50-95 {ap5095:.4f} fit {fit:.4f} "
                 f"(best {best:.4f} @ {best_epoch}) {row[1]} min")
        if epoch - best_epoch >= hyp["patience"]:
            log.info(f"조기 종료: {hyp['patience']} epoch 동안 val 적합도가 오르지 않음")
            break

    config.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_minutes": round((time.time() - t0) / 60, 1),
                   "epochs_run": epoch, "best_epoch": best_epoch, "best_fitness": round(best, 4)})
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for hd in list(log.handlers):
        hd.close(); log.removeHandler(hd)
    for name in ("train.log", "config.json"):
        sanitize_paths(run_dir / name)
    print(f"학습 완료: {run_dir}  다음: python src/yeonchang/dfine/predict.py --exp {args.exp} --split val")


if __name__ == "__main__":
    main()
