"""최종 모델 D-FINE-N 학습·추론 (Hugging Face transformers 구현).

원작성: 정연창 (원 실험 이름 03_dfinen_img640_e50). 제출용으로 경로와 설정 파일만 정리했고,
전처리·증강·학습 루프·추론 방식은 제출 가중치(weights/dfine_n/best)를 만든 코드와 같다.
"""
import copy
import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import metrics as M
from .common import (DATA, check_data_version, env_info, imread, make_records, read_manifest, sanitize_paths,
                     seed_everything, select_device, split_ids)


def to_tensor(img, size):
    """BGR uint8 → size x size RGB float 텐서(0~1). 비율 유지 없이 늘린다."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(x[..., ::-1].transpose(2, 0, 1))).float() / 255.0


def gt_xyxy(manifest, ids):
    recs = make_records(manifest, ids, [[] for _ in ids])
    return {r["image_id"]: np.array(r["gt"], np.float32).reshape(-1, 4) for r in recs}


class XrayDataset(Dataset):
    def __init__(self, ids, manifest, gt, hyp, augment):
        self.ids, self.manifest, self.gt, self.hyp, self.augment = ids, manifest, gt, hyp, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        img = imread(DATA / self.manifest[i]["image_path"])
        h, w = img.shape[:2]
        boxes = self.gt[i].copy()
        if self.augment:
            hp = self.hyp
            img = np.clip(img.astype(np.float32) * random.uniform(1 - hp["hsv_v"], 1 + hp["hsv_v"]), 0, 255).astype(np.uint8)
            s = random.uniform(1 - hp["scale"], 1 + hp["scale"])
            tx = random.uniform(-hp["translate"], hp["translate"]) * w
            ty = random.uniform(-hp["translate"], hp["translate"]) * h
            Mx = np.array([[s, 0, (1 - s) * w / 2 + tx], [0, s, (1 - s) * h / 2 + ty]], np.float32)
            img = cv2.warpAffine(img, Mx, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if len(boxes):
                area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes = boxes * s + np.array([Mx[0, 2], Mx[1, 2], Mx[0, 2], Mx[1, 2]], np.float32)
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                keep = (area > 0.6 * area0 * s * s) & (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
                boxes = boxes[keep]
            if random.random() < hp["fliplr"]:
                img = img[:, ::-1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < hp["flipud"]:
                img = img[::-1]
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        if len(boxes):
            cxcywh = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 1] + boxes[:, 3]) / 2 / h,
                               (boxes[:, 2] - boxes[:, 0]) / w, (boxes[:, 3] - boxes[:, 1]) / h], 1)
        else:
            cxcywh = np.zeros((0, 4), np.float32)
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
    """원본 BGR 이미지 목록 → 이미지별 [[x1, y1, x2, y2, score], ...] (원본 픽셀 좌표, 신뢰도 0.001 이상)."""
    out = []
    for k in range(0, len(imgs), batch):
        chunk = imgs[k:k + batch]
        x = torch.stack([to_tensor(im, size) for im in chunk]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            o = model(pixel_values=x)
        o.logits, o.pred_boxes = o.logits.float(), o.pred_boxes.float()
        res = processor.post_process_object_detection(o, threshold=M.CONF_FLOOR,
                                                      target_sizes=[im.shape[:2] for im in chunk])
        for r in res:
            out.append([[*map(float, b), float(c)] for c, b in zip(r["scores"].tolist(), r["boxes"].tolist())])
    return out


def load_model(weights, device):
    from transformers import AutoImageProcessor, DFineForObjectDetection
    processor = AutoImageProcessor.from_pretrained(weights)
    model = DFineForObjectDetection.from_pretrained(weights).to(device).eval()
    return model, processor


def predict(weights, split, device="0", imgsz=640):
    """split 전체를 추론해 공통 예측 기록 목록을 돌려준다."""
    dev = select_device(device)
    model, processor = load_model(weights, dev)
    manifest = read_manifest()
    ids = split_ids(manifest, split)
    imgs = [imread(DATA / manifest[i]["image_path"]) for i in ids]
    return make_records(manifest, ids, predict_images(model, processor, imgs, imgsz, dev))


def train(hyp, out_dir, device="0", max_iters=0):
    """train으로 학습하고 val로 체크포인트를 고른다. test는 쓰지 않는다.
    결과: out_dir/weights/best (제출 가중치와 같은 형식), config.json, train.log, results.csv"""
    out_dir = Path(out_dir)
    digest = check_data_version()
    (out_dir / "weights").mkdir(parents=True, exist_ok=False)
    log = logging.getLogger("dfine")
    log.setLevel(logging.INFO)
    for hd in (logging.FileHandler(out_dir / "train.log", encoding="utf-8"), logging.StreamHandler()):
        hd.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(hd)

    seed_everything(hyp["seed"])
    dev = select_device(device)
    import transformers
    from transformers import AutoImageProcessor, DFineForObjectDetection
    processor = AutoImageProcessor.from_pretrained(hyp["pretrained"])
    model = DFineForObjectDetection.from_pretrained(hyp["pretrained"], num_labels=1, id2label={0: "defect"},
                                                    label2id={"defect": 0}, ignore_mismatched_sizes=True).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    manifest = read_manifest()
    tr_ids, va_ids = split_ids(manifest, "train"), split_ids(manifest, "val")
    gt_tr = gt_xyxy(manifest, tr_ids)
    va_imgs = [imread(DATA / manifest[i]["image_path"]) for i in va_ids]
    dl = DataLoader(XrayDataset(tr_ids, manifest, gt_tr, hyp, True), batch_size=hyp["batch"], shuffle=True,
                    num_workers=hyp["workers"], collate_fn=collate, drop_last=True,
                    persistent_workers=hyp["workers"] > 0, pin_memory=True)

    decay, no_decay, bb = [], [], []
    for n, p in model.named_parameters():
        (bb if "backbone" in n else no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": bb, "lr": hyp["backbone_lr"], "weight_decay": hyp["weight_decay"]},
                             {"params": decay, "lr": hyp["lr"], "weight_decay": hyp["weight_decay"]},
                             {"params": no_decay, "lr": hyp["lr"], "weight_decay": 0.0}])
    base_lrs = [g["lr"] for g in opt.param_groups]
    iters = min(max_iters, len(dl)) if max_iters else len(dl)
    total, warm = hyp["epochs"] * iters, max(1, int(hyp["warmup_epochs"] * iters))

    def lr_factor(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return hyp["lrf"] + (1 - hyp["lrf"]) * 0.5 * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler("cuda", enabled=hyp["amp"] and dev.type == "cuda")
    ema = ModelEMA(model, hyp["ema_decay"])
    config = {"model": "D-FINE-N", "params": n_params, "init_weights": f"{hyp['pretrained']} (COCO 사전학습)",
              "data": "train 학습, val 체크포인트 선택, test 미사용", "manifest_sha256": digest, "hyp": hyp,
              "env": {**env_info(), "transformers": transformers.__version__},
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"D-FINE-N {n_params:,} params | train {len(tr_ids)} val {len(va_ids)} | {iters} iters/epoch | device {dev}")

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "time_min", "train_loss", "lr", "val_ap50", "val_ap50_95", "fitness"])
    best, best_epoch, step, t0, epoch = -1.0, 0, 0, time.time(), 0
    for epoch in range(1, hyp["epochs"] + 1):
        model.train()
        losses = []
        for it, (x, labels) in enumerate(dl):
            if it >= iters:
                break
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr_factor(step)
            x = x.to(dev, non_blocking=True)
            labels = [{k: v.to(dev) for k, v in lab.items()} for lab in labels]
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
        recs = make_records(manifest, va_ids, predict_images(ema.ema, processor, va_imgs, hyp["imgsz"], dev))
        ap50, ap5095 = M.ap(recs), M.map50_95(recs)
        fit = 0.1 * ap50 + 0.9 * ap5095
        row = [epoch, round((time.time() - t0) / 60, 2), round(float(np.mean(losses)), 4),
               f"{opt.param_groups[1]['lr']:.2e}", round(ap50, 4), round(ap5095, 4), round(fit, 4)]
        with open(out_dir / "results.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        if fit > best:
            best, best_epoch = fit, epoch
            ema.ema.save_pretrained(out_dir / "weights" / "best")
            processor.save_pretrained(out_dir / "weights" / "best")
        log.info(f"epoch {epoch}/{hyp['epochs']} loss {row[2]} val AP50 {ap50:.4f} AP50-95 {ap5095:.4f} "
                 f"fit {fit:.4f} (best {best:.4f} @ {best_epoch}) {row[1]} min")
        if epoch - best_epoch >= hyp["patience"]:
            log.info(f"조기 종료: {hyp['patience']} epoch 동안 val 적합도가 오르지 않음")
            break

    config.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "train_minutes": round((time.time() - t0) / 60, 1),
                   "epochs_run": epoch, "best_epoch": best_epoch, "best_fitness": round(best, 4)})
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for hd in list(log.handlers):
        hd.close()
        log.removeHandler(hd)
    for name in ("train.log", "config.json"):
        sanitize_paths(out_dir / name)
    return out_dir / "weights" / "best"
