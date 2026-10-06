"""RT-DETR 학습 (ultralytics 구현, COCO 사전학습에서 시작).

    $env:PYTHONUTF8="1"
    python src/yeonchang/rtdetr/train.py --exp 02_rtdetrl_img640_e30 --epochs 30 --patience 10
    # val 곡선이 끝까지 오르고 있으면 이어서 (새 실험 폴더, 학습률 스케줄은 새로 시작):
    python src/yeonchang/rtdetr/train.py --exp 03_rtdetrl_img640_e30+20 --init runs/yeonchang/02_rtdetrl_img640_e30/weights/last.pt --epochs 20 --patience 10 --warmup_epochs 1

결과: runs/yeonchang/<exp>/ (config.json, train.log, results.csv, args.yaml, weights/는 Git 제외)
이미 있는 실험 폴더는 덮어쓰지 않고 중단한다.
"""
import argparse
import json
import logging
import sys
import time

from common import ROOT, RUNS, WEIGHTS, check_data_version, env_info, sanitize_paths, write_data_yaml

# 6GB GPU(RTX 4050 Laptop)에 맞춘 기본값. 결함이 약 10px로 작아서 크기를 줄이는 증강은 약하게 둔다.
HYP = dict(
    imgsz=640,          # 원본 최대 576x444 → 640으로 키워 작은 결함을 조금이라도 크게 본다
    epochs=30,          # 1클래스·COCO 사전학습 미세조정. val 곡선이 끝까지 오르면 --init으로 이어 학습
    patience=10,        # val 적합도가 10 epoch 동안 오르지 않으면 조기 종료
    batch=4,            # rtdetr-l + 640 + AMP 기준 6GB 안쪽
    nbs=16,             # 명목 배치(기울기 누적 4회)
    optimizer="AdamW",
    lr0=1e-4,           # DETR 계열 표준 학습률
    lrf=0.1,
    weight_decay=1e-4,
    warmup_epochs=2,
    cos_lr=True,
    amp=True,
    # 증강: 회색조라 색상 증강은 끄고, 결함을 더 작게 만드는 mosaic/큰 축소는 쓰지 않는다
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
    fliplr=0.5, flipud=0.5,
    degrees=0.0, translate=0.1, scale=0.2,
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    cache="ram",
    workers=4,
    seed=0,
    deterministic=False,  # RT-DETR의 grid_sample은 결정적 알고리즘이 없어 True면 오류가 난다
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="실험 폴더 이름 <번호>_<모델>_<설정>")
    ap.add_argument("--model", default="rtdetr-l.pt", help="COCO 사전학습 가중치 (rtdetr-l.pt / rtdetr-x.pt)")
    ap.add_argument("--init", default=None,
                    help="이어서 학습할 체크포인트 (예: runs/yeonchang/02_.../weights/last.pt). 주면 --model 대신 이것에서 시작")
    ap.add_argument("--device", default="0")
    for k, v in HYP.items():
        ap.add_argument(f"--{k}", type=type(v) if not isinstance(v, bool) else lambda s: s.lower() in ("1", "true"), default=v)
    args = ap.parse_args()

    run_dir = RUNS / args.exp
    if run_dir.exists():
        sys.exit(f"이미 있는 실험 폴더입니다: {run_dir} (덮어쓰지 않음, 새 번호를 쓰세요)")
    digest = check_data_version()
    run_dir.mkdir(parents=True)

    # ultralytics 로그를 train.log에도 남긴다
    from ultralytics import RTDETR
    from ultralytics.utils import LOGGER
    from ultralytics.utils.downloads import attempt_download_asset

    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOGGER.addHandler(fh)

    hyp = {k: getattr(args, k) for k in HYP}
    if args.init:
        init = (ROOT / args.init).resolve()
        if not init.exists():
            sys.exit(f"체크포인트가 없습니다: {init}")
        init_desc = f"{init.relative_to(ROOT).as_posix()} (이전 실험에서 이어 학습, 학습률 스케줄은 새로 시작)"
    else:
        init = None
        init_desc = f"weights/{args.model} (COCO 사전학습, ultralytics 배포본)"
    config = {
        "exp": args.exp,
        "model": args.model,
        "init_weights": init_desc,
        "data": "data/ v2 (train 학습, val 학습 중 선택·임계값, test 최종 1회)",
        "manifest_sha256": digest,
        "hyp": hyp,
        "env": env_info(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    weights = str(init) if init else attempt_download_asset(str(WEIGHTS / args.model))
    data_yaml = write_data_yaml(run_dir / "weights" / "data.yaml")
    model = RTDETR(weights)
    t0 = time.time()
    model.train(data=str(data_yaml), project=str(RUNS), name=args.exp, exist_ok=True,
                device=args.device, plots=True, val=True, **hyp)

    config["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    config["train_minutes"] = round((time.time() - t0) / 60, 1)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.removeHandler(fh)
    fh.close()
    for name in ("train.log", "args.yaml", "config.json"):
        sanitize_paths(run_dir / name)
    print(f"학습 완료: {run_dir}  다음 단계: python src/yeonchang/rtdetr/evaluate.py --exp {args.exp} --split val")


if __name__ == "__main__":
    main()
