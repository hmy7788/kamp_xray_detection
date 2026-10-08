"""2단계: 학습. train으로 학습하고 val로 체크포인트를 고른다 (test는 쓰지 않는다).

    python scripts/train.py                        # 최종 모델 D-FINE-N (RTX 4050 Laptop 기준 약 100분)
    python scripts/train.py --model yolov3_tiny    # 베이스라인 (약 60분, 먼저 scripts/download_pretrained.py)

결과: outputs/<model>/train/ (weights/, config.json, train.log, results.csv 또는 results.txt)
이미 있는 폴더는 덮어쓰지 않는다. 동작 확인만 할 때는 --epochs 1 (D-FINE은 --max-iters 20 도 가능).
제출 가중치(weights/<model>/)는 이 스크립트로 만든 것이며, GPU 연산의 비결정성 때문에 다시 학습하면 수치가 조금 다를 수 있다.
"""
import _bootstrap  # noqa: F401

import argparse

from kamp_xray.common import CONFIGS, OUTPUTS, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dfine_n", choices=["dfine_n", "yolov3_tiny"])
    ap.add_argument("--out", default=None, help="기본: outputs/<model>/train")
    ap.add_argument("--device", default="0", help="GPU 번호 또는 cpu")
    ap.add_argument("--epochs", type=int, default=None, help="설정 파일의 epoch 수를 바꿀 때 (동작 확인용)")
    ap.add_argument("--max-iters", type=int, default=0, help="D-FINE 동작 확인용: epoch당 반복 수 제한")
    a = ap.parse_args()
    hyp = load_yaml(CONFIGS / f"{a.model}.yaml")
    out = a.out or OUTPUTS / a.model / "train"
    if a.model == "dfine_n":
        from kamp_xray import dfine
        if a.epochs:
            hyp["epochs"] = a.epochs
        best = dfine.train(hyp, out, a.device, a.max_iters)
    else:
        from kamp_xray import yolov3_tiny
        best = yolov3_tiny.train(hyp, out, a.device, a.epochs)
    print(f"학습 완료. 가중치: {best}")
    print(f"다음: python scripts/predict.py --model {a.model} --split val --weights \"{best}\"")


if __name__ == "__main__":
    main()
