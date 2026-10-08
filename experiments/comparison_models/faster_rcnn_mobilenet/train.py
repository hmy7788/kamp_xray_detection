"""Faster R-CNN MobileNetV3-Large FPN 학습. faster_rcnn/train.py 를 그대로 쓰되 모델만 MobileNet 으로 바꿔 끼운다 (같은 조건 비교용).

  python src/minyeop/faster_rcnn_mobilenet/train.py --name 04_frcnn_mobv3_min640_anc16_v1 --epochs 20 --batch-size 4 --anchor-sizes 16,32,64,128,256

옵션과 결과 저장 위치(runs/minyeop/<name>/)는 faster_rcnn/train.py 와 같다. 진행 확인은 faster_rcnn/watch.py 를 쓴다.
"""
import sys
from pathlib import Path

HERE, FR = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1] / "faster_rcnn"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FR))  # faster_rcnn 의 train.py 가 먼저 잡히도록 맨 앞에 둔다
import train as base  # noqa: E402  (faster_rcnn/train.py)
from mobilenet_model import build_model  # noqa: E402

base.build_model = build_model

if __name__ == "__main__":
    base.main()
