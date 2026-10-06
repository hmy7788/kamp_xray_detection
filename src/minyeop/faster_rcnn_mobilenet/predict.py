"""학습된 Faster R-CNN MobileNetV3-FPN 으로 val/test 를 추론해 검출 결과(preds_<split>.json)를 저장하고 지표를 출력한다.

  python src/minyeop/faster_rcnn_mobilenet/predict.py --name 04_frcnn_mobv3_min640_anc16_v1 --split val  --ckpt best
  python src/minyeop/faster_rcnn_mobilenet/predict.py --name 04_frcnn_mobv3_min640_anc16_v1 --split test --ckpt best   # 최종 후보만 한 번

faster_rcnn/predict.py 와 규칙이 같다 (임계값은 같은 체크포인트의 val 결과에서 정해 test 에 적용). 이후 report.py, visualize.py 는 faster_rcnn 폴더의 것을 그대로 쓴다.
"""
import sys
from pathlib import Path

HERE, FR = Path(__file__).resolve().parent, Path(__file__).resolve().parents[1] / "faster_rcnn"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FR))
import predict as base  # noqa: E402  (faster_rcnn/predict.py)
from mobilenet_model import build_model  # noqa: E402

base.build_model = build_model

if __name__ == "__main__":
    base.main()
