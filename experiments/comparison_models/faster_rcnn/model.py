"""Faster R-CNN (ResNet-50 + FPN, COCO 사전학습) 을 결함 1개 클래스용으로 만든다.

결함이 작다(약 10px)는 점이 핵심이다.
- torchvision 기본값은 짧은 변을 800px로 확대하고 앵커 최소 크기가 32px이다. 입력을 확대하면(예: 640)
  결함이 약 20~25px가 되어 기본 앵커로도 어느 정도 맞는다.
- 확대를 줄이고 싶으면 --anchor-sizes 로 앵커를 작게 줄 수 있다. 수준마다 앵커를 1개씩(종횡비 3종) 두어
  수준당 앵커 수(3)가 기본과 같으므로 사전학습된 RPN 헤드 가중치를 그대로 읽는다.
"""
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(min_size=640, max_size=1000, anchor_sizes=None, pretrained=True, score_thresh=0.001, max_dets=100):
    kw = {}
    if anchor_sizes:
        kw["rpn_anchor_generator"] = AnchorGenerator(
            sizes=tuple((int(s),) for s in anchor_sizes), aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes))
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    # weights 가 있으면 백본도 거기서 오므로, 없을 때 ImageNet 백본을 따로 내려받지 않게 None 으로 둔다
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size,
                                    box_score_thresh=score_thresh, box_detections_per_img=max_dets, **kw)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 배경 + defect
    return model
