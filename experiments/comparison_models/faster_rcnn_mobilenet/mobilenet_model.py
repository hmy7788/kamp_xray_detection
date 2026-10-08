"""Faster R-CNN (MobileNetV3-Large + FPN, COCO 사전학습) 을 결함 1개 클래스용으로 만든다.

faster_rcnn/model.py(ResNet-50 FPN)와 같은 인터페이스(build_model)라서, 학습·예측·보고 코드를 그대로 재사용한다.
- 이 모델의 FPN 은 해상도가 낮은 특징맵(stride 16, 32, 64) 3개만 쓴다. ResNet-50 FPN 은 stride 4 부터 쓰므로 작은 결함(약 10px)에는 불리할 수 있다.
- 기본 앵커는 레벨마다 (32, 64, 128, 256, 512). --anchor-sizes 에 5개 크기를 주면(예: 16,32,64,128,256) 3개 레벨 모두에 적용한다.
  위치당 앵커 수(5 크기 x 3 종횡비 = 15)가 같아 사전학습된 RPN 헤드 가중치를 그대로 읽는다.
"""
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(min_size=640, max_size=1000, anchor_sizes=None, pretrained=True, score_thresh=0.001, max_dets=100):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    # 앵커는 생성 인자로 넘기면 torchvision 이 중복 인자 오류를 내므로, 만든 뒤에 바꿔 끼운다
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None, min_size=min_size, max_size=max_size,
                                              box_score_thresh=score_thresh, box_detections_per_img=max_dets)
    if anchor_sizes:
        assert len(anchor_sizes) == 5, "--anchor-sizes 는 5개 크기가 필요합니다 (예: 16,32,64,128,256)"
        model.rpn.anchor_generator = AnchorGenerator(sizes=(tuple(int(s) for s in anchor_sizes),) * 3, aspect_ratios=((0.5, 1.0, 2.0),) * 3)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 배경 + defect
    return model


build_model.description = "fasterrcnn_mobilenet_v3_large_fpn(COCO_V1)"
build_model.short_name = "Faster R-CNN MobileNetV3-FPN"
