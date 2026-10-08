"""클릭 → 고정 크기 네모. Label Studio 의 "interactive pre-annotation" 보조 서버.

실행:  venv_ml/Scripts/python fixed_box_backend.py   (포트 9090)
동작:  사람이 사진에서 점(KeyPoint)을 찍으면 Label Studio 가 그 좌표를 여기로 보내고,
       여기서 점을 중심으로 한 변 10px × (사진 높이/332) 네모(RectangleLabels)를 돌려준다.
       1·2호기(높이 332) 10px, 3호기(높이 444) 13px. 공식 라벨 중앙값과 같다.
"""
from __future__ import annotations

import logging

from label_studio_ml.api import init_app
from label_studio_ml.model import LabelStudioMLBase

BOX_PX_AT_332 = 10.0
LABEL = "defect"


class FixedBox(LabelStudioMLBase):
    def predict(self, tasks, **kwargs):
        context = kwargs.get("context") or {}
        results = context.get("result") or []
        out = []
        for r in results:
            if r.get("type") not in ("keypointlabels", "keypoint"):
                continue
            W, H = r.get("original_width"), r.get("original_height")
            v = r.get("value", {})
            if not W or not H or "x" not in v:
                continue
            side = BOX_PX_AT_332 * (H / 332.0)
            w, h = side / W * 100, side / H * 100
            out.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": W,
                "original_height": H,
                "image_rotation": 0,
                "value": {"x": v["x"] - w / 2, "y": v["y"] - h / 2, "width": w, "height": h,
                          "rotation": 0, "rectanglelabels": [LABEL]},
            })
        if not out:          # 클릭 없이 사진만 열렸을 때는 아무 예측도 주지 않는다 (빈 탭이 생기지 않게)
            return []
        return [{"result": out, "score": 1.0, "model_version": "fixed_box"} for _ in tasks]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os
    os.makedirs("ml_data", exist_ok=True)
    app = init_app(model_class=FixedBox, model_dir=os.path.abspath("ml_data"))
    app.run(host="0.0.0.0", port=9090, debug=False)
