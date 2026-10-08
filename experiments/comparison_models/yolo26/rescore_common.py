"""YOLO26n 가중치로 val, test 를 다시 추론해 공통 채점 코드로 채점한다 (다른 모델과 같은 규칙으로 비교하기 위함).

    python experiments/comparison_models/yolo26/rescore_common.py --weights <best.pt> --imgsz 640 --out results/comparison_models/yolo26n_img640

필요: ultralytics. 신뢰도 0.001 이상, 최대 300개 검출을 저장하고, val F1 최대 임계값을 test 에 그대로 적용한다.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from kamp_xray import metrics as M  # noqa: E402
from kamp_xray.common import DATA, check_data_version, imread, make_records, read_manifest, save_json, setup_console, split_ids  # noqa: E402


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    check_data_version()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    man = read_manifest()
    out = ROOT / a.out
    thr = None
    for split in ("val", "test"):
        ids = split_ids(man, split)
        dets = []
        for i in ids:
            r = model.predict(imread(DATA / man[i]["image_path"]), imgsz=a.imgsz, conf=0.001, max_det=300,
                              device=a.device, verbose=False)[0]
            dets.append([[*map(float, b), float(c)] for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())])
        recs = make_records(man, ids, dets)
        if split == "val":
            thr = M.best_threshold(recs)
        rep = {"model": f"YOLO26n imgsz {a.imgsz}", "split": split, **M.report(recs, thr, man)}
        save_json(out / f"common_eval_report_{split}.json", rep)
        o = rep["overall"]
        print(f"{split}: thr {thr:.4f} AP50 {o['ap50']} mAP50-95 {o['map50_95']} P {o['precision']} R {o['recall']} F1 {o['f1']}")


if __name__ == "__main__":
    main()
