"""학습 전 준비: COCO 사전학습 가중치를 받는다 (학습할 때만 필요, 제출 가중치로 추론·채점만 할 때는 필요 없음).

    python scripts/download_pretrained.py

- D-FINE-N : Hugging Face ustc-community/dfine-nano-coco (Apache-2.0). Hugging Face 캐시에 받는다
             (학습 시 자동으로도 받으므로 미리 받아 두는 용도).
- YOLOv3-tiny : ultralytics yolov3(2020) 코드가 쓰던 COCO 사전학습 yolov3-tiny.pt 를 weights/pretrained/ 에 받는다.
               원본 코드(third_party/yolov3/models.py attempt_download)와 같은 Google Drive 파일이다.
               받지 못하면 weights/README.md 의 안내대로 파일을 직접 넣는다.
"""
import _bootstrap  # noqa: F401

import hashlib
import http.cookiejar
import re
import urllib.request

from kamp_xray.common import CONFIGS, ROOT, WEIGHTS, load_yaml

YOLOV3_TINY_GDRIVE_ID = "10m_3MlpQwRtZetQxtksm9jqHrPTHZ6vo"   # third_party/yolov3/models.py 의 'yolov3-tiny.pt'


def gdrive_download(file_id, dst):
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    data = opener.open(url, timeout=60).read()
    if data[:1] == b"<":                      # 큰 파일은 확인 페이지가 먼저 온다
        html = data.decode("utf-8", "ignore")
        m = re.search(r'name="uuid" value="([^"]+)"', html)
        confirm = re.search(r'confirm=([0-9A-Za-z_-]+)', html)
        url2 = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=" + \
               (confirm.group(1) if confirm else "t") + (f"&uuid={m.group(1)}" if m else "")
        data = opener.open(url2, timeout=300).read()
    if data[:1] == b"<" or len(data) < 1_000_000:
        raise RuntimeError("Google Drive 에서 받지 못했습니다 (확인 페이지 또는 접근 제한).")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main():
    hyp_d = load_yaml(CONFIGS / "dfine_n.yaml")
    from huggingface_hub import snapshot_download
    path = snapshot_download(hyp_d["pretrained"])
    print(f"D-FINE-N 사전학습: {hyp_d['pretrained']} → Hugging Face 캐시 ({path})")

    hyp_y = load_yaml(CONFIGS / "yolov3_tiny.yaml")
    dst = ROOT / hyp_y["pretrained"]
    if dst.exists():
        print(f"YOLOv3-tiny 사전학습: 이미 있음 ({dst.relative_to(ROOT)})")
        return
    try:
        digest = gdrive_download(YOLOV3_TINY_GDRIVE_ID, dst)
        print(f"YOLOv3-tiny 사전학습: {dst.relative_to(ROOT)} ({dst.stat().st_size / 1e6:.1f} MB, sha256 {digest[:16]}...)")
        return
    except Exception as e:
        print(f"Google Drive 원본 링크 실패 ({e}). Darknet 공식 COCO 가중치를 받아 .pt 로 변환합니다.")
    try:
        darknet_to_pt(dst)
    except Exception as e:
        print(f"YOLOv3-tiny 사전학습 가중치를 준비하지 못했습니다: {e}\n"
              f"  {WEIGHTS / 'README.md'} 의 안내대로 yolov3-tiny.pt 를 {dst} 에 넣어 주세요.")
        raise SystemExit(1)


DARKNET_URL = "https://pjreddie.com/media/files/yolov3-tiny.weights"   # third_party/yolov3/models.py 의 대체 경로와 같음


def darknet_to_pt(dst):
    """Darknet yolov3-tiny.weights(COCO 80클래스) → third_party/yolov3 형식의 .pt.
    1클래스 cfg 를 80클래스로 되돌린 임시 cfg 로 읽고, 학습 시 train.py 가 크기가 다른 출력층만 빼고 불러온다."""
    import sys
    import tempfile
    from pathlib import Path

    import torch
    from kamp_xray.common import YOLOV3_DIR
    sys.path.insert(0, str(YOLOV3_DIR))
    from models import Darknet, load_darknet_weights

    raw = dst.with_suffix(".weights")
    if not raw.exists():
        req = urllib.request.Request(DARKNET_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=300).read()
        if len(data) < 30_000_000:
            raise RuntimeError(f"받은 파일이 너무 작습니다 ({len(data)} bytes)")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(data)
    cfg = (YOLOV3_DIR / "yolov3-tiny.cfg").read_text(encoding="utf-8").replace("filters=18", "filters=255") \
        .replace("classes=1", "classes=80")
    with tempfile.TemporaryDirectory() as td:
        coco_cfg = Path(td) / "yolov3-tiny-coco.cfg"
        coco_cfg.write_text(cfg, encoding="utf-8")
        model = Darknet(str(coco_cfg))
        load_darknet_weights(model, str(raw))
    torch.save({"epoch": -1, "best_fitness": None, "training_results": None, "model": model.state_dict(),
                "optimizer": None}, dst)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    print(f"YOLOv3-tiny 사전학습: {DARKNET_URL} (sha256 {digest[:16]}...) → {dst.relative_to(ROOT)} 변환 완료")


if __name__ == "__main__":
    main()
