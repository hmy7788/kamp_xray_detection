"""대회 제출용 소스코드 ZIP 을 만든다 (팀 내부용 도구).

    python tools/make_submission_zip.py                 # dist/엽총창_소스코드.zip

포함: Git 이 추적하는 파일 전체(코드, data/, results/, 문서) + Git 에서 제외한 제출물
      (weights/dfine_n/best/*, weights/yolov3_tiny/best.pt, results/**/preds_*.json)
제외: outputs/, weights/pretrained/ (학습 시 scripts/download_pretrained.py 로 받음), 캐시
제출 전 확인: 최종 모델 가중치와 test 예측 결과가 있는지, 개인 PC 경로가 남아 있지 않은지 검사한다.
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["weights/dfine_n/best/model.safetensors", "weights/dfine_n/best/config.json",
            "weights/dfine_n/best/preprocessor_config.json", "weights/yolov3_tiny/best.pt",
            "results/dfine_n/test_predictions.csv", "results/dfine_n/eval_report_test.json",
            "results/yolov3_tiny/eval_report_test.json", "requirements.txt", "README.md"]
PRIVATE = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+|/home/[a-z]|OneDrive", re.IGNORECASE)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        sys.exit("제출에 필요한 파일이 없습니다:\n  " + "\n  ".join(missing))
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True).stdout.decode("utf-8").split("\0")
    files = {p for p in tracked if p and (ROOT / p).is_file()}
    files |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "weights").rglob("*")
              if p.is_file() and "pretrained" not in p.parts}
    files |= {p.relative_to(ROOT).as_posix() for p in (ROOT / "results").rglob("preds_*.json")}
    files = sorted(f for f in files if not f.startswith(("outputs/", "dist/", "tools/")))

    leaks = []
    for f in files:
        if f.endswith((".py", ".md", ".json", ".csv", ".log", ".yaml", ".txt", ".ps1", ".sh", ".bat")) and \
                (ROOT / f).stat().st_size < 20_000_000:
            if PRIVATE.search((ROOT / f).read_text(encoding="utf-8", errors="ignore")):
                leaks.append(f)
    if leaks:
        sys.exit("개인 PC 경로가 남은 파일이 있습니다 (지운 뒤 다시 실행):\n  " + "\n  ".join(leaks))

    out = ROOT / "dist" / "엽총창_소스코드.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(ROOT / f, f"kamp_xray_yeopchongchang/{f}")
    print(f"{out}  ({len(files)}개 파일, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
