#!/usr/bin/env bash
# 전체 파이프라인 자동 실행 (Linux / macOS / Git Bash)
#
#   bash run_all.sh                # 제출 가중치로 데이터 검사 → 추론 → 채점 → 속도 → 비교표
#   bash run_all.sh --train        # 처음부터 학습까지
#   DEVICE=cpu bash run_all.sh     # GPU 없이 실행
#   SKIP_SPEED=1 bash run_all.sh   # 속도 측정 생략
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
TRAIN=0
[ "${1:-}" = "--train" ] && TRAIN=1

step() { echo; echo "==== $1"; shift; echo "     $PY $*"; "$PY" "$@"; }

MODELS="yolov3_tiny dfine_n"   # 베이스라인, 최종 모델
step "1. 데이터 검사" scripts/check_data.py

weights() {
  if [ "$TRAIN" = 1 ]; then
    [ "$1" = dfine_n ] && echo outputs/dfine_n/train/weights/best || echo outputs/yolov3_tiny/train/weights/best.pt
  else
    [ "$1" = dfine_n ] && echo weights/dfine_n/best || echo weights/yolov3_tiny/best.pt
  fi
}

if [ "$TRAIN" = 1 ]; then
  step "2. 사전학습 가중치 준비" scripts/download_pretrained.py
  for m in $MODELS; do step "3. 학습: $m" scripts/train.py --model "$m" --device "$DEVICE"; done
fi

for m in $MODELS; do
  for s in val test; do
    step "4. 추론: $m $s" scripts/predict.py --model "$m" --split "$s" --weights "$(weights "$m")" --device "$DEVICE"
    step "5. 채점: $m $s" scripts/evaluate.py --model "$m" --split "$s"
  done
  [ -z "${SKIP_SPEED:-}" ] && step "6. 속도 (CPU 4스레드): $m" scripts/speed.py --model "$m" --weights "$(weights "$m")"
done

step "7. 모델 비교표" scripts/compare.py --root outputs
[ "$TRAIN" = 0 ] && step "8. 제출 결과(results/)와 비교" scripts/compare.py --check
echo; echo "완료. 최종 모델 test 예측 결과: outputs/dfine_n/test_predictions.csv"
