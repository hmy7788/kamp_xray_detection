# 전체 파이프라인 자동 실행 (Windows PowerShell)
#
#   .\run_all.ps1                 # 제출 가중치로 데이터 검사 → 추론 → 채점 → 속도 → 비교표 (GPU 기준 약 5분)
#   .\run_all.ps1 -Train          # 처음부터 학습까지 (데이터 검사 → 사전학습 가중치 → 학습 → 추론 → 채점 → 속도 → 비교표, 약 3시간)
#   .\run_all.ps1 -Device cpu     # GPU 없이 실행
#   .\run_all.ps1 -SkipSpeed      # 속도 측정 생략
#
# 결과는 outputs\ 에 생기고, 제출 시점 결과(results\)와 비교한 내용이 마지막에 출력된다.
param(
    [switch]$Train,
    [switch]$SkipSpeed,
    [string]$Device = "0",
    [string]$Python = "python"
)
$ErrorActionPreference = "Continue"   # 파이썬 경고 메시지로 멈추지 않게 하고, 각 단계의 종료 코드로 실패를 판단한다
$env:PYTHONUTF8 = "1"
Set-Location -Path $PSScriptRoot

function Step([string]$title, [string[]]$cmd) {
    Write-Host ""
    Write-Host "==== $title" -ForegroundColor Cyan
    Write-Host "     $Python $($cmd -join ' ')"
    & $Python @cmd
    if ($LASTEXITCODE -ne 0) { throw "실패: $title (종료 코드 $LASTEXITCODE)" }
}

$models = @("yolov3_tiny", "dfine_n")   # 베이스라인, 최종 모델
Step "1. 데이터 검사" @("scripts/check_data.py")

if ($Train) {
    Step "2. 사전학습 가중치 준비" @("scripts/download_pretrained.py")
    foreach ($m in $models) {
        Step "3. 학습: $m" @("scripts/train.py", "--model", $m, "--device", $Device)
    }
    $w = @{ "yolov3_tiny" = "outputs/yolov3_tiny/train/weights/best.pt"; "dfine_n" = "outputs/dfine_n/train/weights/best" }
} else {
    $w = @{ "yolov3_tiny" = "weights/yolov3_tiny/best.pt"; "dfine_n" = "weights/dfine_n/best" }
}

foreach ($m in $models) {
    foreach ($s in @("val", "test")) {
        Step "4. 추론: $m $s" @("scripts/predict.py", "--model", $m, "--split", $s, "--weights", $w[$m], "--device", $Device)
        Step "5. 채점: $m $s" @("scripts/evaluate.py", "--model", $m, "--split", $s)
    }
    if (-not $SkipSpeed) {
        Step "6. 속도 (CPU 4스레드): $m" @("scripts/speed.py", "--model", $m, "--weights", $w[$m])
    }
}

Step "7. 모델 비교표" @("scripts/compare.py", "--root", "outputs")
if (-not $Train) {
    Step "8. 제출 결과(results/)와 비교" @("scripts/compare.py", "--check")
}
Write-Host ""
Write-Host "완료. 최종 모델 test 예측 결과: outputs\dfine_n\test_predictions.csv" -ForegroundColor Green
