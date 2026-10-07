# PowerPoint 로 presentation.pptx 를 PDF 와 슬라이드별 PNG 로 내보낸다 (PowerPoint 가 설치된 Windows 에서만).
#   powershell -ExecutionPolicy Bypass -File ppt\export_pdf.ps1 [-PngDir <폴더>]
# PNG 는 레이아웃 점검용이다(제출물은 PDF).
param([string]$PngDir = "")
$root = Split-Path -Parent $PSScriptRoot
$pptx = Join-Path $root "ppt\presentation.pptx"
$pdf = Join-Path $root "ppt\presentation.pdf"
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open($pptx, $true, $false, $false)   # 읽기 전용, 제목 없음, 창 없음
    $pres.SaveAs($pdf, 32)                                           # 32 = ppSaveAsPDF
    if ($PngDir -ne "") {
        New-Item -ItemType Directory -Force $PngDir | Out-Null
        $pres.Export($PngDir, "PNG", 1280, 720)
    }
    $pres.Close()
} finally {
    $app.Quit()
}
Write-Output "PDF: $pdf"
