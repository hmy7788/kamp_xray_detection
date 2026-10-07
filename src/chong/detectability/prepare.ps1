# Prepare a local data view without editing shared data or exposing test images.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$sourceData = Join-Path $repo 'data'
$view = Join-Path $repo 'runs/chong/local/work/data_view'
$harness = Join-Path $repo 'src/chong/harness'
$expected = '1942bf3452defc08022b11f464092d46b095ea2bb31961b15b072bc9d623c623'
$bytes = [IO.File]::ReadAllBytes((Join-Path $sourceData 'manifest.csv'))
$utf8 = [Text.UTF8Encoding]::new($false)
$canonical = $utf8.GetBytes($utf8.GetString($bytes).Replace("`r`n", "`n"))
$sha = [Security.Cryptography.SHA256]::Create()
$actual = [BitConverter]::ToString($sha.ComputeHash($canonical)).Replace('-', '').ToLowerInvariant()
if ($actual -ne $expected) { throw 'Manifest content differs from the frozen version.' }
New-Item -ItemType Directory -Force -Path $view | Out-Null
[IO.File]::WriteAllBytes((Join-Path $view 'manifest.csv'), $canonical)
foreach ($name in @('manifest.sha256', 'PASS', 'split_info.json', 'conditions.csv', 'conditions_thresholds.json')) {
    Copy-Item -LiteralPath (Join-Path $sourceData $name) -Destination (Join-Path $view $name) -Force
}
foreach ($part in @('train', 'val')) {
    $link = Join-Path $view $part
    if (-not (Test-Path -LiteralPath $link)) {
        New-Item -ItemType Junction -Path $link -Target (Join-Path $sourceData $part) | Out-Null
    }
}
$dataLink = Join-Path $harness 'data'
if (Test-Path -LiteralPath $dataLink) {
    $item = Get-Item -LiteralPath $dataLink
    if ($item.LinkType -ne 'Junction') { throw 'harness/data exists and is not a junction; refusing to replace it.' }
    # Nonrecursive: removes only the junction, never its destination.
    [IO.Directory]::Delete($dataLink)
}
New-Item -ItemType Junction -Path $dataLink -Target $view | Out-Null
$runsLink = Join-Path $harness 'runs'
if (-not (Test-Path -LiteralPath $runsLink)) {
    New-Item -ItemType Junction -Path $runsLink -Target (Join-Path $repo 'runs/chong') | Out-Null
}
Write-Output 'Prepared train/val-only data view. Original data is unchanged.'
