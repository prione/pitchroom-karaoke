$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$requirements = Join-Path $projectRoot "local_backend\requirements.txt"
if (-not (Test-Path -LiteralPath $venvPython)) {
    $systemPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $systemPython -or -not (Test-Path -LiteralPath $systemPython)) {
        throw "Python 3.12 was not found. Install Python and try again."
    }
    & $systemPython -m venv (Join-Path $projectRoot ".venv")
}

& $venvPython -c "import demucs, torchcrepe, fastapi, yt_dlp, imageio_ffmpeg, piano_transcription_inference" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $venvPython -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Failed to install the audio analysis dependencies." }
}

$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
$nodeExe = if ($nodeCommand) { $nodeCommand.Source } else { "" }
if (-not (Test-Path -LiteralPath $nodeExe)) {
    throw "Node.js was not found. Install Node.js 22 or later."
}
$env:PITCHROOM_NODE_PATH = $nodeExe
$env:PATH = "$([System.IO.Path]::GetDirectoryName($nodeExe));$env:PATH"

$electronExe = Join-Path $projectRoot "node_modules\electron\dist\electron.exe"
if (-not (Test-Path -LiteralPath $electronExe)) {
    $pnpmCommand = Get-Command pnpm -ErrorAction SilentlyContinue
    $pnpmExe = if ($pnpmCommand) { $pnpmCommand.Source } else { "" }
    if (-not (Test-Path -LiteralPath $pnpmExe)) { throw "pnpm was not found." }
    & $pnpmExe install
    if (-not (Test-Path -LiteralPath $electronExe)) {
        & $nodeExe (Join-Path $projectRoot "node_modules\electron\install.js")
    }
}

$distEntry = Join-Path $projectRoot "dist-renderer\index.html"
$uiFiles = Get-ChildItem -Path (Join-Path $projectRoot "app") -Recurse -File
$latestUiWrite = ($uiFiles | Measure-Object -Property LastWriteTimeUtc -Maximum).Maximum
$needsBuild = -not (Test-Path -LiteralPath $distEntry)
if (-not $needsBuild) {
    $needsBuild = $latestUiWrite -gt (Get-Item -LiteralPath $distEntry).LastWriteTimeUtc
}
if ($needsBuild) {
    & $nodeExe (Join-Path $projectRoot "node_modules\vite\bin\vite.js") build --config vite.electron.config.ts
    if ($LASTEXITCODE -ne 0) { throw "Failed to build the PitchRoom interface." }
}

& $electronExe $projectRoot
