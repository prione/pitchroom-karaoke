param(
    [string]$PythonHome = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildRoot = Join-Path $projectRoot "build"
$pythonTarget = Join-Path $buildRoot "python-runtime"
$runtimeTarget = Join-Path $buildRoot "runtime"

function Assert-BuildTarget([string]$Target) {
    $full = [System.IO.Path]::GetFullPath($Target)
    $expected = [System.IO.Path]::GetFullPath($buildRoot) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($expected, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe build target: $full"
    }
}

function Reset-BuildDirectory([string]$Target) {
    Assert-BuildTarget $Target
    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
}

function Copy-Tree([string]$Source, [string]$Target) {
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
    & robocopy.exe $Source $Target /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD __pycache__ tests test | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed ($LASTEXITCODE): $Source"
    }
}

if (-not $PythonHome) {
    $venvConfig = Get-Content -LiteralPath (Join-Path $projectRoot ".venv\pyvenv.cfg") -Encoding utf8
    $homeLine = $venvConfig | Where-Object { $_ -match '^home\s*=\s*' } | Select-Object -First 1
    if (-not $homeLine) { throw "Python home was not found in .venv\pyvenv.cfg" }
    $PythonHome = ($homeLine -replace '^home\s*=\s*', '').Trim()
}
$PythonHome = (Resolve-Path -LiteralPath $PythonHome).Path
$sitePackages = (Resolve-Path -LiteralPath (Join-Path $projectRoot ".venv\Lib\site-packages")).Path

Reset-BuildDirectory $pythonTarget
Copy-Tree $PythonHome $pythonTarget
Copy-Tree $sitePackages (Join-Path $pythonTarget "Lib\site-packages")

Reset-BuildDirectory $runtimeTarget
$runtimeScripts = Join-Path $runtimeTarget "scripts"
New-Item -ItemType Directory -Path $runtimeScripts -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot "scripts\extract-melodia.cjs") -Destination $runtimeScripts

$nodeExecutable = (Get-Command node -ErrorAction SilentlyContinue).Source
if (-not $nodeExecutable) {
    throw "Node.js was not found. Install Node.js 22 or later."
}
$essentiaEntry = (& $nodeExecutable -e "console.log(require.resolve('essentia.js'))" | Select-Object -Last 1).Trim()
$nodeWavEntry = (& $nodeExecutable -e "console.log(require.resolve('node-wav',{paths:[require.resolve('essentia.js')]}))" | Select-Object -Last 1).Trim()
$essentiaRoot = Split-Path -Parent $essentiaEntry
$nodeWavRoot = Split-Path -Parent $nodeWavEntry
Copy-Tree $essentiaRoot (Join-Path $runtimeTarget "node_modules\essentia.js")
Copy-Tree $nodeWavRoot (Join-Path $runtimeTarget "node_modules\node-wav")
Copy-Tree (Join-Path $projectRoot "node_modules\@spotify\basic-pitch\model") (Join-Path $runtimeTarget "basic-pitch-model")

$portablePython = Join-Path $pythonTarget "python.exe"
& $portablePython -c "import fastapi, torch, torchaudio, demucs, yt_dlp; print('portable Python runtime OK')"
if ($LASTEXITCODE -ne 0) { throw "Portable Python runtime verification failed" }

Write-Output "Prepared Python runtime: $pythonTarget"
Write-Output "Prepared application runtime: $runtimeTarget"
