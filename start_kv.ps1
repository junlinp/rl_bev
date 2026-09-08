<# 
.SYNOPSIS
    Start minikeyvalue server for BEV dataset storage.

.PARAMETER Port
    Master server port (default 3000).

.PARAMETER VolumePort
    Volume server port (default 3001).

.PARAMETER DataDir
    Directory for storage (default .\kv_data).
#>
param(
    [int]$Port = 3000,
    [int]$VolumePort = 3001,
    [string]$DataDir = ".\kv_data"
)

$ErrorActionPreference = "Stop"

$volDir = Join-Path $DataDir "vol1"
$indexDir = Join-Path $DataDir "indexdb"
New-Item -ItemType Directory -Force -Path $volDir | Out-Null
New-Item -ItemType Directory -Force -Path $indexDir | Out-Null

$mkv = Join-Path $PSScriptRoot "mkv.exe"
if (-not (Test-Path $mkv)) {
    Write-Error "mkv.exe not found. Build it first: cd mkv_repo\src; go build -o ..\..\mkv.exe ."
    exit 1
}

Write-Host "[minikeyvalue] Volume: :$VolumePort → $volDir"
Write-Host "[minikeyvalue] Master: :$Port → $indexDir"
Write-Host "[minikeyvalue] Ctrl+C to stop"

# Start volume server (nginx-based, use mkv volume mode)
# On Windows, minikeyvalue's volume is a shell script wrapping nginx.
# For simplicity, we run the master in single-node mode with local volume.
Start-Process -FilePath $mkv -ArgumentList `
    "-port", $Port, `
    "-db", $indexDir, `
    "-volumes", "localhost:$VolumePort", `
    "server" -NoNewWindow

Write-Host "[minikeyvalue] Master started on :$Port"
Write-Host "[minikeyvalue] NOTE: Volume server (nginx) needs to run on :$VolumePort"
Write-Host "[minikeyvalue] For single-machine testing, use: python collect_data.py --kv-url http://localhost:$Port"
