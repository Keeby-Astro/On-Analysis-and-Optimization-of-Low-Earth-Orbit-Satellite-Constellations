# Sequentially generate 3D orbit animations for each constellation type.
# Usage: powershell -ExecutionPolicy Bypass -File .\run_all_constellations.ps1

$ErrorActionPreference = 'Continue'

Set-Location -Path 'C:\Users\PC\Code\UMAP_HDBSCAN'

$python = 'python'
$script = '3d_orbits_animation.py'

# Common args shared by all jobs
$commonArgs = @(
    '--gif-only',
    '--gif',
    '--gif-frames', '240',
    '--gif-fps', '60',
    '--gif-dpi', '300',
    '--gif-width-in', '4.0',
    '--gif-height-in', '4.0',
    '--gif-earth-grid', '180',
    '--gif-workers', '4',
    '--output-dir', '3d_orbits_plot'
)

# Per-job args. mega-constellation does not take --count / --walker-planes.
$jobs = @(
    @{ Name = 'mega-constellation';  Extra = @('--constellation', 'mega-constellation') }
)

$results = @()

foreach ($job in $jobs) {
    $allArgs = @($script) + $job.Extra + $commonArgs
    Write-Host ''
    Write-Host ('=' * 80)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Running: $($job.Name)" -ForegroundColor Cyan
    Write-Host ('=' * 80)
    Write-Host "$python $($allArgs -join ' ')" -ForegroundColor DarkGray

    $start = Get-Date
    & $python @allArgs
    $exit = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    $status = if ($exit -eq 0) { 'OK' } else { "FAIL ($exit)" }
    $color  = if ($exit -eq 0) { 'Green' } else { 'Red' }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $($job.Name): $status in $([int]$elapsed.TotalSeconds)s" -ForegroundColor $color

    $results += [pscustomobject]@{
        Constellation = $job.Name
        Status        = $status
        ElapsedSec    = [int]$elapsed.TotalSeconds
    }
}

Write-Host ''
Write-Host ('=' * 80)
Write-Host 'Summary' -ForegroundColor Yellow
Write-Host ('=' * 80)
$results | Format-Table -AutoSize
