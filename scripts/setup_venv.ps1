#!/usr/bin/env pwsh
# PowerShell helper to create/recreate the project venv using the project's Python

param(
    [switch]$Recreate,
    [string]$Python = $null
)

$script = Join-Path -Path (Split-Path -Parent $MyInvocation.MyCommand.Definition) -ChildPath 'create_venv.py'
$args = @('--requirements', '..\requirements.txt')
if ($Recreate) { $args += '--recreate' }
if ($Python) { $args = @('--python', $Python) + $args }

Write-Host "Running: python $script $($args -join ' ')"
python $script $args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
