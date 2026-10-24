$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create a Python 3.12 virtual environment and install requirements.txt first. See README.md.'
}
& $projectPython -u train.py
if ($LASTEXITCODE -ne 0) { throw 'The project command failed. See the error above.' }
