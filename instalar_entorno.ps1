# Crea el entorno de Python del proyecto en una PC nueva (necesita internet, ~1 GB de descarga).
# Uso, desde la carpeta Proyecto_LSP:  powershell -ExecutionPolicy Bypass -File .\instalar_entorno.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (Test-Path .\.venv_fusion\Scripts\python.exe) {
    Write-Host 'El entorno .venv_fusion ya existe; no se reinstala.'
} else {
    python -m venv .venv_fusion
    .\.venv_fusion\Scripts\python.exe -m pip install --upgrade pip
    .\.venv_fusion\Scripts\python.exe -m pip install -r .\requirements_fusion.txt
}
.\.venv_fusion\Scripts\python.exe -c "import torch, torchvision, numpy; print('OK torch', torch.__version__, '| numpy', numpy.__version__)"
