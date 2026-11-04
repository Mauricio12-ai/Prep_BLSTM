# Evalúa un modelo ya entrenado (sin reentrenar) y abre su matriz de confusión.
# Uso, desde la carpeta Proyecto_LSP:
#   powershell -ExecutionPolicy Bypass -File .\demo_evaluar.ps1
#   powershell -ExecutionPolicy Bypass -File .\demo_evaluar.ps1 -Experimento .\resultados\fusion\OTRA_CARPETA
param([string]$Experimento = '')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$py        = '.\.venv_fusion\Scripts\python.exe'
$indice    = '.\datos\indices\union_20261012_153325\indice_comun.csv'
$landmarks = '.\datos\procesados\landmarks\datos_procesados_20261005_160031\landmarks_32f.npz'
$features  = '.\datos\caracteristicas\resnet18'
if (-not $Experimento) {
    # El entrenamiento de fusión más reciente que tenga mejor_modelo.pt
    $Experimento = (Get-ChildItem .\resultados\fusion -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'mejor_modelo.pt') } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
Write-Host "Evaluando: $Experimento"
& $py .\codigo\modelos\cnn_bilstm\evaluar_checkpoint.py --experimento "$Experimento" --indice $indice --landmarks $landmarks --features $features
if ($LASTEXITCODE -ne 0) { throw 'La evaluación falló' }
$t = Get-ChildItem (Join-Path $Experimento 'test_*') -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Start-Process (Join-Path $t.FullName 'matriz_test.html')
