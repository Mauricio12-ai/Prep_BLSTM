# Reconocimiento de palabras de la Lengua de Señas Peruana con CNN-BiLSTM multimodal

Clasificación de **34 palabras aisladas de la Lengua de Señas Peruana (LSP)**, con énfasis en *minimal pairs*
(señas que difieren en un solo parámetro articulatorio). La arquitectura combina dos modalidades:

- **RGB:** cada frame pasa por una **ResNet18 preentrenada y congelada** (512 características por frame).
- **Landmarks:** 75 puntos por frame (33 de pose y 21 por mano), con coordenadas XY normalizadas y máscaras.

Cada modalidad se modela con un **BiLSTM** y ambas se **fusionan por concatenación** antes del clasificador.
El proyecto forma parte de una comparación con una arquitectura ST-GCN (Seminario de Investigación I,
Ingeniería de Sistemas, Universidad de Lima).

```
RGB 32×112×112 → ResNet18 congelada → (32, 512) → BiLSTM → 128 ┐
                                                                ├→ concat 256 → Dropout 0.3 → Lineal → 34 palabras
Landmarks 32×300 (XY + máscaras) ──────────────→ BiLSTM → 128 ┘
```

## Estructura

| Carpeta / archivo | Contenido |
|---|---|
| `codigo/preprocesamiento/landmarks/` | `preprocess_landmarks_c3.py`: imputación de huecos cortos, normalización por hombros, muestreo a 32 frames y máscaras |
| `codigo/preprocesamiento/rgb_landmarks/` | Preparación conjunta de RGB y landmarks desde los ZIP del dataset |
| `codigo/preprocesamiento/union/` | `unir_rgb_landmarks.py`: índice común RGB–landmarks (4388 muestras) |
| `codigo/modelos/cnn_bilstm/` | `extraer_rgb.py` (ResNet18), `modelo.py`, `entrenar_fusion.py`, `evaluar_checkpoint.py` |
| `codigo/modelos/bilstm/` | Baseline BiLSTM solo con landmarks |
| `documentacion/` | Guía completa del proyecto y notas de organización |
| `datos/`, `resultados/` | Solo la estructura de carpetas: datos, pesos y resultados no se publican |
| `instalar_entorno.ps1`, `demo_evaluar.ps1` | Instalación del entorno y evaluación rápida de un modelo entrenado |

El dataset no se distribuye en este repositorio.

## Instalación (Windows, PowerShell)

Requiere Python 3.12 o 3.13. Desde la carpeta del proyecto:

```powershell
powershell -ExecutionPolicy Bypass -File .\instalar_entorno.ps1
```

Instala PyTorch 2.14.1 y torchvision 0.29.1 (versión CPU) y NumPy en `.venv_fusion` (ver `requirements_fusion.txt`).

## Uso

Con los datos procesados dentro de `datos/`:

```powershell
$py        = ".\.venv_fusion\Scripts\python.exe"
$indice    = ".\datos\indices\union_20261012_153325\indice_comun.csv"
$landmarks = ".\datos\procesados\landmarks\datos_procesados_20261005_160031\landmarks_32f.npz"
$features  = ".\datos\caracteristicas\resnet18"

# 1. Extraer características RGB con ResNet18 (una sola vez)
& $py .\codigo\modelos\cnn_bilstm\extraer_rgb.py --indice $indice --salida $features

# 2. Entrenar la fusión con la partición por sujetos
$run = ".\resultados\fusion\ejecucion_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $py .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice $indice --landmarks $landmarks --features $features `
    --salida $run --modo fusion --epochs 50 --patience 10 --test-subjects 5 14 --val-subjects 1 7 8

# 3. Evaluar una sola vez en test (sin reentrenar)
& $py .\codigo\modelos\cnn_bilstm\evaluar_checkpoint.py --experimento $run
```

- `--modo` acepta `fusion`, `rgb` (solo CNN-BiLSTM visual) o `landmarks` (solo BiLSTM).
- La mejor época se elige por **macro-F1 de validación**, con parada temprana; el test no interviene en la selección.
- Cada ejecución guarda `config.json`, `particiones.json`, `historial.csv`, `mejor_modelo.pt`, métricas,
  predicciones y la matriz de confusión (`matriz_test.html`).

## Protocolo y resultados

Partición *cross-subject* (ningún firmante se repite entre conjuntos): entrenamiento con 12 sujetos (2813 muestras),
validación con los sujetos 1, 7 y 8 (988) y prueba con los sujetos 5 y 14 (587).

| Modelo | F1-macro validación | Exactitud test | F1-macro test |
|---|---|---|---|
| CNN-BiLSTM fusión (RGB + landmarks) | 0.688 | 53.3 % | 0.492 |

Es una única partición, no LOSO completo. Con cerca de 17 muestras por palabra en test, las métricas por clase
son sensibles a pocos errores.

## Limitaciones

- ResNet18 congelada con entrada de 112 × 112: puede perder detalle fino de los dedos.
- La profundidad (depth) no se utiliza.
- La lista de *minimal pairs* debe validarse lingüísticamente antes de interpretar sus confusiones.
