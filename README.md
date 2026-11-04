# lsp-cnn-bilstm

Código de un framework CNN-BiLSTM con fusión multimodal (RGB + landmarks) para reconocer palabras aisladas
de la Lengua de Señas Peruana.

```
RGB (32 frames) → ResNet18 congelada → BiLSTM ┐
                                              ├→ concatenación → Dropout → Lineal → palabra
Landmarks (32 frames, XY + máscaras) → BiLSTM ┘
```

El dataset es privado y no se incluye en este repositorio.

## Scripts

| Script | Qué hace |
|---|---|
| `codigo/preprocesamiento/rgb_depth/procesar_c3.py` | Valida cada muestra, toma 32 frames y aplica letterbox a 112 × 112 |
| `codigo/preprocesamiento/landmarks/preprocess_landmarks_c3.py` | Interpola huecos cortos, normaliza por hombros, toma 32 frames y crea máscaras |
| `codigo/preprocesamiento/union/unir_rgb_landmarks.py` | Verifica que RGB y landmarks correspondan y crea el índice común |
| `codigo/modelos/cnn_bilstm/extraer_rgb.py` | Extrae 512 características por frame con ResNet18 congelada |
| `codigo/modelos/cnn_bilstm/modelo.py` | Arquitectura: dos BiLSTM, fusión y clasificador (modos `fusion`, `rgb` y `landmarks`) |
| `codigo/modelos/cnn_bilstm/entrenar_fusion.py` | Partición por sujetos, entrenamiento con parada temprana y métricas |
| `codigo/modelos/cnn_bilstm/evaluar_checkpoint.py` | Evalúa un modelo guardado en los sujetos de prueba, en total y por sujeto |
| `codigo/modelos/cnn_bilstm/demo_muestra.py` | Recorre una muestra por el modelo y muestra cada dimensión |
| `codigo/modelos/cnn_bilstm/comparar_variantes.py` | Tabla de prueba de las tres variantes |
| `codigo/modelos/cnn_bilstm/grupos_pares_minimos.py` | Aciertos y confusiones por grupo de pares mínimos |
| `notebooks/LSP_CNN_BiLSTM_Colab.ipynb` | Ejecuta el framework en Google Colab |

## Entorno

```powershell
powershell -ExecutionPolicy Bypass -File .\instalar_entorno.ps1
```

Crea `.venv_fusion` e instala las dependencias de `requirements_fusion.txt` (PyTorch, torchvision y NumPy).

## Ejemplo

```powershell
$py = '.\.venv_fusion\Scripts\python.exe'
& $py .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice RUTA\indice_comun.csv --landmarks RUTA\landmarks_32f.npz --features RUTA\resnet18 --salida .\resultados\NUEVO --modo fusion --epochs 50 --test-subjects 5 14 --val-subjects 1 7 8
& $py .\codigo\modelos\cnn_bilstm\evaluar_checkpoint.py --experimento .\resultados\NUEVO
```
