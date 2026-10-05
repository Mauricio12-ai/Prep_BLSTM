# Reconocimiento de señas peruanas con BiLSTM

Baseline de clasificación de **34 palabras** a partir de landmarks corporales y de manos. Incluye preprocesamiento, entrenamiento con separación por sujeto y matrices de confusión visuales.

El modelo actual es un **BiLSTM con landmarks**. La arquitectura CNN-BiLSTM con imágenes RGB y la comparación con ST-GCN forman parte del trabajo posterior; no están implementadas en este repositorio.

## 1. Contenido

| Archivo | Función |
|---|---|
| `preprocess_landmarks_c3.py` | Lee coordenadas originales, trata ausencias cortas, normaliza y remuestrea las secuencias. |
| `entrenar.py` | Entrena el BiLSTM, selecciona el mejor checkpoint por macro-F1 de validación y permite evaluar test. |
| `exportar_matrices.py` | Convierte matrices guardadas en tablas HTML con etiquetas. No entrena. |
| `requirements.txt` | Dependencias de Python. |
| `.gitignore` | Excluye datos, pesos y resultados de Git. |

El dataset y los checkpoints no se distribuyen aquí. Se requiere una copia local del dataset C3 empleado en el proyecto. Las dependencias especifican rangos compatibles, no un entorno congelado idéntico al experimento original; `config.json` registra las versiones usadas en cada entrenamiento.

## 2. Preparación en Windows y VS Code

Abre esta carpeta con **Archivo → Abrir carpeta** y abre **Terminal → Nueva terminal** (PowerShell). Ejecuta todos los pasos en esa misma terminal.

```powershell
python --version
python -m pip install -r .\requirements.txt
```

Se recomienda Python 3.10 o posterior. Si el proyecto ya usa un entorno con NumPy y PyTorch instalados, puedes usarlo.

## 3. Ubicación del dataset

El script espera esta estructura relativa:

```text
Dataset/SUBJECT_1/Analyze/sample_1/landmarks/skeleton_real.txt
Dataset/SUBJECT_2/Analyze/sample_1/landmarks/skeleton_real.txt
...
```

`--dataset-root` debe apuntar a la carpeta que contiene directamente los `SUBJECT_...`. En el equipo utilizado para el proyecto:

```powershell
$dataset = "C:\Users\PC\Downloads\Landmarks_C3\Dataset"
(Get-ChildItem "$dataset\SUBJECT_*\*\sample_*\landmarks\skeleton_real.txt" -File).Count
```

Cambia la ruta si trabajas en otro equipo. La base utilizada contiene **4389 archivos**. Un conteo cero significa que la ruta o estructura no coincide.

Cada fila del TXT tiene 225 valores: **75 puntos × 3 coordenadas (x, y, z)**. El orden esperado es 33 puntos de pose, 21 de mano izquierda anatómica y 21 de mano derecha anatómica. El script lee puntos ya extraídos: no ejecuta un detector sobre las imágenes. El formato por sí solo no confirma cuál fue el extractor original ni que cada punto esté sobre la mano correcta.

## 4. Ejecutar el preprocesamiento

```powershell
$procesados = ".\datos_procesados_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python .\preprocess_landmarks_c3.py --dataset-root "$dataset" --output-dir "$procesados" --target-frames 32 --max-interpolation-gap 2
Get-ChildItem "$procesados"
```

### Qué hace

1. Localiza los archivos por sujeto, palabra y muestra.
2. Lee las filas y reorganiza los 225 valores en un tensor `(frames, 75, 3)`.
3. Registra puntos observados y utilizables en máscaras. Interpola huecos cortos de hasta dos frames según las condiciones del código; las ausencias largas no se convierten en movimientos inventados.
4. Centra X/Y en el punto medio de los hombros en cada frame y divide por una escala corporal basada en la distancia mediana entre hombros. El código contiene una alternativa basada en caderas para la escala.
5. Selecciona 32 posiciones distribuidas uniformemente en el tiempo mediante vecino más cercano. Si la secuencia es corta puede repetir índices. No significa tomar únicamente los primeros 32 frames.
6. Guarda coordenadas, máscaras, etiquetas, sujetos y registros de calidad.

`X_xy` es la representación principal. `X_xyz_hybrid` conserva una Z corporal y una Z de manos con referencias diferentes: no debe interpretarse como una reconstrucción 3D métrica común. El BiLSTM de este repositorio utiliza X/Y.

### Archivos generados

| Archivo | Contenido y uso |
|---|---|
| `landmarks_32f.npz` | Coordenadas y máscaras, etiquetas, sujetos e identificadores. Entrada del entrenamiento. |
| `manifest.csv` | Registro por muestra para revisar longitud y calidad. |
| `class_mapping.json` | Relación entre nombre de palabra e índice 0–33. |
| `loso_folds.json` | Definición de particiones por sujeto para organizar experimentos LOSO. |
| `preprocessing_metadata.json` | Configuración, dimensiones, sujetos, clases y conteos de calidad. |

En la ejecución del 5 de octubre de 2026 se obtuvieron 4389 muestras, 17 sujetos y 34 clases: 4146 `OK`, 220 `HIGH_HAND_MISSING`, 12 `HAND_NEVER_DETECTED` y 11 `VERY_SHORT`.

Estas son banderas de revisión; no demuestran que los puntos estén colocados correctamente. No eliminan automáticamente todas las muestras señaladas ni les asignan menos peso en el entrenamiento.

## 5. Entrenar

Para una prueba técnica rápida de tres épocas, sin evaluar test:

```powershell
$datos = Join-Path $procesados "landmarks_32f.npz"
$salida = ".\resultados\BiLSTM_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python .\entrenar.py --datos "$datos" --salida "$salida" --epochs 3 --test-subject 1 --val-subject 2
```

Para una ejecución de hasta 50 épocas, con evaluación final de test:

```powershell
$datos = Join-Path $procesados "landmarks_32f.npz"
$salida = ".\resultados\BiLSTM_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
python .\entrenar.py --datos "$datos" --salida "$salida" --epochs 50 --test-subject 1 --val-subject 2 --evaluar-test
```

Elige el comando apropiado; no hace falta ejecutar ambos. Cada llamada inicia un entrenamiento nuevo. La carpeta de salida debe ser nueva. El script no implementa reanudación de entrenamiento.

Si cerraste la terminal, vuelve a definir `$procesados` con la ruta real de la carpeta generada; las variables de PowerShell no se conservan entre terminales.

### Entrada, arquitectura y salida

- Entrada por muestra: X/Y de 32 frames y 75 puntos, más máscaras de puntos observados y utilizables.
- Por frame: 150 coordenadas + 75 indicadores observados + 75 utilizables = **300 características**.
- LSTM bidireccional de una capa con 64 unidades por dirección.
- Concatenación de estados finales: 128 valores.
- Dropout 0.3 y capa lineal de 128 a 34.
- Salida: 34 puntuaciones (logits); el índice de la mayor puntuación determina la palabra predicha.

Adam, learning rate 0.001, batch 32, entropía cruzada, recorte de norma de gradiente a 1.0 y semilla 42. No se usan pesos de muestra basados en las banderas de calidad.

### Separación por sujeto

| Partición | Sujetos | Muestras esperadas |
|---|---|---:|
| Entrenamiento | Todos excepto 1 y 2 | 3838 |
| Validación | 2 | 157 |
| Test | 1 | 393 |

Se excluye `SUBJECT_2/Analyze/sample_2` para mantener las muestras comunes con el procesamiento RGB/depth. Quedan 4388. El cargador valida este tamaño, 34 clases y 17 sujetos: no está diseñado para entrenar directamente una muestra pequeña del dataset.

Validación selecciona el checkpoint por **macro-F1 de las 34 clases**. Tras 10 épocas consecutivas sin mejora se detiene el entrenamiento. `--epochs 50` es un máximo. Test se calcula al final solo con `--evaluar-test`, usando el checkpoint seleccionado.

Este comando ejecuta una separación fija cross-subject; **no recorre automáticamente los 17 folds LOSO**, aunque exista `loso_folds.json`.

## 6. Qué significan los mensajes y archivos de salida

Cada época es un recorrido completo por las muestras de entrenamiento.

| Indicador en terminal | Interpretación |
|---|---|
| `epoch` | Número de época. |
| `train_loss` | Pérdida en entrenamiento; menor suele ser mejor. No es un porcentaje. |
| `val_loss` | Pérdida en el sujeto de validación. |
| `val_accuracy` | Fracción de muestras de validación clasificadas correctamente. 0.80 equivale a 80 %. |
| `val_macro_f1` | Promedio del F1 por clase, dando igual peso a las 34 clases. |

| Archivo de salida | Para qué sirve |
|---|---|
| `mejor_modelo.pt` | Pesos del checkpoint con mejor macro-F1 de validación. No es un documento de texto ni incluye toda la arquitectura. |
| `historial.csv` | Métricas de cada época; se actualiza durante el entrenamiento. |
| `config.json` | Parámetros, versiones, dispositivo y conteos de muestras. |
| `particiones.json` | Identificadores de muestras asignadas a train, val y test. |
| `metricas.json` | Métricas y matrices de validación y, si se solicitó, test. |
| `predicciones_test.csv` | Identificador, clase real y clase predicha de cada muestra de test. Solo con evaluación de test. |
| `matriz_confusion_validacion.html` | Matriz visual de validación del mejor modelo. |
| `matriz_confusion_test.html` | Matriz visual de test, si se evaluó. |

El entrenador actualizado llama automáticamente a `exportar_matrices.py`; conserva ambos archivos juntos.

## 7. Abrir resultados sin volver a entrenar

En la misma terminal:

```powershell
Import-Csv "$salida\historial.csv" | Format-Table -AutoSize
Start-Process "$salida\matriz_confusion_validacion.html"
Start-Process "$salida\matriz_confusion_test.html"
explorer "$salida"
```

El archivo de test solo existe si se solicitó su evaluación. Si cerraste la terminal, define primero `$salida` con la ruta real de tus resultados.

Para generar las tablas a partir de un `metricas.json` existente:

```powershell
python .\exportar_matrices.py --metricas "$salida\metricas.json"
```

No requiere entrenar ni cargar `mejor_modelo.pt`. El exportador independiente usa el orden de las 34 clases de este proyecto; si cambias las clases debes actualizar su lista `CLASES`. El entrenador obtiene los nombres a partir de las muestras y sus etiquetas.

### Leer la matriz

Filas = palabra real. Columnas = palabra predicha. Cada celda cuenta muestras. La diagonal verde indica aciertos; las celdas naranjas, errores. Los totales son adicionales a las 34 filas y columnas de clases. Las tablas se abren localmente sin internet y permiten desplazamiento horizontal y vertical.

## 8. Resultados de referencia y alcance

En la ejecución previa compartida, el entrenamiento se detuvo tras 35 épocas y seleccionó la época 25: validación 130/157 aciertos (82.80 %), macro-F1 0.7306; test 307/393 aciertos (78.12 %), macro-F1 0.7404. Estos resultados son una referencia de aquella ejecución, no una garantía para otros entornos ni el resultado de LOSO completo.

Pendiente para la comparación de investigación: completar el protocolo por sujetos, revisar detección de manos sobre RGB, analizar pares mínimos y comparar arquitecturas con las mismas particiones. El test debe reservarse para evaluación y no utilizarse para seleccionar hiperparámetros.

## 9. Subir a GitHub desde VS Code

Crea en GitHub un repositorio vacío y copia su URL. En la terminal de esta carpeta:

```powershell
git init
git add README.md requirements.txt .gitignore preprocess_landmarks_c3.py entrenar.py exportar_matrices.py
git status
git commit -m "Agrega baseline BiLSTM, preprocesamiento y matrices"
git branch -M main
git remote add origin URL_DE_TU_REPOSITORIO
git push -u origin main
```

Sustituye `URL_DE_TU_REPOSITORIO` por la URL real. Este bloque es para un repositorio nuevo. Si ya trabajas dentro de un repositorio, no reinicialices ni reemplaces su remoto: revisa primero `git status` y `git remote -v`.

Antes del commit, comprueba que solo se incluyen los seis archivos indicados. El dataset, resultados y pesos se mantienen localmente. `.gitignore` no deja de rastrear archivos que ya estaban incluidos en Git.

## 10. Problemas frecuentes

- **can't open file:** la terminal no está en la carpeta del script. Abre la terminal con clic derecho sobre la carpeta en VS Code.
- **No se encontraron skeleton_real.txt:** `--dataset-root` apunta a un nivel incorrecto.
- **No module named numpy/torch:** instala dependencias usando el mismo `python` que ejecuta los scripts.
- **La carpeta de salida ya existe:** usa un nombre nuevo para el entrenamiento.
- **Test evaluado: False:** no se añadió `--evaluar-test`; no se genera matriz de test en esa ejecución.
- **No se reconoce git:** necesitas una instalación de Git disponible en la terminal; también puedes subir los seis archivos desde la interfaz web del repositorio.
