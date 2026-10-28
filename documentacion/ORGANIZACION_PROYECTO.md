# Proyecto LSP: código, datos y resultados organizados

Este paquete reúne los scripts disponibles que hemos usado o preparado. No contiene
el dataset de 19 GB, las 4388 secuencias procesadas, tus checkpoints o los resultados
guardados en tu PC. Tampoco se han movido o modificado tus carpetas originales.

## Empezar sin perder las rutas actuales

1. Extrae el ZIP en el Escritorio. La carpeta de proyecto se llama `Proyecto_LSP`.
   Si el extractor crea una carpeta adicional, abre la que contiene este README y `codigo`.
2. En VS Code elige Archivo → Abrir carpeta → `Proyecto_LSP`.
3. Conserva por ahora tus datos en sus ubicaciones actuales. Puedes trabajar con ellos
   desde esta carpeta nueva utilizando rutas completas.
4. `config/rutas_actuales.json` registra las rutas conocidas. Es un inventario; los
   scripts no lo leen automáticamente. Debes pasar los argumentos como en los ejemplos.
5. No borres las carpetas anteriores hasta comprobar que esta organización funciona.

## Directorios principales

| Directorio | Qué guardar |
|---|---|
| `codigo/preprocesamiento/` | Scripts que preparan, comprueban y relacionan entradas |
| `codigo/modelos/` | Arquitecturas, extracción CNN, entrenamiento y evaluación |
| `config/` | Referencias a tus rutas y decisiones de configuración |
| `datos/originales/` | Lugar reservado para ZIP y datos sin procesar |
| `datos/procesados/landmarks/` | Lugar reservado para NPZ de landmarks y sus metadatos |
| `datos/procesados/rgb_depth/` | Lugar reservado para C3_RGB_Depth_112 |
| `datos/indices/` | Lugar reservado para cada ejecución de la unión RGB–landmarks |
| `datos/caracteristicas/resnet18/` | Caché de características visuales, calculada una sola vez |
| `resultados/` | Una subcarpeta por modelo y otra por ejecución |
| `documentacion/` | Artículo, papers, figuras y registro de pruebas |

Las carpetas de datos se incluyen como esqueleto. Tenerlas vacías no impide trabajar
con los datos que siguen en Descargas o en tus carpetas anteriores.

## Scripts incluidos y qué hace cada uno

| Directorio | Archivos | Función |
|---|---|---|
| `codigo/preprocesamiento/landmarks/` | `preprocess_landmarks_c3.py` | Preprocesamiento original de skeleton_real.txt; genera landmarks_32f.npz y metadatos |
| `codigo/preprocesamiento/rgb_depth/` | `LEEME.md` | Sitio reservado para tu procesar_c3.py original; ese archivo exacto no se incluye |
| `codigo/preprocesamiento/rgb_landmarks/` | `preparar_rgb_landmarks.py` | Preparación conjunta desde ZIP; la usamos para la muestra pequeña |
| mismo directorio | `preprocess_landmarks_original.py` | Funciones auxiliares importadas por preparar_rgb_landmarks.py; conservar juntos |
| mismo directorio | `comprobar_entrada.py` | Comprueba el NPZ de la preparación conjunta; no recibe el NPZ global antiguo |
| `codigo/preprocesamiento/union/` | `unir_rgb_landmarks.py` | Relaciona los datos ya procesados sin duplicar imágenes |
| `codigo/modelos/bilstm/` | `entrenar.py` | Baseline que usa exclusivamente landmarks XY y máscaras |
| mismo directorio | `exportar_matrices.py` | Exportación de métricas y matrices del baseline; conservar junto a entrenar.py |
| mismo directorio | `requirements.txt` | Dependencias del baseline |
| `codigo/modelos/cnn_bilstm/` | `extraer_rgb.py` | ResNet18 congelada; genera características (32,512) por muestra |
| mismo directorio | `modelo.py` | BiLSTM RGB, BiLSTM landmarks y clasificador; permite rgb, landmarks o fusion |
| mismo directorio | `entrenar_fusion.py` | Entrena la variante elegida; guardar junto a modelo.py |
| `codigo/modelos/stgcn_referencia/` | Scripts del ZIP del compañero | Referencia sin adaptar; leer ESTADO.md y su README original |

El archivo auxiliar `preprocess_landmarks_original.py` y `preprocess_landmarks_c3.py`
contienen el mismo código de origen. Se mantienen ambos nombres para respetar el import
del paso conjunto sin cambiar algoritmos durante esta reorganización.

## Qué archivos de datos pertenecen juntos

| Grupo | Archivos/carpetas que deben conservarse juntos |
|---|---|
| Landmarks procesados | landmarks_32f.npz, manifest.csv, class_mapping.json, loso_folds.json, preprocessing_metadata.json |
| RGB/depth procesados | secuencias/, manifest.csv, config.json, resumen.json |
| Unión | indice_comun.csv, revision.csv, resumen.json, class_mapping.json |
| Características CNN | config_cnn.json, resumen_extraccion.json y todos los NPZ numerados |
| Entrenamiento | mejor_modelo.pt, config.json, particiones.json, historial.csv, metricas.json, predicciones y matrices |

No mezcles varios manifest.csv o config.json en una carpeta: pertenecen a procesos diferentes.
No cambies los nombres de los NPZ numerados de características: están ligados al índice.

## Tu estado actual confirmado

- Landmarks: 4389 muestras, 17 sujetos, 34 clases; XY (4389,32,75,2).
- RGB/depth: 4388 procesadas, una con error en el registro de 4389.
- Unión ejecutada: 4388 compatibles, una exclusión y cero errores adicionales.
- Exclusión: SUBJECT_2/Analyze/sample_2.
- Ya entrenaste el baseline BiLSTM antes. Sus pesos y métricas permanecen en tu equipo;
  esta entrega no los copia ni los sustituye.
- Se entregó el código CPU de CNN-BiLSTM/fusión. Aún no has comunicado una extracción
  completa ni resultados reales del entrenamiento multimodal.

## Primer paso desde la carpeta nueva

Abre una terminal en `Proyecto_LSP`. Para crear un entorno separado:

```powershell
python -m venv .venv_fusion
.\.venv_fusion\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Si ya instalaste el entorno en la carpeta antigua puedes seguir usando su Python con
ruta absoluta. No hace falta instalar de nuevo para continuar allí; estos comandos son
para trabajar de forma independiente desde la nueva organización.

## Siguiente ejecución: diez muestras RGB

No necesitas repetir la unión que ya terminaste. Usa el índice existente:

```powershell
$indice = "C:\Users\PC\Desktop\Paso1_RGB_Landmarks\Paso1_RGB_Landmarks\union_20261012_153325\indice_comun.csv"
$landmarks = "C:\Users\PC\Desktop\Seminario1\BaselineActualizado\datos_procesados_20261005_160031\landmarks_32f.npz"
$features = ".\datos\caracteristicas\resnet18"

.\.venv_fusion\Scripts\python.exe .\codigo\modelos\cnn_bilstm\extraer_rgb.py --indice "$indice" --salida "$features" --limite 10 --hilos 4
```

La carpeta `datos/caracteristicas/resnet18` se incluye vacía para esta ejecución.
La primera vez se descargan los pesos de ResNet18. Al terminar la prueba, procesar todo:

```powershell
.\.venv_fusion\Scripts\python.exe .\codigo\modelos\cnn_bilstm\extraer_rgb.py --indice "$indice" --salida "$features" --hilos 4
```

El extractor reutiliza características válidas de la misma configuración. Si ya habías
empezado una extracción en otra ubicación, usa esa carpeta en `$features` para continuar;
no mezcles archivos de dos extracciones. No es entrenamiento ni reconocimiento todavía.

## Prueba de entrenamiento: tres épocas

Después de completar todas las características:

```powershell
$run = ".\resultados\fusion\prueba_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
.\.venv_fusion\Scripts\python.exe .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice "$indice" --landmarks "$landmarks" --features "$features" --salida "$run" --modo fusion --epochs 3 --test-subject 1 --val-subject 2 --hilos 4
```

`--test-subjects` y `--val-subjects` aceptan uno o varios sujetos. Para el split común
con ST-GCN (12/3/2 sujetos; 2813/988/587 muestras), usar los sujetos acordados, por ejemplo
`--test-subjects 15 16 --val-subjects 1 5 10`. Los nombres antiguos `--test-subject` y
`--val-subject` siguen funcionando, y `evaluar_checkpoint.py` lee ambos formatos de config.json.

`--salida` debe apuntar a una subcarpeta nueva, no directamente a resultados/fusion,
que ya existe. Los scripts de entrenamiento no sobrescriben ejecuciones anteriores.

Para una ejecución más larga usar `--epochs 50 --patience 10` y otra carpeta de salida.
Para evaluar test al terminar la ejecución final, añadir `--evaluar-test`, con la
configuración ya fijada. Elegir el modelo con validación, no con test.

## Otras variantes

| Variante | Script | Argumento | Carpeta de resultados |
|---|---|---|---|
| Baseline anterior | codigo/modelos/bilstm/entrenar.py | --datos ruta_landmarks | resultados/bilstm/nueva_ejecucion |
| CNN-BiLSTM solo RGB | codigo/modelos/cnn_bilstm/entrenar_fusion.py | --modo rgb | resultados/cnn_bilstm/nueva_ejecucion |
| BiLSTM solo landmarks bajo índice común | mismo script | --modo landmarks | resultados/bilstm/nueva_ejecucion |
| Fusión RGB + landmarks | mismo script | --modo fusion | resultados/fusion/nueva_ejecucion |
| ST-GCN | referencia del compañero | Consultar README propio | resultados/stgcn/nueva_ejecucion |

No confundir `entrenar.py` (baseline original) con `entrenar_fusion.py` (variantes del
paquete nuevo). Para conservar resultados comparables, usar las mismas muestras,
sujetos, semillas y presupuesto de entrenamiento. Esta partición test=1/val=2 no es
LOSO completo. Los pares mínimos todavía requieren anotación y análisis específico.

## Si más adelante quieres mover también los datos

1. Copia primero, verifica, y solo después considera retirar las carpetas antiguas.
2. Coloca los landmarks y sus JSON/CSV juntos en datos/procesados/landmarks.
3. Coloca el contenido completo de C3_RGB_Depth_112 en datos/procesados/rgb_depth.
4. Vuelve a ejecutar la unión con esas rutas y una salida nueva dentro de datos/indices.
   El indice_comun.csv actual contiene rutas absolutas antiguas a los NPZ RGB.
5. La caché CNN queda vinculada al contenido del índice. Si cambia, el extractor la
   rechazará: usar una nueva carpeta de características. No editar hashes manualmente.

Por ese motivo, la opción inmediata es organizar el código y mantener los datos en su sitio.

## Límites del código y del paquete

La CNN es ResNet18 ImageNet congelada con entrada letterbox 112×112 y normalización
ImageNet; se evita el recorte central del preset estándar. La baja resolución puede
perder detalle de dedos. Se entrenan BiLSTM y clasificador, no la CNN extremo a extremo.
Depth no entra al nuevo modelo. Se conservan las máscaras del NPZ original; la unión
no corrige posiciones anatómicas ni demuestra sincronía física. No hay inferencia en
vivo. No se generan nuevas métricas reales al reorganizar carpetas.

El .gitignore evita subir datos pesados, pesos y resultados locales. No publica nada
en GitHub. Comparte informes concretos solo cuando hayas decidido cuáles corresponden.
