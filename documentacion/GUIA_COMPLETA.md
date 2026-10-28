# Guía completa de mi proyecto: fusión CNN-BiLSTM para Lengua de Señas Peruana

Esta guía explica, de principio a fin, qué hace mi código, cómo se ejecuta, cómo se leen
los resultados y cómo explicarlo a la profesora. Los números de la sección 8 son los
resultados reales de mi entrenamiento del 12/10 (`resultados/fusion/entrenamiento_50_20261012_163623`).

---

## 1. Mi trabajo en una frase

> Reconozco **34 palabras aisladas de la Lengua de Señas Peruana** a partir de videos,
> combinando dos fuentes de información: **la imagen (RGB)** y **los puntos del cuerpo y
> las manos (landmarks)**. Cada fuente se analiza en el tiempo con un **BiLSTM** y luego
> se **fusionan** para decidir qué palabra se signó.

---

## 2. Conceptos que necesito entender (glosario simple)

### Datos

| Término | Qué significa en mi proyecto |
|---|---|
| **Muestra** | Un video de una persona haciendo una seña. Tengo 4388. |
| **Sujeto** | La persona que hace la seña. Hay 17 (SUBJECT_1 … SUBJECT_17). |
| **Clase** | La palabra que se signa. Hay 34 (Analyze, Better, …, Very_tall). |
| **Frame** | Una imagen del video. Cada video tiene distinta cantidad; yo tomo **32** repartidos uniformemente. |
| **RGB** | La imagen a color normal (Red-Green-Blue). Yo la uso reducida a **112 × 112** píxeles. |
| **Landmarks / skeleton** | Puntos clave detectados con MediaPipe: **33 de pose + 21 de cada mano = 75 puntos** por frame. |
| **Máscara** | Un 1 o 0 que dice si un punto fue detectado (observed) o si se puede usar (usable). Si una mano no se ve, su máscara es 0. |
| **Forma (shape)** | El tamaño de un arreglo. Ej.: `(32, 512)` = 32 frames con 512 números cada uno. |

### Red neuronal

| Término | Explicación sencilla |
|---|---|
| **CNN** (red convolucional) | Red especializada en imágenes. Detecta bordes, formas, manos, posiciones. |
| **ResNet18** | Una CNN conocida de 18 capas. |
| **Preentrenada (ImageNet)** | Ya aprendió a "ver" con 1.2 millones de fotos generales. Reutilizo ese conocimiento. |
| **Congelada** | **No** cambio sus pesos; solo la uso para convertir cada frame en **512 números** (características). |
| **Características (features)** | Un resumen numérico de lo que hay en la imagen. 512 números por frame. |
| **LSTM** | Red que lee **secuencias** paso a paso y recuerda lo importante. Sirve para el movimiento. |
| **BiLSTM** | Dos LSTM: una lee del frame 1 al 32 (**forward**) y otra del 32 al 1 (**backward**). Así entiende el inicio y el final de la seña. |
| **Estado oculto (hidden)** | El "resumen" que la LSTM tiene al final. En mi código mide **64** por dirección → **128** con las dos. |
| **Concatenación** | Pegar dos vectores uno tras otro: 128 (RGB) + 128 (landmarks) = **256**. |
| **Dropout 0.3** | Durante el entrenamiento apaga al azar el 30 % de los valores para que el modelo no memorice. |
| **Capa lineal** | Convierte los 256 valores en **34 puntuaciones** (una por palabra). |
| **Logits** | Esas 34 puntuaciones sin normalizar. |
| **Softmax** | Convierte las puntuaciones en probabilidades que suman 1. La palabra con mayor probabilidad es la predicción. |

### Entrenamiento

| Término | Explicación sencilla |
|---|---|
| **Época (epoch)** | Una pasada completa por todos los datos de entrenamiento. |
| **Batch** | Grupo de muestras que se procesan juntas. Uso **32**. |
| **Loss (pérdida)** | Número que mide qué tan equivocado está el modelo. **Más bajo = mejor.** Uso *cross-entropy*. |
| **Optimizador Adam / learning rate 0.001** | El método que ajusta los pesos para bajar la pérdida, y qué tan grandes son los pasos. |
| **Gradient clipping 1.0** | Limita el tamaño de cada ajuste para que el entrenamiento no se descontrole. |
| **Checkpoint** | Archivo con los pesos guardados (`mejor_modelo.pt`). |
| **Early stopping / patience 10** | Si en 10 épocas seguidas la validación no mejora, se detiene. |
| **Semilla (seed 42)** | Fija el azar para que el experimento se pueda repetir. |
| **Overfitting (sobreajuste)** | El modelo memoriza el entrenamiento y falla con datos nuevos. Señal: la loss de train baja y la de validación sube. |

### Evaluación

| Término | Explicación sencilla |
|---|---|
| **Train / Validación / Test** | Train: para aprender. Validación: para elegir la mejor época. Test: examen final, se mira **una sola vez** al final. |
| **Cross-subject** | Los sujetos de test **nunca** aparecen en entrenamiento. Mide si el modelo funciona con **personas nuevas**. |
| **LOSO** (Leave-One-Subject-Out) | Repetir el experimento 17 veces, dejando cada vez un sujeto distinto para test. Yo **todavía no** hago LOSO completo. |
| **Accuracy** | % de muestras bien clasificadas. |
| **Precision** (por palabra) | De las veces que el modelo dijo "X", ¿cuántas eran X? |
| **Recall** (por palabra) | De todas las X reales, ¿cuántas reconoció? |
| **F1** | Promedio equilibrado entre precision y recall. |
| **Macro-F1** | Promedio del F1 de las 34 palabras, **todas con el mismo peso**. Es mi métrica principal porque hay palabras con más muestras que otras. |
| **Matriz de confusión** | Tabla 34 × 34: filas = palabra real, columnas = palabra predicha. La diagonal son aciertos; fuera de ella, confusiones. |
| **Pares mínimos** | Palabras cuya seña difiere en un solo detalle (ej. *Tall* / *Very_tall*). Son las más difíciles. |

---

## 3. El pipeline completo, paso a paso, con mis archivos

```
Dataset (videos + skeleton_real.txt)
   │
   ├─► [1] preprocess_landmarks_c3.py ─► landmarks_32f.npz (4389 muestras)
   ├─► [2] procesar_c3.py (RGB)       ─► C3_RGB_Depth_112 (4388 muestras)
   │
   └─► [3] unir_rgb_landmarks.py      ─► indice_comun.csv (4388 muestras comunes)
              │
              └─► [4] extraer_rgb.py   ─► características ResNet18 (32 × 512 por muestra)
                        │
                        └─► [5] entrenar_fusion.py  ─► mejor_modelo.pt + métricas de validación
                                  │
                                  └─► [6] evaluar_checkpoint.py ─► métricas y matriz de test
```

### [1] Preprocesamiento de landmarks — `codigo/preprocesamiento/landmarks/preprocess_landmarks_c3.py`
1. Lee cada `skeleton_real.txt`: 225 columnas por frame = 75 puntos × (x, y, z).
2. **Interpola huecos cortos**: si un punto desaparece ≤ 2 frames entre dos detecciones, lo rellena en línea recta. Los huecos largos **no** se inventan.
3. **Normaliza** respecto al cuerpo: centra en el punto medio de los hombros y divide por la distancia entre hombros. Así no importa si la persona está más cerca o más lejos de la cámara, ni si es más alta.
4. **Toma 32 frames** repartidos uniformemente a lo largo del video.
5. Guarda las **máscaras** (`observed`, `usable`) y una etiqueta de calidad (OK, HIGH_HAND_MISSING, …).

### [2] Preprocesamiento RGB — `procesar_c3.py` (está en mi carpeta anterior, no en este paquete)
Reduce cada frame a 112 × 112 con *letterbox* (sin deformar ni recortar la imagen), tomando
**los mismos 32 frames** que los landmarks.

### [3] Unión — `codigo/preprocesamiento/union/unir_rgb_landmarks.py`
Comprueba que cada muestra tenga RGB **y** landmarks, y crea `indice_comun.csv`.
Resultado: **4388 muestras**. Se excluyó `SUBJECT_2/Analyze/sample_2` porque su RGB falló.

### [4] Extracción de características — `codigo/modelos/cnn_bilstm/extraer_rgb.py`
Pasa los 32 frames de cada muestra por la **ResNet18 congelada** y guarda 512 números por
frame en `datos/caracteristicas/resnet18`. **Se hace una sola vez.** Como la CNN no se
entrena, no tiene sentido recalcularla en cada época (en CPU tardaría muchísimo).

### [5] Entrenamiento — `codigo/modelos/cnn_bilstm/entrenar_fusion.py` (+ `modelo.py`)
Entrena las **dos BiLSTM y el clasificador a la vez**. Elige la mejor época por macro-F1 de validación.

### [6] Evaluación — `codigo/modelos/cnn_bilstm/evaluar_checkpoint.py`
Carga el mejor modelo **sin volver a entrenarlo** y lo prueba con los sujetos de test.

---

## 4. Mi arquitectura en detalle (con las formas reales del código)

```
RAMA RGB
  32 frames RGB 112×112
     → ResNet18 congelada          → (32, 512)   [paso 4, precalculado]
     → BiLSTM (512 → 64 por dirección)
     → estado final forward + backward → 128

RAMA LANDMARKS
  32 frames × 300 valores
     (150 = 75 puntos × XY  +  75 máscaras observed  +  75 máscaras usable)
     → BiLSTM (300 → 64 por dirección)
     → estado final forward + backward → 128

FUSIÓN
  concatenar 128 + 128 = 256
     → Dropout 0.3
     → Capa lineal 256 → 34
     → (Softmax) → palabra con mayor probabilidad
```

- **Parámetros entrenables: 492 066.** Los de la ResNet18 (~11 millones) no se entrenan.
- En `modelo.py` no hay una capa Softmax escrita: la función de pérdida `CrossEntropyLoss`
  ya la aplica internamente, y para predecir se toma la puntuación más alta (`argmax`).
  El resultado es el mismo.
- Las coordenadas XY se multiplican por la máscara `usable`: si un punto no es fiable vale 0,
  y la máscara le avisa al modelo que ese 0 significa "no detectado".
- **Depth (profundidad) no se usa** en esta arquitectura.

### ¿Por qué estas decisiones? (útil para defender)

| Decisión | Razón |
|---|---|
| ResNet18 **congelada** | Pocos datos (4388 videos) y entrenamiento en CPU. Reutilizar una red preentrenada evita sobreajuste y es mucho más rápido. |
| **BiLSTM** | Una seña es movimiento: importa el orden de los frames. Leer en los dos sentidos ayuda a capturar el inicio y el final del gesto. |
| **Dos modalidades** | RGB ve la forma de la mano y la apariencia; los landmarks dan posiciones precisas e independientes del fondo y la ropa. Se complementan. |
| **Fusión por concatenación** | Es la forma más simple y transparente de combinar: cada rama aporta su resumen y el clasificador decide. |
| **Macro-F1** | Las clases no tienen la misma cantidad de muestras; macro-F1 no deja que las palabras frecuentes oculten a las difíciles. |
| **Cross-subject** | Lo realista es reconocer a personas que el modelo nunca vio. |

---

## 5. Cómo se entrena (qué pasa dentro de `entrenar_fusion.py`)

1. Carga el índice, los landmarks y las características RGB, y verifica que todo coincida (IDs, etiquetas, sujetos, frames).
2. Separa train / validación / test **por sujeto**.
3. Por cada época:
   - Recorre el train en batches de 32 → calcula la loss → ajusta los pesos (Adam, lr 0.001, clipping 1.0).
   - Evalúa en validación → calcula accuracy y macro-F1.
   - Si el macro-F1 de validación es el mejor hasta ahora, guarda `mejor_modelo.pt`.
   - Si lleva 10 épocas sin mejorar, se detiene (early stopping).
4. Al final guarda las métricas de validación. El test **solo** se evalúa si se pide (`--evaluar-test` o `evaluar_checkpoint.py`).

---

## 6. Comandos (PowerShell, desde la carpeta `Proyecto_LSP`)

Abrir la terminal en `C:\Users\PC\Desktop\Proyecto_LSP` y definir las rutas (una vez por terminal):

```powershell
$indice    = "C:\Users\PC\Desktop\Paso1_RGB_Landmarks\Paso1_RGB_Landmarks\union_20261012_153325\indice_comun.csv"
$landmarks = "C:\Users\PC\Desktop\Seminario1\BaselineActualizado\datos_procesados_20261005_160031\landmarks_32f.npz"
$features  = ".\datos\caracteristicas\resnet18"
$py        = ".\.venv_fusion\Scripts\python.exe"
```

### Extraer características RGB (ya está hecho; solo si se cambia el índice)
```powershell
& $py .\codigo\modelos\cnn_bilstm\extraer_rgb.py --indice "$indice" --salida "$features" --hilos 4
```

### Entrenar mi arquitectura (fusión) con el split común 12/3/2
```powershell
$run = ".\resultados\fusion\split_comun_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $py .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice "$indice" --landmarks "$landmarks" --features "$features" --salida "$run" --modo fusion --epochs 50 --patience 10 --test-subjects 15 16 --val-subjects 1 5 10 --hilos 4
```

### Evaluar en test el modelo recién entrenado
```powershell
& $py .\codigo\modelos\cnn_bilstm\evaluar_checkpoint.py --experimento "$run"
```

### Abrir la matriz de confusión de test
```powershell
$t = Get-ChildItem "$run\test_*" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Start-Process "$($t.FullName)\matriz_test.html"
```

### Variantes para comparar (opcionales, con el mismo split)
```powershell
# Solo RGB (CNN-BiLSTM)
& $py .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice "$indice" --landmarks "$landmarks" --features "$features" --salida ".\resultados\cnn_bilstm\split_comun_$(Get-Date -Format 'yyyyMMdd_HHmmss')" --modo rgb --epochs 50 --patience 10 --test-subjects 15 16 --val-subjects 1 5 10
# Solo landmarks (BiLSTM)
& $py .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice "$indice" --landmarks "$landmarks" --salida ".\resultados\bilstm\split_comun_$(Get-Date -Format 'yyyyMMdd_HHmmss')" --modo landmarks --epochs 50 --patience 10 --test-subjects 15 16 --val-subjects 1 5 10
```

### Qué significa cada opción

| Opción | Significado |
|---|---|
| `--modo fusion / rgb / landmarks` | Qué ramas se usan. `fusion` es mi arquitectura. |
| `--epochs 50` | Máximo de épocas. |
| `--patience 10` | Épocas sin mejorar antes de detenerse. |
| `--test-subjects 15 16` | Sujetos reservados para el examen final. |
| `--val-subjects 1 5 10` | Sujetos para elegir la mejor época. |
| `--hidden 64` | Tamaño de cada LSTM por dirección (por defecto 64 → 128 con las dos). |
| `--dropout 0.3`, `--lr 0.001`, `--batch 32`, `--seed 42` | Hiperparámetros (valores por defecto). |
| `--salida` | Carpeta **nueva** donde se guarda todo. Nunca sobrescribe. |

---

## 7. Archivos que produce cada ejecución

| Archivo | Qué contiene |
|---|---|
| `config.json` | Todos los parámetros usados, sujetos de cada conjunto, cantidad de muestras y huellas (hash) de los datos. Sirve para reproducir. |
| `particiones.json` | La lista exacta de muestras en train, val y test. |
| `historial.csv` | Una fila por época: train_loss, val_loss, val_accuracy, val_macro_f1. |
| `mejor_modelo.pt` | Pesos de la mejor época. |
| `metricas.json` | Mejor época y métricas de validación. |
| `matriz_val.csv`, `metricas_por_clase_val.csv`, `predicciones_val.csv` | Detalle de validación. |
| `test_FECHA/metricas_test.json` | **Resultado final**: accuracy, macro-F1, aciertos y errores en test. |
| `test_FECHA/metricas_por_clase_test.csv` | Precision, recall y F1 de cada palabra. |
| `test_FECHA/predicciones_test.csv` | Cada video de test: palabra real, predicha y si acertó. |
| `test_FECHA/matriz_test.html` | Matriz de confusión para abrir en el navegador (verde = aciertos, rojo = confusiones). |

---

## 8. Cómo interpretar los resultados (con mis números reales)

Entrenamiento del 12/10: **train = sujetos 3–17, validación = sujeto 2, test = sujeto 1**.

### 8.1 El historial: cómo aprendió

| Época | train_loss | val_loss | val_accuracy | val_macro_f1 |
|---|---|---|---|---|
| 1 | 2.77 | 2.03 | 49.0 % | 0.37 |
| 5 | 0.97 | 0.91 | 75.8 % | 0.69 |
| 10 | 0.54 | 0.68 | 78.3 % | 0.72 |
| **18 (mejor)** | **0.26** | **0.64** | **80.9 %** | **0.743** |
| 28 (se detuvo) | 0.17 | 0.75 | 77.1 % | 0.70 |

Cómo leerlo:
- **Épocas 1–10:** las dos pérdidas bajan rápido → el modelo está aprendiendo.
- **Época 18:** mejor macro-F1 en validación → ahí se guardó el modelo.
- **Épocas 19–28:** train_loss sigue bajando (0.26 → 0.17), pero val_loss sube (0.64 → 0.75) → **empieza el sobreajuste**. Después de 10 épocas sin mejorar, el early stopping lo detuvo en la 28.
- Una val_loss de 2.03 → 0.64 significa que el modelo pasó de "casi adivinar" a estar bastante seguro de sus respuestas.

### 8.2 Resultado final en test (sujeto 1, 393 videos, nunca vistos)

| Métrica | Valor | Lectura |
|---|---|---|
| Accuracy | **75.1 %** | 295 aciertos y 98 errores. |
| Macro-F1 | **0.704** | En promedio, las 34 palabras se reconocen razonablemente, pero no todas igual. |
| Macro-precision | 0.772 | Cuando el modelo dice una palabra, suele acertar. |
| Macro-recall | 0.734 | Algunas palabras se le "escapan". |

Que test (75 %) sea un poco menor que validación (81 %) es **normal**: la validación se usó
para elegir la época, y el sujeto 1 es otra persona con su propio estilo.

Como referencia, adivinar al azar entre 34 palabras daría ~3 %.

### 8.3 Por palabra (de `metricas_por_clase_test.csv`)

- **Perfectas (F1 = 1.00):** Better, Crowd, Forget, Language, Live, Poor, Very_poor.
- **Muy bien (F1 ≥ 0.90):** Complaint, Learn_everything, Learn_something, Small_cell_phone, Punish.
- **Las más difíciles (F1 < 0.40):** Progress (0.00), Tall (0.00), Analyze (0.06), Fine (0.17), People (0.18), Cunning (0.29), Shout (0.38).

Cómo leer precision vs. recall: *Fine* tiene precision 1.00 pero recall 0.09 → cuando dice
"Fine" acierta, pero casi nunca lo dice: lo confunde con otras palabras. *Friday* tiene
recall 1.00 y precision 0.44 → reconoce todos los Friday, pero también llama "Friday" a otras señas.

### 8.4 Principales confusiones (de la matriz de test)

| Real → Predicha | Veces | Comentario |
|---|---|---|
| **Tall → Very_tall** | 11 de 11 | **Par mínimo**: nunca distingue *Tall* de *Very_tall*. |
| People → Analyze | 9 de 10 | |
| Cunning → Friday | 9 de 12 | |
| Progress → Stagnate | 8 de 10 | Significados opuestos; conviene revisar en video si el movimiento es parecido. |
| Shout → Analyze | 6 de 10 | |
| Person → Green | 5 de 10 | |
| Fine → Decrease | 5 de 11 | |
| Extra_large_cell_phone → Large_cell_phone | 3 de 10 | **Par mínimo** (tamaño). |

### 8.5 Pares mínimos (lo central del tema)

| Grupo | Qué pasó en test (sujeto 1) |
|---|---|
| Tall / Very_tall / Very_short | Very_tall y Very_short bien; **Tall se confunde siempre con Very_tall**. |
| Poor / Very_poor | **Perfecto**, sin confusiones. |
| Small_cell_phone / Large_cell_phone / Extra_large_cell_phone / Small | Small y Large bien; Extra_large se confunde 3 veces con Large. |
| Learn_everything / Learn_something | **Perfecto**. |
| People / Person / Crowd | No se confunden entre sí, pero People se confunde con Analyze. |

Conclusión honesta: **algunos pares mínimos se distinguen perfectamente y otros no**. La
diferencia de intensidad (*Tall* vs *Very_tall*) es el caso más difícil para mi modelo.
⚠️ La lista oficial de pares mínimos debe definirse/anotarse con el equipo; estos grupos son
los que se ven por los nombres de las palabras.

### 8.6 Cuidado al interpretar

- **Un solo sujeto de test** → el resultado puede depender mucho de esa persona.
- Cada palabra tiene **~10 videos** en test: **un solo error baja el recall de esa palabra un 10 %**.
- No es LOSO completo; con el split 12/3/2 (2 sujetos de test) será más representativo.
- Estos números son del split antiguo; al reentrenar con 15,16 / 1,5,10 cambiarán.

---

## 9. Cómo explicárselo a la profesora

### Guion de 3–5 minutos

1. **Problema (30 s).** "Mi trabajo reconoce 34 palabras aisladas de la Lengua de Señas
   Peruana. El reto principal son los pares mínimos: señas casi iguales que cambian en un solo
   detalle, como *Tall* y *Very_tall*."
2. **Datos (40 s).** "Uso 4388 videos de 17 personas. De cada video tengo la imagen RGB y 75
   landmarks de MediaPipe: 33 del cuerpo y 21 de cada mano. Tomo 32 frames por video y
   normalizo los landmarks respecto a los hombros para que no influya la distancia a la cámara."
3. **Arquitectura (1 min 30 s).** [Mostrar la figura] "Mi modelo tiene dos ramas. En la rama
   visual, una ResNet18 preentrenada y congelada convierte cada frame en 512 características, y
   un BiLSTM analiza cómo cambian en el tiempo. En la rama de landmarks, otro BiLSTM analiza el
   movimiento de los puntos junto con máscaras que indican si cada punto fue detectado. Cada
   rama produce un resumen de 128 valores; los concateno en 256, aplico dropout y una capa
   lineal clasifica entre las 34 palabras."
4. **Protocolo (40 s).** "La evaluación es cross-subject: las personas de test nunca se usan en
   entrenamiento, para medir si el modelo generaliza a firmantes nuevos. Elijo la mejor época
   con validación y uso macro-F1 como métrica principal porque las clases están desbalanceadas."
5. **Resultados (1 min).** "En la prueba con un sujeto nuevo obtuve 75 % de accuracy y
   macro-F1 de 0.70. Algunos pares mínimos como *Poor/Very_poor* se distinguen perfectamente,
   pero *Tall* se confunde siempre con *Very_tall*. Eso muestra que la diferencia de intensidad
   es lo más difícil de capturar."
6. **Siguiente paso (30 s).** "Vamos a comparar con el ST-GCN de mi compañero usando
   exactamente los mismos sujetos, y evaluar cada rama por separado para ver cuánto aporta la fusión."

### Preguntas probables y respuestas

| Pregunta | Respuesta |
|---|---|
| ¿Por qué la CNN está congelada? | Por la cantidad de datos y el entrenamiento en CPU. Una red preentrenada ya sabe extraer rasgos visuales; entrenarla completa con 4388 videos tendería a sobreajustar. Es una limitación: con GPU se podría ajustar (fine-tuning). |
| ¿Por qué BiLSTM y no solo CNN? | La CNN mira cada frame por separado; la seña es un movimiento. El BiLSTM modela la secuencia en ambos sentidos. |
| ¿Por qué fusionar? | RGB aporta forma de la mano y apariencia; los landmarks aportan posiciones precisas y robustas al fondo. Se compara con cada rama sola para comprobar si la fusión ayuda. |
| ¿Qué es macro-F1 y por qué no solo accuracy? | Es el promedio del F1 de cada palabra con igual peso. La accuracy puede verse alta aunque algunas palabras se reconozcan mal. |
| ¿Qué significa cross-subject? | Que las personas de test no aparecen en entrenamiento. Es la evaluación más realista. |
| ¿Por qué 32 frames? | Los videos tienen distinta duración; se toman 32 uniformemente para que todas las entradas tengan el mismo tamaño. |
| ¿Usan profundidad (depth)? | No en esta arquitectura. Solo RGB y landmarks. |
| ¿Es LOSO? | Todavía no. Es un split cross-subject fijo. LOSO completo sería repetirlo con cada sujeto de test. |
| ¿Cómo se evita hacer trampa con el test? | El test no se usa para elegir nada; la mejor época se elige con validación y el test se evalúa al final, sin reentrenar. |

---

## 10. Limitaciones (decirlas antes de que las pregunten)

1. ResNet18 congelada y entrada de 112 × 112: se puede perder detalle fino de los dedos.
2. Un solo split cross-subject, no LOSO completo.
3. Pocas muestras de test por palabra (~10 por sujeto): las métricas por palabra son sensibles.
4. No se usa depth.
5. Los pares mínimos aún requieren una anotación formal y su propio análisis.
6. Sin inferencia en tiempo real.

---

## 11. Llevar el proyecto a otra laptop

Todo lo necesario ya está **dentro** de `Proyecto_LSP` (copias; las carpetas originales no se tocaron):

| Qué | Dónde | Tamaño |
|---|---|---|
| Código | `codigo/` | < 1 MB |
| Índice común | `datos/indices/union_20261012_153325/` | ~1 MB |
| Landmarks | `datos/procesados/landmarks/datos_procesados_20261005_160031/` | ~140 MB |
| Características ResNet18 (4388) | `datos/caracteristicas/resnet18/` | ~190 MB |
| Modelos entrenados y resultados | `resultados/` | ~4 MB |

**No hace falta llevar** el dataset original (19 GB) ni `C3_RGB_Depth_112`: solo se usan
para preprocesar o volver a extraer características, y eso ya está hecho.
**No se copia** `.venv_fusion`: el entorno de Python se instala de nuevo en la otra laptop.

### Pasos
1. En esta PC ya está creado `C:\Users\PC\Desktop\Proyecto_LSP_portable.zip` (~320 MB). Copiarlo a USB o Drive.
2. En la otra laptop: instalar **Python 3.12 o 3.13** desde python.org (marcar *Add python.exe to PATH*).
3. Descomprimir el ZIP (por ejemplo en el Escritorio) y abrir PowerShell **dentro** de `Proyecto_LSP`.
4. Instalar el entorno (una sola vez, necesita internet):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\instalar_entorno.ps1
   ```
5. Mostrar el resultado del modelo ya entrenado (no reentrena, tarda segundos/minutos):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\demo_evaluar.ps1
   ```
   Imprime accuracy y macro-F1 y abre `matriz_test.html` en el navegador.

### Entrenar en la otra laptop (rutas relativas, funcionan en cualquier PC)
```powershell
$py = ".\.venv_fusion\Scripts\python.exe"
$indice = ".\datos\indices\union_20261012_153325\indice_comun.csv"
$landmarks = ".\datos\procesados\landmarks\datos_procesados_20261005_160031\landmarks_32f.npz"
$features = ".\datos\caracteristicas\resnet18"
$run = ".\resultados\fusion\split_comun_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $py .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice "$indice" --landmarks "$landmarks" --features "$features" --salida "$run" --modo fusion --epochs 50 --patience 10 --test-subjects 15 16 --val-subjects 1 5 10 --hilos 4
& $py .\codigo\modelos\cnn_bilstm\evaluar_checkpoint.py --experimento "$run"
```
Entrenar en CPU puede tardar bastante; para una demo en vivo es mejor **evaluar** un modelo
ya entrenado (paso 5) y mostrar `historial.csv` y la matriz.

`evaluar_checkpoint.py` acepta `--indice`, `--landmarks` y `--features` cuando los datos
están en otra ruta que la guardada en `config.json`; comprueba el hash, así que no se puede
evaluar por error con datos distintos.

---

## 12. Próximos pasos

1. Reentrenar la fusión con el split común (test 15, 16 / val 1, 5, 10).
2. Entrenar las variantes solo RGB y solo landmarks con el mismo split.
3. Hacer una tabla comparativa: BiLSTM (landmarks) vs CNN-BiLSTM (RGB) vs Fusión vs ST-GCN.
4. Definir la lista de pares mínimos y medir sus confusiones.
