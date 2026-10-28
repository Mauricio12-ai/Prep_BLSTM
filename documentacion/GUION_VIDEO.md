# Guion para grabar la explicación del código (rama CNN-BiLSTM)

Duración estimada: **15–18 minutos** (versión corta de ~8 min al final).
Cada escena dice **qué mostrar**, **qué decir** (puedes leerlo casi literal) y **qué ejecutar**.

---

## 0. Antes de grabar (checklist)

1. Abre **VS Code** con la carpeta `C:\Users\PC\Desktop\Proyecto_LSP` (Archivo → Abrir carpeta).
2. Sube el tamaño de letra para que se lea en el video: `Ctrl` + `+` dos o tres veces.
3. Abre una terminal **PowerShell** dentro de VS Code (Terminal → Nueva terminal) y comprueba que estás en `Proyecto_LSP`.
4. Ten listos para mostrar:
   - `documentacion\articulo\Figura1_pipeline_con_arquitectura.png` (pipeline)
   - `C:\Users\PC\Desktop\claude\LSP_figuras\arquitectura_fusion.png` (arquitectura)
5. Prueba **una vez** que la demo funciona (escena 9), para no tener sorpresas al grabar:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\demo_evaluar.ps1
   ```
6. Cierra notificaciones (WhatsApp, correo) y pestañas que no uses.
7. Para grabar la pantalla completa con voz: **Clipchamp** (viene con Windows) → *Grabar pantalla*, o **OBS Studio**.
   La Xbox Game Bar (`Win + Alt + R`) graba solo una ventana: si cambias a otra, no se ve.

> Consejo: no expliques línea por línea. Muestra la función, señala 2–3 líneas clave y di **qué hace y por qué**.

---

## Escena 1 — Presentación (1 min)

**Mostrar:** la figura del pipeline en pantalla completa.

**Decir:**
> "Hola, miss. Soy [nombre], y en este video voy a explicar el código de mi rama del proyecto
> *Comparación de arquitecturas de Deep Learning para el reconocimiento de palabras de la Lengua de
> Señas Peruana mediante minimal pairs*. Mi arquitectura es una **CNN-BiLSTM con fusión multimodal**:
> combina la imagen RGB del video con los landmarks del cuerpo y las manos. Voy a recorrer el pipeline
> en orden: los datos, el preprocesamiento, la extracción de características, el modelo, el
> entrenamiento y la evaluación, y al final mostraré los resultados ejecutando el código."

---

## Escena 2 — Organización del proyecto (1 min)

**Mostrar:** el explorador de VS Code con las carpetas desplegadas.

**Decir:**
> "El proyecto está organizado en carpetas. En `codigo/preprocesamiento` están los scripts que preparan
> los datos; en `codigo/modelos/cnn_bilstm` está mi arquitectura: `extraer_rgb.py`, `modelo.py`,
> `entrenar_fusion.py` y `evaluar_checkpoint.py`. En `datos` están el índice común, los landmarks
> procesados y las características visuales. Cada entrenamiento guarda sus resultados en una carpeta
> nueva dentro de `resultados`, así nunca se sobrescribe un experimento anterior."

---

## Escena 3 — Dataset y pipeline (1.5 min)

**Mostrar:** la Figura 1 (pipeline). Señala con el cursor cada bloque mientras hablas.

**Decir:**
> "El dataset tiene 4389 videos de 34 palabras, hechos por 17 personas. Cada muestra tiene video RGB,
> profundidad y un archivo `skeleton_real.txt` con 75 landmarks por frame: 33 del cuerpo y 21 de cada
> mano. Yo no extraje los landmarks; venían con el dataset.
> El pipeline tiene dos ramas. La de arriba es la mía: recibe **RGB y landmarks**. La de abajo es el
> ST-GCN de mi compañero, que recibe solo landmarks. Las dos ramas comparten el dataset, la división
> por sujetos y las métricas, para que la comparación sea justa. La profundidad no la uso."

---

## Escena 4 — Preprocesamiento de landmarks (2.5 min)

**Mostrar:** `codigo/preprocesamiento/landmarks/preprocess_landmarks_c3.py`

| Ir a la línea | Función | Qué señalar |
|---|---|---|
| 40 | `load_landmarks` | `reshape(-1, 75, 3)`: 225 columnas → 75 puntos × (x, y, z) |
| 48 | `interpolate_short_gaps` | `max_gap = 2`, y la condición `bounded and gap <= max_gap` |
| 80 | `torso_normalize` | punto medio de hombros y `np.median(valid_scale)` |
| 118 | `uniform_indices` | `np.rint(np.linspace(0, length - 1, target_frames))` |
| 124 | `quality_flag` | VERY_SHORT, HAND_NEVER_DETECTED, HIGH_HAND_MISSING |
| 237 | `np.savez_compressed` | guarda `landmarks_32f.npz` |

**Decir:**
> "Este script prepara los landmarks. Primero, `load_landmarks` lee cada archivo y convierte las 225
> columnas en 75 puntos con tres coordenadas.
> Después, `interpolate_short_gaps` trata los puntos que MediaPipe no detectó. Solo relleno huecos de
> hasta **dos frames** y solo si hay datos válidos antes y después; los huecos largos no los invento.
> Guardo dos máscaras: `observed`, si el punto se detectó, y `usable`, si se detectó o se interpoló.
> Luego `torso_normalize` centra todo en el punto medio de los hombros y lo divide por la distancia
> **mediana** entre hombros de la muestra. Así no importa si la persona está cerca o lejos de la cámara,
> ni si es más alta o más baja.
> Como cada video tiene distinta duración, `uniform_indices` toma **32 frames** repartidos de forma
> uniforme. La mediana del dataset es de 23 frames, así que en la mayoría de videos se repiten frames.
> Finalmente marco la calidad de cada muestra, pero no elimino ninguna, y guardo todo en un archivo NPZ."

---

## Escena 5 — RGB y unión de modalidades (1.5 min)

**Mostrar:** `codigo/preprocesamiento/union/unir_rgb_landmarks.py`, línea 14 (`EXCLUDED`) y línea 59.
Después abre `datos\indices\union_20261012_153325\indice_comun.csv` y muestra las columnas.

**Decir:**
> "Los frames RGB se prepararon con el mismo criterio: los **mismos 32 índices** que los landmarks, para
> que las dos modalidades estén alineadas en el tiempo, y redimensionados a 112 por 112 píxeles sin
> deformar ni recortar la imagen.
> Este script une las dos modalidades. Comprueba que cada muestra tenga RGB y landmarks con el mismo
> identificador, la misma clase y el mismo sujeto. Una muestra, `SUBJECT_2/Analyze/sample_2`, tenía una
> inconsistencia y se excluyó. El resultado es este índice común de **4388 muestras**: cada fila tiene
> el ID, la ruta del RGB, la fila de landmarks, la clase y el sujeto."

---

## Escena 6 — Extracción de características con ResNet18 (1.5 min)

**Mostrar:** `codigo/modelos/cnn_bilstm/extraer_rgb.py`

| Línea | Qué señalar |
|---|---|
| 23–37 (`encode`) | `/255`, `mean` y `std` de ImageNet; salida `(32, 512)` |
| 64 | `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` |
| 65 | `model.fc = nn.Identity()` |
| 66–67 | `model.eval()` y `requires_grad_(False)` |

**Decir:**
> "Aquí está la parte **CNN**. Uso una ResNet18 preentrenada en ImageNet. Le quito la última capa de
> clasificación con `nn.Identity`, entonces por cada frame me devuelve un vector de **512
> características**. La red está **congelada**: `requires_grad_(False)` significa que sus pesos no se
> entrenan, solo la uso como extractor.
> Como no cambia, calculo estas características **una sola vez** para las 4388 muestras y las guardo.
> Así el entrenamiento en CPU es mucho más rápido. Cada muestra queda como una matriz de 32 frames por 512."

---

## Escena 7 — La arquitectura (2 min)

**Mostrar:** primero la figura de la arquitectura; luego `codigo/modelos/cnn_bilstm/modelo.py` completo (27 líneas).

| Línea | Qué señalar |
|---|---|
| 13 | `nn.LSTM(512, hidden, bidirectional=True)` → rama RGB |
| 15 | `nn.LSTM(300, hidden, bidirectional=True)` → rama landmarks |
| 17 | `Dropout` + `Linear(width, 34)` |
| 23 y 26 | `torch.cat((h[-2], h[-1]))` → forward + backward |
| 27 | `torch.cat(vectors)` → **fusión** |

**Decir:**
> "Esta es mi arquitectura. Tiene dos ramas que funcionan en paralelo.
> La rama visual recibe las 32 características de 512 y las pasa por un **BiLSTM**: una LSTM lee la
> seña del inicio al final y otra del final al inicio. Tomo el estado final de cada dirección, 64 y 64,
> y los junto en un vector de **128**.
> La rama de landmarks hace lo mismo con otro BiLSTM, pero su entrada por frame son **300 valores**:
> 150 coordenadas XY, 75 de la máscara observada y 75 de la máscara usable. También da 128.
> En la línea 27 está la **fusión**: concateno los dos vectores en uno de **256**. Aplico dropout de
> 0.3 para evitar sobreajuste y una capa lineal da **34 puntuaciones**, una por palabra. La más alta es
> la predicción.
> El mismo código permite usar solo RGB o solo landmarks con el parámetro `modo`, para medir cuánto
> aporta la fusión. En total el modelo tiene 492 066 parámetros entrenables."

---

## Escena 8 — Entrenamiento (2.5 min)

**Mostrar:** `codigo/modelos/cnn_bilstm/entrenar_fusion.py`

| Línea | Qué señalar |
|---|---|
| 31 (`load_inputs`) | las verificaciones (`raise ValueError(...)`): 4388 muestras, mismos IDs, mismas clases |
| 48 | construcción del vector de 300 de landmarks |
| 87 (`particion`) | división **por sujeto** con `np.isin` |
| 130 | `Adam` y `CrossEntropyLoss` |
| 150–157 | bucle: pérdida → `backward` → `clip_grad_norm_(…, 1.0)` → `step` |
| 163–167 | guardar el mejor modelo por macro-F1 y early stopping |

**Decir:**
> "Este script entrena el modelo. Primero `load_inputs` carga todo y **verifica** que coincidan los IDs,
> las clases y los sujetos entre el índice, los landmarks y las características RGB. Si algo no
> coincide, se detiene con un error en vez de entrenar con datos mezclados.
> En la línea 48 se arma el vector de 300 valores: multiplico las coordenadas por la máscara usable,
> para que los puntos no confiables valgan cero, y le agrego las dos máscaras.
> La función `particion` divide los datos **por sujeto**: las personas de prueba nunca se usan para
> entrenar. Así medimos si el modelo funciona con personas nuevas.
> En cada época el modelo procesa lotes de 32 muestras, calcula la pérdida de entropía cruzada,
> retropropaga el error y Adam ajusta los pesos. El recorte de gradiente a 1.0 evita saltos bruscos.
> Al final de cada época evalúo en validación. Si el **macro-F1** mejora, guardo el modelo; si pasan
> 10 épocas sin mejorar, se detiene. El conjunto de prueba **no se usa** para elegir nada."

**Mostrar en la terminal (no lo ejecutes, solo explica):**
```powershell
& $py .\codigo\modelos\cnn_bilstm\entrenar_fusion.py --indice "$indice" --landmarks "$landmarks" --features "$features" --salida "$run" --modo fusion --epochs 50 --patience 10 --test-subjects 15 16 --val-subjects 1 5 10
```
> "Este es el comando: `--modo fusion` entrena mi arquitectura completa, y `--test-subjects` y
> `--val-subjects` indican qué personas van a prueba y a validación."

**Mostrar:** `resultados\fusion\entrenamiento_50_20261012_163623\historial.csv`
> "Este es el historial de un entrenamiento. En las primeras épocas la pérdida baja rápido. El mejor
> macro-F1 de validación fue en la **época 18**. Después la pérdida de entrenamiento siguió bajando pero
> la de validación empezó a subir, que es señal de sobreajuste, y el early stopping lo detuvo en la 28."

---

## Escena 9 — Demostración en vivo: evaluación (2 min)

**Ejecutar en la terminal:**
```powershell
powershell -ExecutionPolicy Bypass -File .\demo_evaluar.ps1
```

**Decir mientras corre:**
> "Ahora ejecuto `evaluar_checkpoint.py`. Carga el mejor modelo guardado **sin volver a entrenarlo** y lo
> prueba con el sujeto de test, que el modelo nunca vio. También verifica con un hash que los datos sean
> exactamente los mismos con los que se entrenó."

**Cuando termine, mostrar la salida y decir:**
> "El resultado: de 393 videos de prueba acertó **295**, que es una exactitud de **75.1 %**, y un
> macro-F1 de **0.70**. Como referencia, adivinar al azar entre 34 palabras daría alrededor de 3 %."

Se abre `matriz_test.html` en el navegador. **Mostrar y decir:**
> "Esta es la matriz de confusión. Las filas son la palabra real y las columnas la predicha. La diagonal
> verde son los aciertos y lo rojo, las confusiones."

---

## Escena 10 — Resultados y pares mínimos (1.5 min)

**Mostrar:** en la matriz, busca la fila **Tall** y la columna **Very_tall**; luego **Poor / Very_poor**.
Opcional: abre `metricas_por_clase_test.csv` de la misma carpeta.

**Decir:**
> "Lo más interesante para nuestro tema son los pares mínimos. Por ejemplo, *Poor* y *Very_poor* se
> distinguen perfectamente, igual que *Learn_everything* y *Learn_something*. Pero *Tall* se confundió
> **las 11 veces** con *Very_tall*: el modelo no logra captar esa diferencia de intensidad.
> También hay palabras difíciles como *Progress*, que confunde con *Stagnate*, o *People* con *Analyze*.
> La lista definitiva de pares mínimos la vamos a validar con usted antes del análisis final."

---

## Escena 11 — Limitaciones y cierre (1 min)

**Mostrar:** la figura de la arquitectura otra vez.

**Decir:**
> "Algunas limitaciones: la ResNet18 está congelada y trabaja con imágenes de 112 píxeles, por lo que
> puede perder detalle fino de los dedos. Este resultado es de una sola partición por sujeto, no de un
> LOSO completo, y cada palabra tiene unos 10 videos de prueba, así que un solo error cambia bastante su
> métrica.
> Este resultado corresponde a la partición anterior, con un sujeto de prueba. El siguiente paso es
> reentrenar con la partición común acordada con mi compañero, de 12, 3 y 2 sujetos, comparar con su
> ST-GCN y con las variantes solo RGB y solo landmarks, y analizar los pares mínimos validados.
> Gracias, miss."

---

## Versión corta (~8 min)

Escenas **1 → 3 → 7 → 8** (solo la parte del bucle y el historial) **→ 9 → 10 → 11**.
En la 3, menciona el preprocesamiento en dos frases: "relleno huecos cortos, normalizo respecto a los
hombros y tomo 32 frames; la ResNet18 congelada convierte cada frame RGB en 512 características".

---

## Preguntas que te puede hacer después (respuestas cortas)

| Pregunta | Respuesta |
|---|---|
| ¿Por qué congelada la ResNet18? | Pocos datos y entrenamiento en CPU; ya sabe extraer rasgos visuales y entrenarla entera tendería a sobreajustar. |
| ¿Por qué BiLSTM? | La seña es movimiento; leerla en ambos sentidos captura el inicio y el final del gesto. |
| ¿Por qué macro-F1? | Las clases tienen distinta cantidad de muestras; macro-F1 da el mismo peso a las 34 palabras. |
| ¿Para qué las máscaras? | Para que el modelo distinga un cero real de un punto que no se detectó. |
| ¿Por qué 32 frames? | Los videos tienen distinta duración y el modelo necesita una entrada fija; es una configuración inicial. |
| ¿El test influye en el entrenamiento? | No. La mejor época se elige con validación; el test se evalúa una vez al final. |
| ¿Usas profundidad? | No en esta arquitectura. |
