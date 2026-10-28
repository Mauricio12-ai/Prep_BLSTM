# Alcance de la verificación

- Entorno de desarrollo: Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu.
- Extracción real sobre SUBJECT_1/Analyze/sample_1 con ResNet18 ImageNet.
- Comprobación de salida (32,512) finita y reanudación sin recalcular la muestra.
- Comparación de características con lotes de 1 y 16 imágenes en modo evaluación.
- Forward y backward de las tres variantes con las entradas reales de esa muestra.
- Bucle de entrenamiento/validación, checkpoint, recarga y exportación de matrices
  ejercitados con un conjunto sintético de 102 ejemplos. Es una prueba técnica;
  no se incluyen sus métricas como evidencia de reconocimiento de señas.
- El test solo se evalúa cuando se solicita --evaluar-test.

No se entrenó ni evaluó el conjunto real de 4388 muestras en este entorno.
Eso debe ejecutarse en el equipo donde se encuentran los datos. No hay pesos propios
entrenados ni resultados de precisión de la fusión incluidos en este paquete.
