# AudioAuthenticationTDG

**Detección y localización de empalmes entre dispositivos móviles mediante aprendizaje de métricas de distancia**

Prototipo de autenticación forense de audio desarrollado como trabajo de grado en la Escuela de Ingenierías Eléctrica, Electrónica y Telecomunicaciones (E3T), Universidad Industrial de Santander (UIS).

---

## Descripción

Las grabaciones de audio usadas como evidencia pueden alterarse mediante **empalmes** (*splicing*): fragmentos grabados con distintos dispositivos se combinan en un mismo archivo. Este proyecto aborda el problema como una **verificación de consistencia entre segmentos**: cada micrófono deja una huella en la señal, y un salto de distancia entre ventanas consecutivas sugiere un cambio de fuente.

El sistema:

1. Divide el audio en ventanas de 1 s y calcula espectrogramas STFT (log-potencia, 1025 × 22).
2. Procesa cada ventana con una CNN 2D de pesos compartidos (estructura siamesa) que produce un *embedding* de 64 dimensiones normalizado en L2.
3. Calcula la distancia entre embeddings de ventanas consecutivas (salto de 0.25 s). El **máximo** de la secuencia es la puntuación del audio.
4. Declara el audio empalmado si la puntuación supera un **umbral calibrado por EER** en validación y luego congelado.
5. Localiza el empalme como el instante del pico de distancia.

La red se entrena con **CosFace** usando únicamente segmentos **no vocalizados y ruido de fondo** (detectados con pYIN) del corpus POLIPHONE.

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `training.py` | Clase `Trainer`: bucle de entrenamiento con scheduler, detención temprana, recorte de gradiente y checkpoints reanudables. |
| `eer_evaluation.py` | Evaluación a nivel de ventana y de audio (EER, FAR, FRR), generación de empalmes sintéticos, estrategias de localización (base, mediana, entropía), evaluación de empalmes con IA y reportes. |
| Notebook principal | Pipeline completo: carga de datos, partición, extracción de fragmentos no vocalizados, validación cruzada, entrenamiento, calibración, evaluaciones finales, bootstrap y diagnósticos. Pensado para ejecutarse en **Google Colab**. |

## Datos

| Conjunto | Uso | Fuente |
|---|---|---|
| **POLIPHONE** (20 teléfonos, habla, 44.1 kHz) | Entrenamiento, validación y prueba | [Zenodo](https://doi.org/10.5281/zenodo.13903412) (CC BY 4.0) |
| **MOBIPHONE** | Dispositivos no vistos (generalización) | [liweigithub128/AudioDataset](https://github.com/liweigithub128/AudioDataset) |
| Empalmes y segmentos generados con IA | Evaluación de manipulación con voz sintética | [BasededatosAutenticaciondeAudios](https://github.com/DanielMorelos/BasededatosAutenticaciondeAudios) |

El notebook descarga los subconjuntos necesarios (`original_recordings`, `ia_splicing_audio`, `ia_pure_audio`) con un *sparse checkout* del repositorio de datos.

> Los empalmes con IA se generaron con Silero VAD, Whisper y XTTS v2 (clonación de voz). Los detalles de la cadena están en el Anexo K del informe.

## Instalación y uso

### Requisitos

- Python 3.10+
- GPU recomendada para entrenar (la inferencia es viable en CPU)

```bash
pip install torch torchaudio librosa soundfile scikit-learn scipy numpy pandas \
            matplotlib seaborn tqdm joblib pytorch-metric-learning torcheval thop pymupdf
```

### Ejecución en Google Colab

1. Copie `training.py` y `eer_evaluation.py` a `MyDrive/AudioAuthentication/`.
2. Abra el notebook principal y monte Google Drive.
3. Ajuste `output_path` y `run_name` en la sección de configuración.
4. Ejecute las celdas en orden. El pipeline guarda en Drive los datos intermedios, checkpoints, modelo, tablas y figuras, y puede **reanudarse** si ya existen resultados (`SKIP_IF_KFOLD_COMPLETE`, `SKIP_IF_FINAL_MODEL_EXISTS`, `RESUME_FROM_CHECKPOINT`).

### Uso de los módulos

```python
from eer_evaluation import evaluate_eer_complete

results = evaluate_eer_complete(
    model=model, test_audios=test_audios, device_to_label=device_to_label,
    device=device, n_audios_test=200,
    sr=44100, n_fft=2048, hop_length_fft=2048,
    win_sec=1.0, hop_sec=0.25, top_db=30,
    fixed_threshold={"window": tau_w, "audio": tau_a},  # umbral congelado
    distance_metric="euclidean",
)
```

## Configuración principal

| Parámetro | Valor |
|---|---|
| Frecuencia de muestreo | 44.1 kHz |
| Ventana / salto (entrenamiento) | 1 s / 1 s (sin traslape) |
| Ventana / salto (evaluación) | 1 s / 0.25 s |
| STFT | `n_fft = 2048`, `hop = 2048`, ventana Hann, log-potencia + z-score por ventana |
| Embedding | 64 dimensiones, norma L2 |
| Pérdida | CosFace (`m = 0.2`, `s = 30`) |
| Optimizador | AdamW (`lr = 1e-3`, `weight decay = 1e-4`) |
| Lote | 128 (MPerClassSampler) |
| Métricas de distancia | Euclidiana, Manhattan, Mahalanobis (Ledoit-Wolf), SNR |
| Semilla | 42 |

## Resultados principales

Todos los umbrales se calibran una sola vez en validación y se congelan. Detalles, intervalos de confianza y tablas completas en el informe.

| Aspecto | Resultado |
|---|---|
| Separabilidad de dispositivos vistos (k-NN, prueba) | **96.19 %** |
| EER por ventana, prueba (euclidiana) | 19.4 % |
| EER por audio, prueba | 23.5 – 32 % según la métrica |
| Empalmes con IA, detección por audio (EER propio) | ≈ 21 % |
| Dispositivos no vistos, EER con umbral propio | 0.5 – 4 % (el umbral congelado **no** se transfiere) |
| Parámetros / costo por ventana | 912 576 / ≈ 0.89 GMACs |
| Tiempo para analizar un audio de ~75 s | 5.7 s en CPU · 0.55 s en GPU (T4) |

**Lectura de los resultados:** el espacio de embeddings separa bien los dispositivos conocidos y conserva el orden de las distancias en dispositivos nuevos, pero el umbral calibrado depende del dominio. Con voz sintetizada por IA el sistema detecta la presencia del empalme a nivel de audio, pero no su ubicación.

## Limitaciones

- Un solo corpus de entrenamiento (20 teléfonos, habla en inglés, cámara anecoica) y sin códecs, mensajería ni grabaciones de campo.
- Los empalmes sintéticos son cortes secos en pausas; en cada uno cambian también la voz y el contenido.
- MOBIPHONE difiere del corpus principal en ancho de banda, ruido y voces, por lo que el efecto del micrófono no puede aislarse.
- Hiperparámetros fijados a priori, sin búsqueda ni análisis de sensibilidad.
- Evaluaciones con 100 audios por clase: diferencias de pocos puntos entre métricas no son concluyentes.
- **Prototipo académico, no validado para uso judicial.**

## Trabajo futuro

- Recalibración del umbral con una muestra pequeña del dominio nuevo.
- Entrenamiento con más dispositivos, condiciones de captura y ruido real.
- Localización con modelado temporal de la secuencia de distancias y múltiples picos candidatos.
- Evaluación con compresión (MP3, AAC, Opus) y manipulaciones más realistas.
- Combinación con un detector de artefactos de síntesis para empalmes con IA.

## Cita

```bibtex
@misc{morelos2026audioauth,
  author = {Morelos, Daniel},
  title  = {AudioAuthenticationTDG: Detección y localización de empalmes entre dispositivos móviles mediante aprendizaje de métricas},
  year   = {2026},
  url    = {https://github.com/DanielMorelos/AudioAuthenticationTDG}
}
```

## Referencias clave

- Salvi, D. et al. (2024). *POLIPHONE: A dataset for smartphone model identification from audio recordings.* Zenodo.
- Kotropoulos, C., & Samaras, S. (2014). *Mobile phone identification using recorded speech signals.* IEEE DSP.
- Wang, H. et al. (2018). *CosFace: Large margin cosine loss for deep face recognition.* CVPR.
- Leonzio, D. U. et al. (2023). *Audio splicing detection and localization based on acquisition device traces.* IEEE TIFS.
- Calderón Ardila, E. F. (2023). *Enfoque de aprendizaje de distancias para la detección de cambios de micrófono en archivos de audio.* UIS.

## Licencia

Por definir. Los datos de POLIPHONE se distribuyen bajo CC BY 4.0; consulte las condiciones de uso de MOBIPHONE antes de redistribuirlo.

## Autor

**Daniel Morelos** · Escuela de Ingenierías Eléctrica, Electrónica y Telecomunicaciones (E3T), Universidad Industrial de Santander.
