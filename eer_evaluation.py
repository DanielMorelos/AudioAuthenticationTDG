"""
eer_evaluation.py
=================
Evaluación con Equal Error Rate (EER) para el detector de empalmes de audio.
 
Dos niveles de evaluación
--------------------------
1. EER a nivel de VENTANA   → ¿el modelo separa bien ventanas de distintos micrófonos?
2. EER a nivel de AUDIO     → ¿el sistema detecta correctamente audios empalmados?
 
Uso rápido (pega esto al final de tu notebook, después de entrenar):
---------------------------------------------------------------------
    from eer_evaluation import evaluar_eer_completo
 
    resultados = evaluar_eer_completo(
        model           = model,          # EmbeddingNet ya entrenado
        test_audios     = test_audios,    # lista de dicts crudos reservados al inicio
        device_to_label = device_to_label,
        mean_global     = mean_global,
        std_global      = std_global,
        device          = device,
        n_audios_test   = 60,             # cuántos pares generar para la prueba
        guardar_pdf     = True,           # guarda "reporte_eer.pdf"
    )
    print(resultados)
"""
 
import numpy as np
import torch
import librosa
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
 
# ══════════════════════════════════════════════════════════════════════════════
# 1.  Utilidades internas
# ══════════════════════════════════════════════════════════════════════════════
 
def _mel_ventana(ventana: np.ndarray, sr: int = 16000,
                 mean_global: float = 0.0, std_global: float = 1.0) -> np.ndarray:
    """Mel-espectrograma normalizado de una ventana de audio."""
    mel = librosa.feature.melspectrogram(y=ventana, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)   # por instancia
    mel_norm = (mel_norm - mean_global) / (std_global + 1e-8)      # global
    return mel_norm
 
 
def _embeddings_audio(audio: np.ndarray, model, device,
                      mean_global: float, std_global: float,
                      sr: int = 16000, win_sec: float = 1.0,
                      hop_sec: float = 0.5) -> tuple[np.ndarray, list[float]]:
    """
    Segmenta un audio en ventanas deslizantes y extrae embeddings.
    Devuelve (embeddings [N, D], timestamps [N]).
    """
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)
    ventanas, timestamps = [], []
 
    for start in range(0, len(audio) - win_len, hop_len):
        v = audio[start:start + win_len]
        ventanas.append(_mel_ventana(v, sr, mean_global, std_global))
        timestamps.append(start / sr)
 
    if not ventanas:
        return np.empty((0, 1)), []
 
    specs = torch.tensor(np.array(ventanas), dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        embs = model(specs).cpu().numpy()
 
    return embs, timestamps
 
 
def _distancias_consecutivas(embs: np.ndarray) -> np.ndarray:
    """Distancia euclidiana entre embeddings de ventanas contiguas."""
    if len(embs) < 2:
        return np.array([])
    return np.array([np.linalg.norm(embs[i] - embs[i - 1]) for i in range(1, len(embs))])
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2.  Cálculo del EER
# ══════════════════════════════════════════════════════════════════════════════
 
def calcular_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Calcula el Equal Error Rate y el threshold óptimo.
 
    Convención de labels
    --------------------
    - label = 1  →  par GENUINO   (mismo dispositivo, distancia debería ser BAJA)
    - label = 0  →  par IMPOSTOR  (distinto dispositivo, distancia debería ser ALTA)
 
    Como roc_curve espera scores altos = clase positiva, invertimos las distancias.
 
    Devuelve
    --------
    eer       : float  (0-1)
    threshold : float  (en la escala original de distancias)
    """
    # sklearn espera que "score alto → clase 1 (genuino)"
    # Nuestras distancias son bajas para genuinos → invertimos
    scores_inv = -scores
 
    fpr, tpr, thresholds = roc_curve(labels, scores_inv, pos_label=1)
    fnr = 1.0 - tpr
 
    # Punto donde FAR ≈ FRR
    idx_eer = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2)
 
    # Convertir threshold de vuelta a la escala original (distancia)
    threshold_optimo = float(-thresholds[idx_eer])
 
    return eer, threshold_optimo
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3.  EER a nivel de VENTANA
# ══════════════════════════════════════════════════════════════════════════════
 
def eer_nivel_ventana(model, test_audios: list[dict], device_to_label: dict,
                      mean_global: float, std_global: float, device,
                      sr: int = 16000) -> dict:
    """
    Para cada ventana del test set extrae el embedding y compara con
    la ventana anterior dentro del mismo audio.
 
    Score de cada par (i-1, i):  distancia euclidiana entre embeddings.
    Label del par:
        1 = genuino  → ambas ventanas pertenecen al MISMO dispositivo
        0 = impostor → la ventana cruza un punto de empalme (device_a ≠ device_b)
 
    Para construir los pares con ground-truth limpio, generamos audios
    empalmados y auténticos desde test_audios, igual que en el pipeline
    principal, y utilizamos el campo `cut_point` para saber qué pares
    son genuinos y cuáles son impostores.
    """
    from itertools import combinations
 
    dispositivos = list(set(a['device'] for a in test_audios))
 
    # ── Generar pares de ventanas con etiqueta ────────────────────────────────
    all_scores, all_labels = [], []
 
    # A) Pares genuinos: ventanas dentro de un audio auténtico
    for audio_info in test_audios:
        embs, _ = _embeddings_audio(
            audio_info['audio'], model, device, mean_global, std_global, sr
        )
        dists = _distancias_consecutivas(embs)
        # Todas las ventanas son del mismo dispositivo → genuinas
        all_scores.extend(dists.tolist())
        all_labels.extend([1] * len(dists))
 
    # B) Pares impostores: ventanas alrededor del punto de empalme
    audios_por_dev = {d: [a for a in test_audios if a['device'] == d]
                      for d in dispositivos}
 
    for dev_a in dispositivos:
        for dev_b in dispositivos:
            if dev_a == dev_b:
                continue
            if not audios_por_dev[dev_a] or not audios_por_dev[dev_b]:
                continue
 
            a_info = random.choice(audios_por_dev[dev_a])
            b_info = random.choice(audios_por_dev[dev_b])
 
            # Crear empalme artificial
            y1, y2 = a_info['audio'], b_info['audio']
            intervals = librosa.effects.split(y1, top_db=30)
            if len(intervals) < 2:
                continue
            idx_corte = random.randint(0, len(intervals) - 2)
            cut = intervals[idx_corte][1]
            y_spliced = np.concatenate([y1[:cut], y2[cut:]])
 
            embs, timestamps = _embeddings_audio(
                y_spliced, model, device, mean_global, std_global, sr
            )
            dists = _distancias_consecutivas(embs)
            if len(dists) == 0:
                continue
 
            hop_len = int(0.5 * sr)
            win_len = int(1.0 * sr)
 
            # Label por par de ventanas: impostor si cruza el cut_point
            for i, dist in enumerate(dists):
                start_prev = i * hop_len
                end_curr   = (i + 1) * hop_len + win_len
                cruza_corte = (start_prev <= cut <= end_curr)
                label = 0 if cruza_corte else 1  # 0=impostor, 1=genuino
                all_scores.append(dist)
                all_labels.append(label)
 
    scores = np.array(all_scores)
    labels = np.array(all_labels)
 
    eer, threshold_optimo = calcular_eer(scores, labels)
 
    return {
        'eer':              eer,
        'threshold_optimo': threshold_optimo,
        'scores':           scores,
        'labels':           labels,
        'n_genuinos':       int((labels == 1).sum()),
        'n_impostores':     int((labels == 0).sum()),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4.  EER a nivel de AUDIO
# ══════════════════════════════════════════════════════════════════════════════
 
def eer_nivel_audio(model, test_audios: list[dict], device_to_label: dict,
                    mean_global: float, std_global: float, device,
                    n_audios_test: int = 60, sr: int = 16000) -> dict:
    """
    Cada audio produce un único score = distancia máxima entre ventanas
    consecutivas. Los audios genuinos deberían tener score bajo; los
    empalmados, alto.
 
    Label:
        1 = audio AUTÉNTICO   (score debería ser bajo)
        0 = audio EMPALMADO   (score debería ser alto)
    """
    dispositivos = list(set(a['device'] for a in test_audios))
    audios_por_dev = {d: [a for a in test_audios if a['device'] == d]
                      for d in dispositivos}
 
    all_scores, all_labels = [], []
    n_por_clase = n_audios_test // 2
 
    # ── Audios AUTÉNTICOS ─────────────────────────────────────────────────────
    for _ in range(n_por_clase):
        dev = random.choice(dispositivos)
        if not audios_por_dev[dev]:
            continue
        a_info = random.choice(audios_por_dev[dev])
        embs, _ = _embeddings_audio(
            a_info['audio'], model, device, mean_global, std_global, sr
        )
        dists = _distancias_consecutivas(embs)
        if len(dists) == 0:
            continue
        all_scores.append(float(dists.max()))
        all_labels.append(1)   # auténtico
 
    # ── Audios EMPALMADOS ─────────────────────────────────────────────────────
    for _ in range(n_por_clase):
        dev_a, dev_b = random.sample(dispositivos, 2)
        if not audios_por_dev[dev_a] or not audios_por_dev[dev_b]:
            continue
        y1 = random.choice(audios_por_dev[dev_a])['audio']
        y2 = random.choice(audios_por_dev[dev_b])['audio']
 
        intervals = librosa.effects.split(y1, top_db=30)
        if len(intervals) < 2:
            continue
        cut = intervals[random.randint(0, len(intervals) - 2)][1]
        y_spliced = np.concatenate([y1[:cut], y2[cut:]])
 
        embs, _ = _embeddings_audio(
            y_spliced, model, device, mean_global, std_global, sr
        )
        dists = _distancias_consecutivas(embs)
        if len(dists) == 0:
            continue
        all_scores.append(float(dists.max()))
        all_labels.append(0)   # empalmado
 
    scores = np.array(all_scores)
    labels = np.array(all_labels)
 
    eer, threshold_optimo = calcular_eer(scores, labels)
 
    return {
        'eer':              eer,
        'threshold_optimo': threshold_optimo,
        'scores':           scores,
        'labels':           labels,
        'n_autenticos':     int((labels == 1).sum()),
        'n_empalmados':     int((labels == 0).sum()),
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5.  Reporte visual completo
# ══════════════════════════════════════════════════════════════════════════════
 
def _curva_det(ax, scores, labels, threshold_optimo, titulo):
    """Dibuja la curva DET (FAR vs FRR) y marca el punto EER."""
    fpr, tpr, thresholds = roc_curve(labels, -scores, pos_label=1)
    fnr = 1.0 - tpr
 
    ax.plot(fpr * 100, fnr * 100, color='steelblue', linewidth=2, label='Curva DET')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.4, label='Azar')
 
    # Marcar punto EER
    idx = np.argmin(np.abs(fpr - fnr))
    eer_pct = (fpr[idx] + fnr[idx]) / 2 * 100
    ax.scatter([fpr[idx] * 100], [fnr[idx] * 100],
               color='red', s=80, zorder=5,
               label=f'EER = {eer_pct:.1f}%  (thr={threshold_optimo:.3f})')
 
    ax.set_xlabel('FAR — False Acceptance Rate (%)')
    ax.set_ylabel('FRR — False Rejection Rate (%)')
    ax.set_title(titulo, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
 
 
def _histograma_scores(ax, scores, labels, threshold_optimo, titulo):
    """Histograma de distribución de scores por clase."""
    genuinos   = scores[labels == 1]
    impostores = scores[labels == 0]
 
    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(genuinos,   bins=bins, alpha=0.6, color='steelblue',
            label=f'Auténticos  (n={len(genuinos)})',   density=True)
    ax.hist(impostores, bins=bins, alpha=0.6, color='tomato',
            label=f'Empalmados  (n={len(impostores)})', density=True)
    ax.axvline(threshold_optimo, color='black', linestyle='--', linewidth=1.5,
               label=f'Umbral óptimo = {threshold_optimo:.3f}')
 
    ax.set_xlabel('Score (distancia máxima entre embeddings)')
    ax.set_ylabel('Densidad')
    ax.set_title(titulo, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
 
def generar_reporte_eer(res_ventana: dict, res_audio: dict,
                        guardar_pdf: bool = True) -> plt.Figure:
    """
    Produce una figura con 4 paneles:
    - Panel 1 & 2 : Curva DET  (ventana / audio)
    - Panel 3 & 4 : Histograma (ventana / audio)
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"EVALUACIÓN EER — Detector de Empalce de Audio\n"
        f"{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
 
    # ── Curvas DET ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _curva_det(ax1,
               res_ventana['scores'], res_ventana['labels'],
               res_ventana['threshold_optimo'],
               f"Curva DET — Nivel VENTANA\nEER = {res_ventana['eer']*100:.2f}%")
 
    ax2 = fig.add_subplot(gs[0, 1])
    _curva_det(ax2,
               res_audio['scores'], res_audio['labels'],
               res_audio['threshold_optimo'],
               f"Curva DET — Nivel AUDIO\nEER = {res_audio['eer']*100:.2f}%")
 
    # ── Histogramas ───────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _histograma_scores(ax3,
                       res_ventana['scores'], res_ventana['labels'],
                       res_ventana['threshold_optimo'],
                       "Distribución de Scores — Nivel VENTANA")
 
    ax4 = fig.add_subplot(gs[1, 1])
    _histograma_scores(ax4,
                       res_audio['scores'], res_audio['labels'],
                       res_audio['threshold_optimo'],
                       "Distribución de Scores — Nivel AUDIO")
 
    # ── Tabla de resumen ──────────────────────────────────────────────────────
    resumen = (
        f"  Nivel VENTANA →  EER: {res_ventana['eer']*100:.2f}%   "
        f"Umbral óptimo: {res_ventana['threshold_optimo']:.4f}   "
        f"(genuinos: {res_ventana['n_genuinos']} | impostores: {res_ventana['n_impostores']})  "
        f"\n  Nivel AUDIO   →  EER: {res_audio['eer']*100:.2f}%   "
        f"Umbral óptimo: {res_audio['threshold_optimo']:.4f}   "
        f"(auténticos: {res_audio['n_autenticos']} | empalmados: {res_audio['n_empalmados']})"
    )
    fig.text(0.5, 0.01, resumen, ha='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff', edgecolor='steelblue'))
 
    if guardar_pdf:
        fig.savefig("reporte_eer.pdf", bbox_inches='tight', dpi=150)
        print("Reporte guardado en reporte_eer.pdf")
 
    plt.show()
    return fig
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6.  Función principal de entrada
# ══════════════════════════════════════════════════════════════════════════════
 
def evaluar_eer_completo(model,
                         test_audios: list[dict],
                         device_to_label: dict,
                         mean_global: float,
                         std_global: float,
                         device,
                         n_audios_test: int = 60,
                         guardar_pdf: bool = True,
                         sr: int = 16000) -> dict:
    """
    Calcula EER a nivel de ventana y a nivel de audio, genera el reporte
    visual y devuelve un dict con todos los resultados.
 
    Parámetros
    ----------
    model           : EmbeddingNet entrenado
    test_audios     : lista de dicts crudos (con claves 'audio', 'sr', 'device')
    device_to_label : dict {'huawei_nova_9': 0, ...}
    mean_global     : media global calculada durante el entrenamiento final
    std_global      : desviación estándar global
    device          : torch.device
    n_audios_test   : cuántos audios sintéticos generar para el EER a nivel audio
    guardar_pdf     : si True guarda "reporte_eer.pdf"
    sr              : frecuencia de muestreo
 
    Devuelve
    --------
    dict con claves:
        'ventana' : resultados EER a nivel ventana
        'audio'   : resultados EER a nivel audio
        'threshold_recomendado' : umbral sugerido para generar_reporte_forense()
    """
    print("── Calculando EER a nivel de VENTANA ──")
    res_ventana = eer_nivel_ventana(
        model, test_audios, device_to_label,
        mean_global, std_global, device, sr
    )
    print(f"   EER ventana : {res_ventana['eer']*100:.2f}%  "
          f"| umbral óptimo : {res_ventana['threshold_optimo']:.4f}")
 
    print("── Calculando EER a nivel de AUDIO ────")
    res_audio = eer_nivel_audio(
        model, test_audios, device_to_label,
        mean_global, std_global, device,
        n_audios_test, sr
    )
    print(f"   EER audio   : {res_audio['eer']*100:.2f}%  "
          f"| umbral óptimo : {res_audio['threshold_optimo']:.4f}")
 
    # El umbral recomendado para el reporte forense viene del nivel AUDIO
    # (es la decisión final sobre el audio completo)
    threshold_recomendado = res_audio['threshold_optimo']
    print(f"\n✔ Umbral recomendado para generar_reporte_forense(): "
          f"{threshold_recomendado:.4f}")
 
    generar_reporte_eer(res_ventana, res_audio, guardar_pdf)
 
    return {
        'ventana':               res_ventana,
        'audio':                 res_audio,
        'threshold_recomendado': threshold_recomendado,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7.  Snippet listo para pegar al final del notebook
# ══════════════════════════════════════════════════════════════════════════════
#
#   from eer_evaluation import evaluar_eer_completo
#
#   resultados_eer = evaluar_eer_completo(
#       model           = model,
#       test_audios     = test_audios,
#       device_to_label = device_to_label,
#       mean_global     = mean_global,
#       std_global      = std_global,
#       device          = device,
#       n_audios_test   = 60,
#       guardar_pdf     = True,
#   )
#
#   # Usar el umbral calculado automáticamente en el reporte forense:
#   resultado = generar_reporte_forense(
#       audio_sospechoso = audio_con_splicing,
#       modelo           = model,
#       device           = device,
#       threshold        = resultados_eer['threshold_recomendado'],
#   )
 