"""
eer_evaluation.py
==================
Equal Error Rate (EER) evaluation for the audio splice detector.

Two evaluation levels
----------------------
1. WINDOW-level EER : does the model separate windows from different microphones well?
2. AUDIO-level EER  : does the system correctly detect spliced audios?
"""

import numpy as np
import torch
import librosa
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from sklearn.metrics import roc_curve
import os
import csv
import pickle as _pickle
from sklearn.metrics import roc_curve, confusion_matrix
from scipy.signal import medfilt
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis as _scipy_mahalanobis
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# 0.  Splice-localization strategy registry
# ══════════════════════════════════════════════════════════════════════════════
# Single source of truth for the localization strategies compared in every
# audio-level report, so labels/colors/ordering are never hardcoded per plot.
LOCALIZATION_STRATEGY_KEYS = ("baseline", "median", "entropy", "median_entropy")

LOCALIZATION_STRATEGY_LABELS = {
    "baseline":       "Baseline\n(raw-distance argmax)",
    "median":         "Median-filtered\n(distance argmax)",
    "entropy":        "Entropy-weighted\n(raw distances)",
    "median_entropy": "Median-filtered +\nEntropy-weighted",
}

LOCALIZATION_STRATEGY_COLORS = {
    "baseline":       "darkorange",
    "median":         "mediumpurple",
    "entropy":        "seagreen",
    "median_entropy": "indianred",
}

LOCALIZATION_ERROR_POPULATIONS = (
    ("localization_error_stats",          "All spliced audios", 0.85, None),
    ("localization_error_stats_detected", "Detected only",      0.55, "//"),
)
# Module-level switch for progress bars (set to False to silence them)
SHOW_PROGRESS_BARS = True
# Minimum share of a window's content that must switch source between two
# consecutive windows for the pair to be labeled impostor. Pairs whose cut
# falls closer to the union edge than this are ambiguous (both windows are
# almost entirely one source) and are excluded from the window-level EER.
# Upper bound is hop_sec / win_sec (0.25 with the current evaluation config).
WINDOW_LEVEL_MIN_CROSS_SOURCE_FRACTION = 0.10
# Upper bound on splice-generation attempts, expressed as a multiple of the
# requested number of spliced audios. 
SPLICE_MAX_ATTEMPTS_FACTOR = 10
# Splice validity / audio-level fairness
SPLICE_MIN_EDGE_MARGIN_WINDOWS = 1.0      # min distance from the cut to either audio edge, in analysis windows
SPLICE_MIN_EDGE_MARGIN_EXTRA_HOPS = 1
AUDIO_LEVEL_EQUALIZE_DURATION = True      # crop authentic and spliced audios to the shortest audio in the pool
AUDIO_LEVEL_MIN_DURATION_WINDOWS = 3.0    # min audio duration, in analysis windows, to enter the audio-level EER
def _point_localization_error_s(detected_time_s: float, true_time_s: float) -> float:
    """Absolute localization error against a single ground-truth cut instant."""
    return float(abs(detected_time_s - true_time_s))


def _interval_localization_error_s(detected_time_s: float,
                                    interval_start_s: float,
                                    interval_end_s: float) -> float:
    """
    Localization error against a ground-truth insertion interval: zero inside
    the interval, distance to the nearest boundary outside it. Used by the
    IA-splicing set, whose ground truth is [insert_start_s, insert_end_s].
    """
    if detected_time_s < interval_start_s:
        return float(interval_start_s - detected_time_s)
    if detected_time_s > interval_end_s:
        return float(detected_time_s - interval_end_s)
    return 0.0


def _localization_error_statistics(errors_by_strategy: dict) -> dict:
    """
    Aggregates per-strategy localization errors into mean / std / median,
    skipping strategies with no samples (e.g. entropy comparison disabled or
    median filtering not configured for this run).
    """
    statistics = {}
    for strategy_key in LOCALIZATION_STRATEGY_KEYS:
        errors = errors_by_strategy.get(strategy_key)
        if not errors:
            continue
        errors_array = np.asarray(errors, dtype=np.float64)
        statistics[strategy_key] = {
            "mean_error_s":   float(errors_array.mean()),
            "std_error_s":    float(errors_array.std()),
            "median_error_s": float(np.median(errors_array)),
            "n_samples":      int(errors_array.size),
        }
    return statistics


def _format_localization_error_summary(audio_results: dict) -> str:
    """
    One-line textual rendering of the per-strategy mean ± std localization
    error, for both the unconditional and the detected-only populations.
    """
    summary_lines = []
    for results_key, population_label, _, _ in LOCALIZATION_ERROR_POPULATIONS:
        error_statistics = audio_results.get(results_key) or {}
        if not error_statistics:
            continue
        summary_parts = []
        for strategy_key in LOCALIZATION_STRATEGY_KEYS:
            if strategy_key not in error_statistics:
                continue
            label = LOCALIZATION_STRATEGY_LABELS[strategy_key].replace("\n", " ")
            strategy_stats = error_statistics[strategy_key]
            summary_parts.append(
                f"{label}: {strategy_stats['mean_error_s']:.4f} ± "
                f"{strategy_stats['std_error_s']:.4f} s (n={strategy_stats['n_samples']})"
            )
        summary_lines.append(f"[{population_label}] " + "   |   ".join(summary_parts))
    return "\n  ".join(summary_lines)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Internal utilities
# ══════════════════════════════════════════════════════════════════════════════

def _fft_log_power_window(window: np.ndarray, sr: int = None,
                           #global_mean: float = None, global_std: float = None,
                           n_fft: int = None, hop_length: int = None) -> np.ndarray:
    """
    2D normalized FFT (log-power) spectrogram of an audio window.
    extract_fft_log_power() from the notebook so that EER is consistent
    with what the model saw during training.
    """
    stft = librosa.stft(window, n_fft=n_fft, hop_length=hop_length, window='hann')
    power_spec = np.abs(stft) ** 2
    log_power_spec = 10 * np.log10(power_spec + 1e-10)   # shape: (freq_bins, n_frames)

    feat_norm = (log_power_spec - log_power_spec.mean()) / (log_power_spec.std() + 1e-8)  # per instance
    #feat_norm = (feat_norm - global_mean) / (global_std + 1e-8)                            # global
    return feat_norm


def _audio_embeddings(audio: np.ndarray, model,
                       #global_mean: float = None, global_std: float = None,
                       device=None,
                       sr: int = None, win_sec: float = None,
                       hop_sec: float = None,
                       n_fft: int = None, hop_length_fft: int = None) -> tuple[np.ndarray, list[float]]:
    """
    Segments an audio into sliding windows and extracts embeddings.
    Returns (embeddings [N, D], timestamps [N]).
    """
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)
    windows, timestamps = [], []

    for start in range(0, len(audio) - win_len, hop_len):
        w = audio[start:start + win_len]
        windows.append(_fft_log_power_window(w, sr=sr, #global_mean=global_mean, global_std=global_std,
                                              n_fft=n_fft, hop_length=hop_length_fft))
        timestamps.append(start / sr)

    if not windows:
        return np.empty((0, 1)), []

    specs = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        embs = model(specs).cpu().numpy()

    return embs, timestamps

def _consecutive_distances(embs: np.ndarray, distance_metric: str = "euclidean",
                            VI: np.ndarray = None) -> np.ndarray:
    """Distance (Euclidean or Mahalanobis) between embeddings of contiguous windows."""
    if len(embs) < 2:
        return np.array([])
    return np.array([
        _pairwise_distance(embs[i], embs[i - 1], distance_metric=distance_metric, VI=VI)
        for i in range(1, len(embs))
    ])
# ══════════════════════════════════════════════════════════════════════════════
# 2b.  Distance metric abstraction (Euclidean / Mahalanobis)
# ══════════════════════════════════════════════════════════════════════════════


def estimate_mahalanobis_vi(reference_embeddings: np.ndarray,
                             eigenvalue_floor_ratio: float = 1e-10) -> np.ndarray:
    """
    Estimates the inverse covariance matrix (VI) required by the
    Mahalanobis distance, from a reference population of embeddings.

    Parameters
    ----------
    reference_embeddings : np.ndarray, shape (n_samples, embedding_dim)
    eigenvalue_floor_ratio : float
        Relative floor applied to eigenvalues before inversion, as a
        fraction of the largest eigenvalue. Scale-invariant.

    Returns
    -------
    VI : np.ndarray, shape (embedding_dim, embedding_dim)
    """
    covariance_estimator = LedoitWolf().fit(reference_embeddings)
    covariance_matrix = covariance_estimator.covariance_
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalue_floor = eigenvalues.max() * eigenvalue_floor_ratio
    eigenvalues_clipped = np.clip(eigenvalues, eigenvalue_floor, None)

    inverse_covariance = eigenvectors @ np.diag(1.0 / eigenvalues_clipped) @ eigenvectors.T
    inverse_covariance = (inverse_covariance + inverse_covariance.T) / 2
    return inverse_covariance


def _pairwise_distance(vec_a: np.ndarray, vec_b: np.ndarray,
                        distance_metric: str = "euclidean",
                        VI: np.ndarray = None) -> float:

    if distance_metric == "euclidean":
        return float(np.linalg.norm(vec_a - vec_b))
    elif distance_metric == "manhattan":
        return float(np.sum(np.abs(vec_a - vec_b)))
    elif distance_metric == "mahalanobis":
        if VI is None:
            raise ValueError("VI (inverse covariance matrix) is required for distance_metric='mahalanobis'.")
        return float(_scipy_mahalanobis(vec_a, vec_b, VI))
    elif distance_metric == "snr":
        # Matches pytorch_metric_learning.distances.SNRDistance:
        # SNR(query, ref) = Var(query - ref) / Var(query).
        # NOTE: asymmetric  vec_a is treated as the query (unbiased variance, ddof=1).
        query_variance = np.var(vec_a, ddof=1)
        if query_variance == 0.0:
            return float("inf")
        return float(np.var(vec_a - vec_b, ddof=1) / query_variance)
    else:
        raise ValueError(
            f"Unknown distance_metric: {distance_metric!r}. "
            "Use 'euclidean', 'manhattan', 'mahalanobis' or 'snr'."
        )
    
def _apply_median_filter(distances: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Smooths a single audio's consecutive-window distance sequence with a
    1D median filter.
 
    Parameters
    ----------
    distances   : 1D array of consecutive-window distances for ONE audio.
    kernel_size : window size of the median filter. Must be odd; kernel_size
                  <= 1 disables filtering entirely.
 
    Returns
    -------
    Filtered distance sequence. If kernel_size <= 1 or the sequence is
    shorter than kernel_size (scipy.signal.medfilt's own requirement), the
    original, unfiltered sequence is returned unchanged.
    """
    if kernel_size <= 1 or len(distances) < kernel_size:
        return distances
    return medfilt(distances, kernel_size=kernel_size)

def _local_distance_entropy(distances: np.ndarray, entropy_window: int) -> np.ndarray:
    """
    Computes the local Shannon entropy of the embedding-distance sequence
    over a sliding window centered at each index.

    Parameters
    ----------
    distances : np.ndarray
        1D array of consecutive-window embedding distances for a single
        audio (as produced by `_consecutive_distances`, optionally already
        passed through `_apply_median_filter`).
    entropy_window : int
        Size of the local sliding window used to estimate entropy at each
        index. Must be passed explicitly by the caller (see
        `evaluate_ia_splicing_detection`) rather than hardcoded here.

    Returns
    -------
    np.ndarray
        Local entropy values, same length as `distances`.
    """
    half_window = entropy_window // 2
    local_entropy = np.zeros(len(distances))

    for i in range(len(distances)):
        lo = max(0, i - half_window)
        hi = min(len(distances), i + half_window + 1)
        local_segment = distances[lo:hi]
        probabilities = local_segment / (local_segment.sum() + 1e-10)
        local_entropy[i] = -np.sum(probabilities * np.log(probabilities + 1e-10))

    return local_entropy


def entropy_weighted_localization(distances: np.ndarray, timestamps: list[float],
                                    entropy_window: int) -> dict:
    """
    Alternative splice-localization estimate that combines distance
    magnitude with local peak sharpness (inverse entropy), instead of
    relying on raw distance magnitude alone (`argmax(distances)`).

    score[i] = distances[i] / (local_entropy[i] + eps)

    A high score requires both a large distance AND a sharply peaked
    local neighborhood, penalizing peaks that are large but diffuse
    (more likely to be noise) relative to peaks that are large and
    isolated (more likely to be a genuine splice).

    Parameters
    ----------
    distances : np.ndarray
        Consecutive-window embedding distances for a single audio.
    timestamps : list[float]
        Timestamps aligned with `distances` (i.e. `timestamps[1:]` from
        `_audio_embeddings`, matching the convention used elsewhere in
        this module).
    entropy_window : int
        Local window size passed to `_local_distance_entropy`.

    Returns
    -------
    dict
        entropy_detected_time_s : float
            Timestamp of the entropy-weighted candidate.
        entropy_max_distance : float
            Raw distance at that candidate index.
        entropy_score : float
            Combined score at that candidate index.
        local_entropy_at_peak : float
            Local entropy value at that candidate index.
    """
    local_entropy = _local_distance_entropy(distances, entropy_window=entropy_window)
    combined_score = distances / (local_entropy + 1e-10)
    best_idx = int(np.argmax(combined_score))

    return {
        'entropy_detected_time_s': timestamps[best_idx],
        'entropy_max_distance': float(distances[best_idx]),
        'entropy_score': float(combined_score[best_idx]),
        'local_entropy_at_peak': float(local_entropy[best_idx]),
    }

def generate_simple_splicing(y1: np.ndarray, y2: np.ndarray, top_db: int = None,
                             min_edge_margin_samples: int = None):
    """
    Generates a splice between two audio signals.
    Single source of truth for this logic, used both in training
    (notebook) and in EER evaluation, so both always match.

    Parameters
    ----------
    min_edge_margin_samples : if given, the cut is restricted to
        [min_edge_margin_samples, min(len(y1), len(y2)) - min_edge_margin_samples],
        guaranteeing that both sides of the splice contain real content from
        each source (avoids degenerate "splices" when y2 is shorter than the cut).
        If None, the original unconstrained behavior is kept.

    Returns
    -------
    y_spliced : np.ndarray or None if the cut could not be generated
    cut       : int (sample where the cut occurs) or None
    """
    intervals = librosa.effects.split(y1, top_db=top_db)
    if len(intervals) < 2:
        return None, None

    candidate_cuts = intervals[:-1, 1]
    if min_edge_margin_samples is not None:
        #upper_bound = len(y2) - min_edge_margin_samples
        upper_bound = min(len(y1), len(y2)) - min_edge_margin_samples
        is_valid_cut = (candidate_cuts >= min_edge_margin_samples) & (candidate_cuts <= upper_bound)
        candidate_cuts = candidate_cuts[is_valid_cut]
        if len(candidate_cuts) == 0:
            return None, None

    cut = int(candidate_cuts[random.randint(0, len(candidate_cuts) - 1)])
    y_spliced = np.concatenate([y1[:cut], y2[cut:]])

    return y_spliced, cut

# ══════════════════════════════════════════════════════════════════════════════
# 2.  EER computation
# ══════════════════════════════════════════════════════════════════════════════

def calculate_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """
    Computes the Equal Error Rate and the optimal threshold.

    Label convention
    -----------------
    - label = 1  :  GENUINE pair   (same device, distance should be LOW)
    - label = 0  :  IMPOSTOR pair  (different device, distance should be HIGH)

    Since roc_curve expects high scores = positive class, distances are inverted.

    Returns
    -------
    eer                : float  (0-1)
    optimal_threshold   : float  (in the original distance scale)
    """
    scores_inv = -scores

    fpr, tpr, thresholds = roc_curve(labels, scores_inv, pos_label=1)
    fnr = 1.0 - tpr

    idx_eer = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2)

    optimal_threshold = float(-thresholds[idx_eer])

    return eer, optimal_threshold

def calculate_far_frr_at_threshold(scores: np.ndarray, labels: np.ndarray,
                                    threshold: float) -> tuple[float, float]:
    """
    Computes FAR and FRR using an ALREADY FIXED threshold (does not search/optimize it).
    Used to report performance on an evaluation set without
    recalibrating the threshold with that same data (avoids leakage).

    label = 1 : genuine/authentic   (score should be LOW)
    label = 0 : impostor/spliced    (score should be HIGH)
    """
    pred_impostor = scores > threshold  # True = the system flags it as impostor/spliced

    is_impostor = (labels == 0)
    is_genuine = (labels == 1)

    # FAR: real impostors the system did NOT detect (let through as genuine)
    far = float((~pred_impostor[is_impostor]).sum() / is_impostor.sum()) if is_impostor.sum() > 0 else float('nan')

    # FRR: real genuine samples the system wrongly flagged as impostors
    frr = float(pred_impostor[is_genuine].sum() / is_genuine.sum()) if is_genuine.sum() > 0 else float('nan')

    return far, frr

def calculate_confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """
    2x2 confusion matrix for the genuine(1)/impostor(0) detection task.
    Prediction: impostor if score > threshold.
    Fixed label order labels=[0,1] -> rows/cols = [impostor, genuine].
    """
    preds = np.where(scores > threshold, 0, 1)
    return confusion_matrix(labels, preds, labels=[0, 1])
# ══════════════════════════════════════════════════════════════════════════════
# 3.  WINDOW-level EER
# ══════════════════════════════════════════════════════════════════════════════

def eer_window_level(model, test_audios: list[dict], device_to_label: dict,
                      #global_mean: float = None, global_std: float = None,
                      device=None,
                      sr: int = None,
                      win_sec: float = None, hop_sec: float = None,
                      n_fft: int = None, hop_length_fft: int = None,
                      top_db: int = None, fixed_threshold: float = None, 
                      #median_filter_kernel_size: int = None, 
                      distance_metric: str = "euclidean", VI: np.ndarray = None,
                      min_cross_source_fraction: float = WINDOW_LEVEL_MIN_CROSS_SOURCE_FRACTION) -> dict:
    """
    For each window in the test set, the embedding is extracted and compared
    with the previous window within the same audio.

    Score of each pair (i-1, i):  distance between embeddings.
    Pair label:
        1 = genuine  : both windows belong to the SAME device
        0 = impostor : the window crosses a splice point (device_a != device_b)

    """
    devices = sorted(set(a['device'] for a in test_audios))
    
    all_scores, all_labels = [], []

    # A) Genuine pairs: windows within an authentic audio
    for audio_info in tqdm(test_audios,
                           desc="Window-level EER | genuine pairs",
                           disable=not SHOW_PROGRESS_BARS):
        embs, _ = _audio_embeddings(
            audio_info['audio'], model, device=device,
            sr=sr, win_sec=win_sec, hop_sec=hop_sec,
            n_fft=n_fft, hop_length_fft=hop_length_fft
        )
        dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
        all_scores.extend(dists.tolist())
        all_labels.extend([1] * len(dists))

    # B) Impostor pairs: windows around the splice point
    audios_by_device = {d: [a for a in test_audios if a['device'] == d]
                        for d in devices}

    for dev_a in tqdm(devices,
                    desc="Window-level EER | impostor pairs (by source device)",
                    disable=not SHOW_PROGRESS_BARS):
        for dev_b in devices:
            if dev_a == dev_b:
                continue
            if not audios_by_device[dev_a] or not audios_by_device[dev_b]:
                continue

            a_info = random.choice(audios_by_device[dev_a])
            b_info = random.choice(audios_by_device[dev_b])

            y1, y2 = a_info['audio'], b_info['audio']
            y_spliced, cut = generate_simple_splicing(y1, y2, top_db=top_db)
            if y_spliced is None:
                continue

            embs, timestamps = _audio_embeddings(
                y_spliced, model, device=device, #global_mean=global_mean, global_std=global_std,
                sr=sr, win_sec=win_sec, hop_sec=hop_sec,
                n_fft=n_fft, hop_length_fft=hop_length_fft
            )
            dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
            if len(dists) == 0:
                continue
            #dists = _apply_median_filter(dists, median_filter_kernel_size) if median_filter_kernel_size else dists

            hop_len = int(hop_sec * sr)
            win_len = int(win_sec * sr)

            for i, dist in enumerate(dists):
                prev_start = i * hop_len
                curr_start = (i + 1) * hop_len

                # Fraction of each window lying AFTER the cut (i.e. source B content)
                source_b_fraction_prev = np.clip((prev_start + win_len - cut) / win_len, 0.0, 1.0)
                source_b_fraction_curr = np.clip((curr_start + win_len - cut) / win_len, 0.0, 1.0)
                cross_source_fraction = source_b_fraction_curr - source_b_fraction_prev

                if cross_source_fraction >= min_cross_source_fraction:
                    label = 0        # impostor
                elif cross_source_fraction <= 0.0:
                    label = 1        # genuine
                else:
                    continue         # ambiguous: cut too close to the union edge

                all_scores.append(dist)
                all_labels.append(label)

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if fixed_threshold is None:
        eer, optimal_threshold = calculate_eer(scores, labels)
    else:
        optimal_threshold = fixed_threshold
        eer, _ = calculate_eer(scores, labels)  # informative, not used to set the threshold

    far, frr = calculate_far_frr_at_threshold(scores, labels, optimal_threshold)
    conf_matrix = calculate_confusion_matrix(scores, labels, optimal_threshold)
    return {
        'eer':                eer,
        'optimal_threshold':  optimal_threshold,
        'far_at_threshold':   far,
        'frr_at_threshold':   frr,
        'scores':             scores,
        'labels':             labels,
        'n_genuine':          int((labels == 1).sum()),
        'n_impostors':        int((labels == 0).sum()),
        'confusion_matrix':   conf_matrix,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 4.  AUDIO-level EER
# ══════════════════════════════════════════════════════════════════════════════

def eer_audio_level(model, test_audios: list[dict], device_to_label: dict,
                     #global_mean: float, global_std: float, 
                     device=None, n_audios_test: int = None, sr: int = None,
                     win_sec: float = None, hop_sec: float = None,
                     n_fft: int = None, hop_length_fft: int = None,
                     top_db: int = None, fixed_threshold: float = None,
                     compute_entropy_comparison: bool = False,
                     entropy_window: int = None, localization_tolerance_s: float = None, 
                     median_filter_kernel_size: int = None, distance_metric: str = "euclidean", 
                     VI: np.ndarray = None) -> dict:
    """
    Each audio produces a single score = maximum distance between consecutive
    windows. Genuine audios should have a low score; spliced ones, a high one.

    Label:
        1 = AUTHENTIC audio   (score should be low)
        0 = SPLICED audio     (score should be high)
    """

    min_edge_margin_samples = int(
            (SPLICE_MIN_EDGE_MARGIN_WINDOWS * win_sec
            + SPLICE_MIN_EDGE_MARGIN_EXTRA_HOPS * hop_sec) * sr
        )
    min_duration_samples = int(AUDIO_LEVEL_MIN_DURATION_WINDOWS * win_sec * sr)

    # Audios too short to host a valid splice (edge margin on both sides) are excluded
    # up front, so a single short recording cannot shrink the common length of the pool.
    eligible_audios = [a for a in test_audios if len(a['audio']) >= min_duration_samples]
    n_excluded = len(test_audios) - len(eligible_audios)
    if n_excluded:
        print(f"NOTE: {n_excluded}/{len(test_audios)} audios shorter than "
              f"{AUDIO_LEVEL_MIN_DURATION_WINDOWS * win_sec:.2f}s excluded from the audio-level EER.")

    devices = sorted(set(a['device'] for a in eligible_audios))
    if len(devices) < 2:
        raise ValueError("Audio-level EER requires eligible audios from at least 2 devices.")
    audios_by_device = {d: [a for a in eligible_audios if a['device'] == d] for d in devices}

    common_length_samples = (min(len(a['audio']) for a in eligible_audios)
                             if AUDIO_LEVEL_EQUALIZE_DURATION else None)

    def _crop_to_common_length(audio_signal: np.ndarray) -> np.ndarray:
        return audio_signal if common_length_samples is None else audio_signal[:common_length_samples]

    all_scores, all_labels = [], []
    n_per_class = n_audios_test // 2

    # AUTHENTIC audios sampled without replacement from the eligible pool, so each
    # audio is scored at most once and every device is represented in proportion to
    # its real number of audios (no per-device equal-probability bias).
    n_authentic_requested = n_per_class
    n_authentic_available = len(eligible_audios)
    n_authentic_to_sample = min(n_authentic_requested, n_authentic_available)
    if n_authentic_to_sample < n_authentic_requested:
        print(f"WARNING: requested {n_authentic_requested} authentic audios but the pool "
              f"only has {n_authentic_available}; using {n_authentic_to_sample} unique audios.")

    for a_info in random.sample(eligible_audios, n_authentic_to_sample):
        embs, _ = _audio_embeddings(
            _crop_to_common_length(a_info['audio']), model, device=device,   
            sr=sr, win_sec=win_sec, hop_sec=hop_sec,
            n_fft=n_fft, hop_length_fft=hop_length_fft
        )
        dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
        if len(dists) == 0:
            continue
        all_scores.append(float(dists.max()))
        all_labels.append(1)   # authentic

    # SPLICED audios
    localization_records = []
    spliced_combos = []
    n_spliced_generated = 0
    n_splice_attempts = 0
    max_splice_attempts = n_per_class * SPLICE_MAX_ATTEMPTS_FACTOR
    while n_spliced_generated < n_per_class and n_splice_attempts < max_splice_attempts:
        n_splice_attempts += 1
        dev_a, dev_b = random.sample(devices, 2)
        if not audios_by_device[dev_a] or not audios_by_device[dev_b]:
            continue
        a_choice = random.choice(audios_by_device[dev_a])
        b_choice = random.choice(audios_by_device[dev_b])
        y1 = _crop_to_common_length(a_choice['audio'])                        
        y2 = _crop_to_common_length(b_choice['audio'])                        
        y_spliced, cut = generate_simple_splicing(                            
            y1, y2, top_db=top_db,
            min_edge_margin_samples=min_edge_margin_samples,
        )
        if y_spliced is None:
            continue

        embs, timestamps = _audio_embeddings(
            y_spliced, model, device=device,
            sr=sr, win_sec=win_sec, hop_sec=hop_sec,
            n_fft=n_fft, hop_length_fft=hop_length_fft
        )
        raw_dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
        if len(raw_dists) == 0:
            continue
        #dists = (_apply_median_filter(raw_dists, median_filter_kernel_size)
                 #if median_filter_kernel_size else raw_dists)
        filtered_dists = (_apply_median_filter(raw_dists, median_filter_kernel_size)
                  if median_filter_kernel_size else None)
        
        dist_timestamps = timestamps[1:]
        max_idx = int(np.argmax(raw_dists))   # baseline strategy = raw-distance argmax

        #all_scores.append(float(dists.max()))
        all_scores.append(float(raw_dists.max()))
        all_labels.append(0)   # spliced
        spliced_combos.append((a_choice.get('condition'), b_choice.get('condition')))

        localization_records.append({
            'max_distance':       float(raw_dists.max()),
            'detected_time_s':  dist_timestamps[max_idx],
            'true_cut_time_s':  cut / sr,
            'raw_distances':    raw_dists,
            'filtered_distances': filtered_dists, 
            'distances':          raw_dists,            
            'dist_timestamps':  dist_timestamps,
        })
        n_spliced_generated += 1
    if n_spliced_generated < n_per_class:
        print(f"WARNING: only {n_spliced_generated}/{n_per_class} spliced audios generated "
              f"after {n_splice_attempts} attempts; classes are imbalanced.")
    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if fixed_threshold is None:
        eer, optimal_threshold = calculate_eer(scores, labels)
    else:
        optimal_threshold = fixed_threshold
        eer, _ = calculate_eer(scores, labels)  # informative, not used to set the threshold

    localization_rate = None
    localization_rate_entropy = None
    localization_error_stats = {}
    localization_error_stats_detected = {}

    if localization_records:
        errors_by_strategy = {key: [] for key in LOCALIZATION_STRATEGY_KEYS}
        errors_by_strategy_detected = {key: [] for key in LOCALIZATION_STRATEGY_KEYS}

        for record in localization_records:
            record['detected'] = record['max_distance'] > optimal_threshold
            true_cut_time_s = record['true_cut_time_s']

            baseline_error_s = _point_localization_error_s(record['detected_time_s'], true_cut_time_s)
            record['baseline_localization_error_s'] = baseline_error_s
            errors_by_strategy['baseline'].append(baseline_error_s)
            if record['detected']:
                errors_by_strategy_detected['baseline'].append(baseline_error_s)

            if median_filter_kernel_size:
                median_max_idx = int(np.argmax(record['filtered_distances']))
                median_detected_time_s = record['dist_timestamps'][median_max_idx]
                median_error_s = _point_localization_error_s(median_detected_time_s, true_cut_time_s)
                record['median_detected_time_s'] = median_detected_time_s
                record['median_localization_error_s'] = median_error_s
                errors_by_strategy['median'].append(median_error_s)
                if record['detected']:
                    errors_by_strategy_detected['median'].append(median_error_s)

            if compute_entropy_comparison:
                if entropy_window is None:
                    raise ValueError(
                        "entropy_window must be provided when compute_entropy_comparison=True"
                    )
                entropy_result = entropy_weighted_localization(
                    record['raw_distances'], record['dist_timestamps'], entropy_window=entropy_window
                )
                record.update(entropy_result)
                entropy_error_s = _point_localization_error_s(
                    entropy_result['entropy_detected_time_s'], true_cut_time_s
                )
                record['entropy_localization_error_s'] = entropy_error_s
                errors_by_strategy['entropy'].append(entropy_error_s)
                if record['detected']:
                    errors_by_strategy_detected['entropy'].append(entropy_error_s)

                if median_filter_kernel_size:
                    median_entropy_result = entropy_weighted_localization(
                        record['filtered_distances'], record['dist_timestamps'],
                        entropy_window=entropy_window
                    )
                    median_entropy_error_s = _point_localization_error_s(
                        median_entropy_result['entropy_detected_time_s'], true_cut_time_s
                    )
                    record['median_entropy_detected_time_s'] = median_entropy_result['entropy_detected_time_s']
                    record['median_entropy_localization_error_s'] = median_entropy_error_s
                    errors_by_strategy['median_entropy'].append(median_entropy_error_s)
                    if record['detected']:
                        errors_by_strategy_detected['median_entropy'].append(median_entropy_error_s)

            if localization_tolerance_s is not None:
                record['correct_localization'] = record['detected'] and baseline_error_s <= localization_tolerance_s
                if compute_entropy_comparison:
                    record['entropy_correct_localization'] = (
                        record['detected'] and record['entropy_localization_error_s'] <= localization_tolerance_s
                    )

        localization_error_stats = _localization_error_statistics(errors_by_strategy)
        localization_error_stats_detected = _localization_error_statistics(errors_by_strategy_detected)

        if localization_tolerance_s is not None:
            localization_rate = float(np.mean([r['correct_localization'] for r in localization_records]))
            if compute_entropy_comparison:
                localization_rate_entropy = float(
                    np.mean([r['entropy_correct_localization'] for r in localization_records])
                )

    far, frr = calculate_far_frr_at_threshold(scores, labels, optimal_threshold)
    conf_matrix = calculate_confusion_matrix(scores, labels, optimal_threshold)
    return {
        'eer':                    eer,
        'optimal_threshold':      optimal_threshold,
        'far_at_threshold':       far,
        'frr_at_threshold':       frr,
        'scores':                 scores,
        'labels':                 labels,
        'n_authentic':            int((labels == 1).sum()),
        'n_spliced':              int((labels == 0).sum()),
        'confusion_matrix':       conf_matrix,
        'localization_rate':          localization_rate,
        'localization_rate_entropy':  localization_rate_entropy,
        'localization_tolerance_s':   localization_tolerance_s,
        'localization_records':       localization_records,
        'localization_error_stats':   localization_error_stats,
        'localization_error_stats_detected': localization_error_stats_detected,
    }

# Full visual report

def _det_curve(ax, scores, labels, optimal_threshold, title):
    """Draws the DET curve (FAR vs FRR) and marks the EER point."""
    fpr, tpr, thresholds = roc_curve(labels, -scores, pos_label=1)
    fnr = 1.0 - tpr

    ax.plot(fpr * 100, fnr * 100, color='steelblue', linewidth=2, label='DET Curve')
    ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.4, label='Chance')

    idx = np.argmin(np.abs(fpr - fnr))
    eer_pct = (fpr[idx] + fnr[idx]) / 2 * 100
    ax.scatter([fpr[idx] * 100], [fnr[idx] * 100],
               color='red', s=80, zorder=5,
               label=f'EER = {eer_pct:.1f}%  (thr={optimal_threshold:.3f})')

    ax.set_xlabel('FAR — False Acceptance Rate (%)')
    ax.set_ylabel('FRR — False Rejection Rate (%)')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)


def _score_histogram(ax, scores, labels, optimal_threshold, title):
    """Histogram of score distribution per class."""
    genuine   = scores[labels == 1]
    impostors = scores[labels == 0]

    bins = np.linspace(scores.min(), scores.max(), 40)
    ax.hist(genuine,   bins=bins, alpha=0.6, color='steelblue',
            label=f'Authentic  (n={len(genuine)})',   density=True)
    ax.hist(impostors, bins=bins, alpha=0.6, color='tomato',
            label=f'Spliced  (n={len(impostors)})', density=True)
    ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=1.5,
               label=f'Optimal threshold = {optimal_threshold:.3f}')

    ax.set_xlabel('Score (maximum distance between embeddings)')
    ax.set_ylabel('Density')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

def _plot_confusion_matrix(ax, cm, class_names, title, normalize=None):
    cm_plot = cm.astype(float)
    if normalize == "true":
        cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm_plot = cm_plot / cm_plot.sum(axis=0, keepdims=True)

    im = ax.imshow(cm_plot, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=15)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontweight='bold')

    fmt = ".4f" if normalize else "d"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_plot[i, j] if normalize else cm[i, j]
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    color="white" if cm_plot[i, j] > cm_plot.max()/2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)

def _ia_splicing_rate_summary(ax, audio_results: dict, title: str) -> None:
    """Bar chart summarizing detection and localization rates for the IA-splicing audio-level evaluation."""
    labels = ["Detection Rate", f"Localization Rate\n(±{audio_results['tolerance_s']}s)"]
    values = [audio_results['detection_rate'], audio_results['localization_rate']]

    bars = ax.bar(labels, values, color=['steelblue', 'darkorange'], alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}",
                ha='center', fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"{title}\n(n = {audio_results['n_samples']} samples, "
                 f"threshold = {audio_results['threshold_used']:.4f})", fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

def _localization_comparison_summary(ax, audio_results: dict, title: str) -> None:
    """
    Bar chart comparing baseline (argmax-distance) vs entropy-weighted
    splice-localization accuracy, when both were computed for this
    evaluation run (see `compute_entropy_comparison` in eer_audio_level /
    evaluate_ia_splicing_detection). Bars are added dynamically depending
    on which metrics are present in `audio_results`, so the same helper
    works for both the synthetic-splicing and IA-splicing reports.
    """
    tolerance_s = audio_results.get('localization_tolerance_s', audio_results.get('tolerance_s'))

    bar_labels, bar_values, bar_colors = [], [], []

    if audio_results.get('detection_rate') is not None:
        bar_labels.append("Detection Rate")
        bar_values.append(audio_results['detection_rate'])
        bar_colors.append('steelblue')

    bar_labels.append(f"Localization Rate\n(baseline, ±{tolerance_s}s)")
    bar_values.append(audio_results['localization_rate'])
    bar_colors.append('darkorange')

    if audio_results.get('localization_rate_median') is not None:
        bar_labels.append(f"Localization Rate\n(median-filtered, ±{tolerance_s}s)")
        bar_values.append(audio_results['localization_rate_median'])
        bar_colors.append('mediumpurple')

    if audio_results.get('localization_rate_entropy') is not None:
        bar_labels.append(f"Localization Rate\n(entropy-weighted, ±{tolerance_s}s)")
        bar_values.append(audio_results['localization_rate_entropy'])
        bar_colors.append('seagreen')

    if audio_results.get('localization_rate_median_entropy') is not None:
        bar_labels.append(f"Localization Rate\n(median + entropy, ±{tolerance_s}s)")
        bar_values.append(audio_results['localization_rate_median_entropy'])
        bar_colors.append('indianred')

    bars = ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.85)
    for bar, value in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2%}",
                ha='center', fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(title, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

def _localization_error_summary(ax, audio_results: dict, title: str) -> None:
    """
    Grouped bar chart of the mean ± standard deviation of the splice-localization
    error (seconds) per strategy, reported over two populations: every spliced
    audio (threshold-independent localization ability) and only those audios the
    detector actually flagged (operational localization accuracy). Strategies are
    drawn in LOCALIZATION_STRATEGY_KEYS order, and only those present in the
    unconditional statistics are shown.
    """
    reference_statistics = audio_results.get('localization_error_stats') or {}
    strategy_keys = [key for key in LOCALIZATION_STRATEGY_KEYS if key in reference_statistics]
    if not strategy_keys:
        ax.axis('off')
        return

    populations = [
        (population_label, audio_results.get(results_key) or {}, bar_alpha, bar_hatch)
        for results_key, population_label, bar_alpha, bar_hatch in LOCALIZATION_ERROR_POPULATIONS
        if audio_results.get(results_key)
    ]

    strategy_positions = np.arange(len(strategy_keys), dtype=np.float64)
    bar_width = 0.8 / max(len(populations), 1)
    upper_bounds = []

    for population_index, (population_label, error_statistics, bar_alpha, bar_hatch) in enumerate(populations):
        offsets = strategy_positions + (population_index - (len(populations) - 1) / 2) * bar_width
        mean_errors, std_errors, sample_counts = [], [], []
        for strategy_key in strategy_keys:
            strategy_stats = error_statistics.get(strategy_key)
            mean_errors.append(strategy_stats['mean_error_s'] if strategy_stats else 0.0)
            std_errors.append(strategy_stats['std_error_s'] if strategy_stats else 0.0)
            sample_counts.append(strategy_stats['n_samples'] if strategy_stats else 0)

        bars = ax.bar(offsets, mean_errors, bar_width, yerr=std_errors, capsize=5,
                      color=[LOCALIZATION_STRATEGY_COLORS[key] for key in strategy_keys],
                      alpha=bar_alpha, hatch=bar_hatch, edgecolor='white',
                      label=f"{population_label} (n = {max(sample_counts) if sample_counts else 0})")

        for bar, mean_error, std_error in zip(bars, mean_errors, std_errors):
            ax.text(bar.get_x() + bar.get_width() / 2, mean_error + std_error,
                    f"{mean_error:.3f}\n± {std_error:.3f}",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
            upper_bounds.append(mean_error + std_error)

    ax.set_xticks(strategy_positions)
    ax.set_xticklabels([LOCALIZATION_STRATEGY_LABELS[key] for key in strategy_keys], fontsize=8)
    ax.set_ylabel("Localization error (seconds)")
    ax.set_ylim(0, (max(upper_bounds) if upper_bounds else 1.0) * 1.30)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
def generate_eer_report(window_results, audio_results, save_pdf=True, output_path=None, pdf_filename: str = "eer_report.pdf",
                         class_names=("Spliced (impostor)", "Authentic (genuine)"),
                         normalize=None, dpi: int = 400):
    """
    Produces a figure with 6 panels (DET curves, histograms, confusion
    matrices at window/audio level), plus an optional 7th panel comparing
    baseline vs entropy-weighted splice-localization accuracy when
    `eer_audio_level` was run with `compute_entropy_comparison=True`.
    """
    include_localization_panel = bool(audio_results.get('localization_error_stats'))
    num_rows = 4 if include_localization_panel else 3
    figure_height = 21 if include_localization_panel else 17

    fig = plt.figure(figsize=(16, figure_height))
    fig.suptitle(
        f"EER EVALUATION — Audio Splice Detector\n"
        f"{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(num_rows, 2, figure=fig, hspace=0.5, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _det_curve(ax1,
               window_results['scores'], window_results['labels'],
               window_results['optimal_threshold'],
               f"DET Curve — WINDOW Level\nEER = {window_results['eer']*100:.4f}%")

    ax2 = fig.add_subplot(gs[0, 1])
    _det_curve(ax2,
               audio_results['scores'], audio_results['labels'],
               audio_results['optimal_threshold'],
               f"DET Curve — AUDIO Level\nEER = {audio_results['eer']*100:.4f}%")

    ax3 = fig.add_subplot(gs[1, 0])
    _score_histogram(ax3,
                      window_results['scores'], window_results['labels'],
                      window_results['optimal_threshold'],
                      "Score Distribution — WINDOW Level")

    ax4 = fig.add_subplot(gs[1, 1])
    _score_histogram(ax4,
                      audio_results['scores'], audio_results['labels'],
                      audio_results['optimal_threshold'],
                      "Score Distribution — AUDIO Level")

    ax5 = fig.add_subplot(gs[2, 0])
    _plot_confusion_matrix(ax5, window_results['confusion_matrix'], class_names,
                            f"Confusion Matrix — WINDOW Level\n(threshold={window_results['optimal_threshold']:.3f})",
                            normalize=normalize)

    ax6 = fig.add_subplot(gs[2, 1])
    _plot_confusion_matrix(ax6, audio_results['confusion_matrix'], class_names,
                            f"Confusion Matrix — AUDIO Level\n(threshold={audio_results['optimal_threshold']:.3f})",
                            normalize=normalize)

    if include_localization_panel:
        ax7 = fig.add_subplot(gs[3, :])
        _localization_error_summary(
            ax7, audio_results,
            f"Splice Localization Error — AUDIO Level "
            f"(n = {audio_results['n_spliced']} spliced audios)"
        )

    summary = (
        f"  WINDOW Level => EER: {window_results['eer']*100:.4f}%   "
        f"Optimal threshold: {window_results['optimal_threshold']:.4f}   "
        f"(genuine: {window_results['n_genuine']} | impostors: {window_results['n_impostors']})  "
        f"\n  AUDIO Level   =>  EER: {audio_results['eer']*100:.4f}%   "
        f"Optimal threshold: {audio_results['optimal_threshold']:.4f}   "
        f"(authentic: {audio_results['n_authentic']} | spliced: {audio_results['n_spliced']})"
    )
    if include_localization_panel:
        summary += (
            "\n  Localization error (mean ± std)\n  "
            + _format_localization_error_summary(audio_results)
        )
    fig.text(0.5, 0.01, summary, ha='center', fontsize=10,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff', edgecolor='steelblue'))
    if save_pdf:

        folder = output_path if output_path is not None else "."
        report_path = os.path.join(folder, pdf_filename)
        fig.savefig(report_path, bbox_inches='tight', dpi=dpi)
        print(f"Report saved to {report_path}")

    plt.show()
    return fig

def generate_ia_splicing_report(window_results, audio_results, save_pdf=True, output_path=None,
                                 pdf_filename: str = "ia_splicing_report.pdf",
                                 class_names=("Spliced (impostor)", "Authentic (genuine)"),
                                 normalize=None, dpi: int = 400):
    """
    Visual report for the AI-splicing evaluation set, mirroring generate_eer_report():
    - Panel 1: DET curve (window level)
    - Panel 2: Score histogram (window level)
    - Panel 3: Confusion matrix (window level)
    - Panel 4: Detection / localization rate summary (audio level)
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"IA-SPLICING EVALUATION — Audio Splice Detector\n"
        f"{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        fontsize=14, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _det_curve(ax1, window_results['scores'], window_results['labels'],
               window_results['optimal_threshold'],
               f"DET Curve — WINDOW Level\nEER = {window_results['eer']*100:.4f}%")

    ax2 = fig.add_subplot(gs[0, 1])
    _score_histogram(ax2, window_results['scores'], window_results['labels'],
                      window_results['optimal_threshold'], "Score Distribution — WINDOW Level")

    ax3 = fig.add_subplot(gs[1, 0])
    _plot_confusion_matrix(ax3, window_results['confusion_matrix'], class_names,
                            f"Confusion Matrix — WINDOW Level\n(threshold={window_results['optimal_threshold']:.3f})",
                            normalize=normalize)

    #ax4 = fig.add_subplot(gs[1, 1])
    #_localization_comparison_summary(ax4, audio_results, "Detection Performance — AUDIO Level")
    ax4 = fig.add_subplot(gs[1, 1])
    _localization_error_summary(
        ax4, audio_results,
        f"Splice Localization Error — AUDIO Level (n = {audio_results['n_samples']} samples)"
    )

    summary = (
        f"  WINDOW Level => EER: {window_results['eer']*100:.4f}%   "
        f"FAR: {window_results['far_at_threshold']*100:.4f}%   FRR: {window_results['frr_at_threshold']*100:.4f}%   "
        f"(genuine: {window_results['n_genuine']} | impostors: {window_results['n_impostors']})  "
        f"\n  AUDIO Level   =>  Detection rate: {audio_results['detection_rate']*100:.4f}%   "
        f"(n = {audio_results['n_samples']})"
        f"\n  Localization error (mean ± std)\n  "
        + _format_localization_error_summary(audio_results)
    )
    fig.text(0.5, 0.01, summary, ha='center', fontsize=10,
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff', edgecolor='steelblue'))

    if save_pdf:
        folder = output_path if output_path is not None else "."
        report_path = os.path.join(folder, pdf_filename)
        fig.savefig(report_path, bbox_inches='tight', dpi=dpi)
        print(f"Report saved to {report_path}")

    plt.show()
    return fig
# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_eer_complete(model, test_audios: list[dict], device_to_label: dict,
                           #mean_global: float,
                           #std_global: float,
                           device=None, n_audios_test: int = None, save_pdf: bool = True,
                           sr: int = None, win_sec: float = None,
                           hop_sec: float = None, n_fft: int = None,
                           hop_length_fft: int = None, top_db: int = None,
                           output_path: str = None, pdf_filename: str = None, run_name: str = None,
                           fixed_threshold: dict = None, class_names=("Spliced (impostor)", "Authentic (genuine)"),
                           normalize=None,dpi: int = 400, median_filter_kernel_size: int = None,
                           compute_entropy_comparison: bool = False, entropy_window: int = None,
                           localization_tolerance_s: float = None, distance_metric: str = "euclidean", 
                           VI: np.ndarray = None) -> dict:
    """
    Computes EER at window level and at audio level, generates the visual
    report, and returns a dict with all the results.

    Parameters
    ----------
    save_pdf        : if True saves "eer_report.pdf"
    fixed_threshold : dict {'window': float, 'audio': float} with an
                      already-calibrated threshold. If None, the threshold
                      is optimized on this same set (only use that mode for
                      calibration — never on the final test set, to avoid
                      leakage).
    """
    if pdf_filename is None:
        pdf_filename = f"eer_report_{run_name}.pdf" if run_name else "eer_report.pdf"
    fixed_threshold_window = fixed_threshold['window'] if fixed_threshold is not None else None
    fixed_threshold_audio  = fixed_threshold['audio']  if fixed_threshold is not None else None

    print(" Computing WINDOW-level EER ")
    window_results = eer_window_level(
        model, test_audios, device_to_label,
        #mean_global, std_global,
        device=device,
        sr=sr, win_sec=win_sec, hop_sec=hop_sec,
        n_fft=n_fft, hop_length_fft=hop_length_fft, top_db=top_db,
        fixed_threshold=fixed_threshold_window, #median_filter_kernel_size=median_filter_kernel_size,
        distance_metric=distance_metric, VI=VI
    )
    if fixed_threshold_window is None:
        print(f"   Window EER : {window_results['eer']*100:.4f}%  "
              f"| optimal threshold (calibrated here) : {window_results['optimal_threshold']:.4f}")
    else:
        print(f"   FIXED threshold used: {window_results['optimal_threshold']:.4f}  "
              f"| FAR: {window_results['far_at_threshold']*100:.4f}%  "
              f"| FRR: {window_results['frr_at_threshold']*100:.4f}%  "
              f"(reference EER for this sample: {window_results['eer']*100:.4f}%)")

    print(" Computing AUDIO-level EER ")
    audio_results = eer_audio_level(
        model, test_audios, device_to_label,
        #mean_global, std_global, 
        device=device,
        n_audios_test=n_audios_test, sr=sr,
        win_sec=win_sec, hop_sec=hop_sec,
        n_fft=n_fft, hop_length_fft=hop_length_fft, top_db=top_db,
        fixed_threshold=fixed_threshold_audio, compute_entropy_comparison=compute_entropy_comparison,
        entropy_window=entropy_window,
        localization_tolerance_s=localization_tolerance_s, median_filter_kernel_size=median_filter_kernel_size,
        distance_metric=distance_metric, VI=VI
    )
    if audio_results.get('localization_error_stats'):
        print("   Localization error (mean ± std):")
        print("   " + _format_localization_error_summary(audio_results))
    #if audio_results['localization_rate_entropy'] is not None:
    #    print(f"   Localization rate (entropy-weighted):          {audio_results['localization_rate_entropy']*100:.4f}%  "
    #          f"(±{localization_tolerance_s}s)")
    if fixed_threshold_audio is None:
        print(f"   Audio EER   : {audio_results['eer']*100:.4f}%  "
              f"| optimal threshold (calibrated here) : {audio_results['optimal_threshold']:.4f}")
    else:
        print(f"   FIXED threshold used: {audio_results['optimal_threshold']:.4f}  "
              f"| FAR: {audio_results['far_at_threshold']*100:.4f}%  "
              f"| FRR: {audio_results['frr_at_threshold']*100:.4f}%  "
              f"(reference EER for this sample: {audio_results['eer']*100:.4f}%)")

    recommended_threshold = window_results['optimal_threshold']
    print(f"Threshold used for generate_forensic_report(): {recommended_threshold:.4f}")

    generate_eer_report(window_results, audio_results, save_pdf, output_path=output_path, pdf_filename=pdf_filename,
                         class_names=class_names, normalize=normalize, dpi=dpi)

    return {
        'window':                window_results,
        'audio':                 audio_results,
        'recommended_threshold': recommended_threshold,
        'distance_metric':       distance_metric,
    }

# ══════════════════════════════════════════════════════════════════════════
# 7. Cross-verification by condition (clean vs noise)
# ══════════════════════════════════════════════════════════════════════════

def _average_audio_embedding(audio, model, #global_mean, global_std, 
                              device=None,
                              sr=None, win_sec=None, hop_sec=None,
                              n_fft=None, hop_length_fft=None):
    """Representative embedding of a full audio: average of its windows."""
    embs, _ = _audio_embeddings(audio, model, #global_mean=global_mean, global_std=global_std,
                                 device=device,
                                 sr=sr, win_sec=win_sec, hop_sec=hop_sec,
                                 n_fft=n_fft, hop_length_fft=hop_length_fft)
    if len(embs) == 0:
        return None
    return embs.mean(axis=0)


def eer_cross_verification(model, test_audios, #global_mean, global_std,
                            device=None,
                            n_pairs=None, sr=None,
                            win_sec=None, hop_sec=None,
                            n_fft=None, hop_length_fft=None,
                            condition_filter=None):
    """
    Compares pairs of FULL AUDIOS (not windows within the same audio).
    condition_filter = (cond_a, cond_b), e.g. ('speech_clean', 'speech_noise')
    to force which condition each member of the pair comes from.
    """
    devices = sorted(set(a['device'] for a in test_audios))
    scores, labels = [], []

    for _ in range(n_pairs):
        genuine = random.random() < 0.5

        if genuine:
            dev = random.choice(devices)
            pool = [a for a in test_audios if a['device'] == dev]
        else:
            dev_a, dev_b = random.sample(devices, 2)
            pool_a = [a for a in test_audios if a['device'] == dev_a]
            pool_b = [a for a in test_audios if a['device'] == dev_b]

        if condition_filter is not None:
            cond_a, cond_b = condition_filter
            if genuine:
                pool_a = [a for a in pool if a['condition'] == cond_a]
                pool_b = [a for a in pool if a['condition'] == cond_b]
            else:
                pool_a = [a for a in pool_a if a['condition'] == cond_a]
                pool_b = [a for a in pool_b if a['condition'] == cond_b]
        elif genuine:
            pool_a = pool_b = pool

        if not pool_a or not pool_b:
            continue

        if genuine:
            if len(set(map(id, pool_a)) | set(map(id, pool_b))) < 2:
                continue  # not enough distinct audios to form a genuine pair
            while True:
                a_info = random.choice(pool_a)
                b_info = random.choice(pool_b)
                if a_info is not b_info:
                    break
        else:
            a_info = random.choice(pool_a)
            b_info = random.choice(pool_b)

        emb_a = _average_audio_embedding(a_info['audio'], model, device=device, #global_mean=global_mean, global_std=global_std,
                                          sr=sr, win_sec=win_sec, hop_sec=hop_sec,
                                          n_fft=n_fft, hop_length_fft=hop_length_fft)
        emb_b = _average_audio_embedding(b_info['audio'], model, device=device, #global_mean=global_mean, global_std=global_std,
                                          sr=sr, win_sec=win_sec, hop_sec=hop_sec,
                                          n_fft=n_fft, hop_length_fft=hop_length_fft)
        if emb_a is None or emb_b is None:
            continue

        scores.append(np.linalg.norm(emb_a - emb_b))
        labels.append(1 if genuine else 0)

    scores = np.array(scores)
    labels = np.array(labels)
    eer, optimal_threshold = calculate_eer(scores, labels)

    return {
        'eer': eer,
        'optimal_threshold': optimal_threshold,
        'scores': scores,
        'labels': labels,
        'n_genuine': int((labels == 1).sum()),
        'n_impostors': int((labels == 0).sum()),
    }


def evaluate_noise_robustness(model, test_audios, #mean_global, std_global, 
                               device=None,
                               n_pairs=None, sr=None,
                               win_sec=None, hop_sec=None,
                               n_fft=None, hop_length_fft=None):
    """
    Runs clean-clean, noise-noise and clean-noise and compares the EERs.
    """
    combos = {
        'clean_vs_clean': ('speech_clean', 'speech_clean'),
        'noise_vs_noise': ('speech_noise', 'speech_noise'),
        'clean_vs_noise': ('speech_clean', 'speech_noise'),
    }
    results = {}
    for name, combo in combos.items():
        print(f"Evaluating {name} ...")
        res = eer_cross_verification(model, test_audios, #mean_global, std_global,
                                      device=device, n_pairs=n_pairs, sr=sr,
                                      win_sec=win_sec, hop_sec=hop_sec,
                                      n_fft=n_fft, hop_length_fft=hop_length_fft,
                                      condition_filter=combo)
        results[name] = res
        print(f"  EER = {res['eer']*100:.4f}%  (thr={res['optimal_threshold']:.4f}, "
              f"genuine={res['n_genuine']}, impostors={res['n_impostors']})")
    return results
    # ══════════════════════════════════════════════════════════════════════════
# 8. Saving results to compare runs
# ══════════════════════════════════════════════════════════════════════════




def save_eer_results(results: dict, output_path: str,
                      run_name: str = None,
                      config_extra: dict = None,
                      csv_name: str = "eer_history.csv",
                      save_detail_pickle: bool = True) -> str:
    """
    Saves a summary of an evaluate_eer_complete() run to a cumulative CSV
    (one row per run), to compare results over time. Optionally also saves
    the full detail (scores, labels, curves) in a separate .pkl per run.
    """
    os.makedirs(output_path, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if run_name is None:
        run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')

    window_results = results['window']
    audio_results = results['audio']

    row = {
        "timestamp": timestamp,
        "run_name": run_name,
        "eer_window_pct": round(window_results['eer'] * 100, 3),
        "threshold_window": round(window_results['optimal_threshold'], 5),
        "n_genuine_window": window_results['n_genuine'],
        "n_impostors_window": window_results['n_impostors'],
        "eer_audio_pct": round(audio_results['eer'] * 100, 3),
        "threshold_audio": round(audio_results['optimal_threshold'], 5),
        "n_authentic_audio": audio_results['n_authentic'],
        "n_spliced_audio": audio_results['n_spliced'],
        "recommended_threshold": round(results['recommended_threshold'], 5),
    }
    tn_v, fp_v, fn_v, tp_v = window_results['confusion_matrix'].ravel()
    tn_a, fp_a, fn_a, tp_a = audio_results['confusion_matrix'].ravel()
    row.update({
        "cm_window_tn": int(tn_v), "cm_window_fp": int(fp_v),
        "cm_window_fn": int(fn_v), "cm_window_tp": int(tp_v),
        "cm_audio_tn": int(tn_a), "cm_audio_fp": int(fp_a),
        "cm_audio_fn": int(fn_a), "cm_audio_tp": int(tp_a),
    })
    for results_key, population_label, _, _ in LOCALIZATION_ERROR_POPULATIONS:
        population_suffix = "all" if results_key.endswith("stats") else "detected"
        for strategy_key, strategy_stats in (audio_results.get(results_key) or {}).items():
            row[f"loc_error_mean_s_{strategy_key}_{population_suffix}"] = round(strategy_stats['mean_error_s'], 5)
            row[f"loc_error_std_s_{strategy_key}_{population_suffix}"] = round(strategy_stats['std_error_s'], 5)
            row[f"loc_error_n_{strategy_key}_{population_suffix}"] = strategy_stats['n_samples']
    if config_extra:
        for k, v in config_extra.items():
            row[k] = v

    csv_path = os.path.join(output_path, csv_name)
    _append_csv_row(csv_path, row)
    print(f"Result appended to {csv_path}")

    if save_detail_pickle:
        detail_path = os.path.join(output_path, f"eer_detail_{run_name}.pkl")
        with open(detail_path, "wb") as f:
            _pickle.dump(results, f, protocol=_pickle.HIGHEST_PROTOCOL)
        print(f"Full detail saved to {detail_path}")

    return csv_path
# ══════════════════════════════════════════════════════════════════════════
# 9. Saving a COMPLETE EXPERIMENT (to compare pipeline versions)
# ══════════════════════════════════════════════════════════════════════════

def save_complete_experiment(output_path: str,
                              run_name: str,
                              calibrated_threshold: dict,
                              calibration_results: dict,
                              clean_results: dict = None,
                              noise_results: dict = None,
                              robustness_results: dict = None,
                              final_test_results: dict = None,
                              tsne_data: dict = None,
                              config_extra: dict = None,
                              csv_name: str = "experiment_history.csv") -> str:
    """
    Saves EVERYTHING relevant from a complete pipeline run, so it can be
    compared against future versions of the code/model.

    Parameters
    ----------
    calibrated_threshold : dict {'window': float, 'audio': float}
    calibration_results   : dict returned by evaluate_eer_complete() on val_audios
    clean_results          : dict returned by evaluate_eer_complete() on test_clean (fixed threshold)
    noise_results          : same, on test_noise
    robustness_results     : dict returned by evaluate_noise_robustness()
    final_test_results     : dict returned by evaluate_eer_complete() on test_audios (fixed threshold)
    tsne_data              : optional dict with t-SNE coords/labels
    config_extra           : hyperparameters to log (SR, EPOCHS, N_FFT, etc.)
    """
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if run_name is None:
        run_name = datetime.now().strftime('experiment_%Y%m%d_%H%M%S')

    row = {
        "timestamp": timestamp,
        "run_name": run_name,
        "frozen_threshold_window": round(calibrated_threshold['window'], 5),
        "frozen_threshold_audio": round(calibrated_threshold['audio'], 5),
    }

    def _add_metrics(prefix, res, level):
        if res is None:
            return
        r = res[level]
        row[f"{prefix}_eer_{level}_pct"] = round(r['eer'] * 100, 3)
        if 'far_at_threshold' in r and r['far_at_threshold'] is not None:
            row[f"{prefix}_far_{level}_pct"] = round(r['far_at_threshold'] * 100, 3)
            row[f"{prefix}_frr_{level}_pct"] = round(r['frr_at_threshold'] * 100, 3)
        n_pos_key = 'n_genuine' if level == 'window' else 'n_authentic'
        n_neg_key = 'n_impostors' if level == 'window' else 'n_spliced'
        row[f"{prefix}_n_pos_{level}"] = r[n_pos_key]
        row[f"{prefix}_n_neg_{level}"] = r[n_neg_key]
        if 'confusion_matrix' in r and r['confusion_matrix'] is not None:
            tn, fp, fn, tp = r['confusion_matrix'].ravel()
            row[f"{prefix}_cm_{level}_tn"] = int(tn)
            row[f"{prefix}_cm_{level}_fp"] = int(fp)
            row[f"{prefix}_cm_{level}_fn"] = int(fn)
            row[f"{prefix}_cm_{level}_tp"] = int(tp)

    _add_metrics("calib", calibration_results, "window")
    _add_metrics("calib", calibration_results, "audio")
    _add_metrics("clean", clean_results, "window")
    _add_metrics("clean", clean_results, "audio")
    _add_metrics("noise", noise_results, "window")
    _add_metrics("noise", noise_results, "audio")
    _add_metrics("test", final_test_results, "window")
    _add_metrics("test", final_test_results, "audio")

    if robustness_results is not None:
        for combo_name, combo_result in robustness_results.items():
            row[f"robustness_{combo_name}_eer_pct"] = round(combo_result['eer'] * 100, 3)
            row[f"robustness_{combo_name}_thr"] = round(combo_result['optimal_threshold'], 5)

    if config_extra:
        for k, v in config_extra.items():
            row[k] = v

    csv_path = os.path.join(output_path, csv_name)
    _append_csv_row(csv_path, row)
    print(f"Experiment summary appended to {csv_path}")

    detail = {
        "timestamp": timestamp,
        "run_name": run_name,
        "calibrated_threshold": calibrated_threshold,
        "calibration_results": calibration_results,
        "clean_results": clean_results,
        "noise_results": noise_results,
        "robustness_results": robustness_results,
        "final_test_results": final_test_results,
        "tsne_data": tsne_data,
        "config_extra": config_extra,
    }
    detail_path = os.path.join(output_path, f"experiment_{run_name}.pkl")
    with open(detail_path, "wb") as f:
        _pickle.dump(detail, f, protocol=_pickle.HIGHEST_PROTOCOL)
    print(f"Full experiment detail saved to {detail_path}")

    return csv_path

def _append_csv_row(csv_path: str, row: dict) -> None:
    """
    Appends a row to a cumulative CSV, merging columns if the CSV
    already exists with a different column set (e.g. you added a new
    hyperparameter in config_extra). Does not lose previous history.
    """
    previous_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            previous_rows = list(csv.DictReader(f))

    columns = list(row.keys())
    for pr in previous_rows:
        for k in pr.keys():
            if k not in columns:
                columns.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for pr in previous_rows:
            writer.writerow(pr)
        writer.writerow(row)
# ══════════════════════════════════════════════════════════════════════════════
# 10. AI-Splicing evaluation (pre-spliced audios with known ground-truth cut)
# ══════════════════════════════════════════════════════════════════════════════
#
# Unlike eer_window_level() / eer_audio_level(), these audios are NOT spliced
# on the fly — the splice already exists in the file, and the ground-truth
# location comes from metadata.json (splice.splice_info.insert_start_s /
# insert_end_s). Only detection at a FIXED, already-calibrated threshold is
# meaningful here; the threshold must NOT be re-optimized on this set.

def eer_window_level_ia_splicing(model, ia_dataset: list[dict],
                                  #global_mean: float, global_std: float, 
                                  device=None,
                                  sr: int = None, win_sec: float = None, hop_sec: float = None,
                                  n_fft: int = None, hop_length_fft: int = None,
                                  margin_s: float = None, fixed_threshold: float = None, 
                                  median_filter_kernel_size: int = None, 
                                  distance_metric: str = "euclidean", VI: np.ndarray = None) -> dict:
    """
    Window-level detection over pre-spliced AI-generated audios.

    Pair label (window i-1, i):
        0 = impostor : the pair straddles [insert_start_s, insert_end_s]
        1 = genuine  : both windows lie fully within a real segment, at
                       least `margin_s` seconds away from the splice zone
    Windows too close to the boundary to classify confidently are skipped.
    median_filter_kernel_size : see eer_window_level(). Applied per audio to
        the raw distance sequence before it is split into genuine/impostor
        labels below.
    """
    all_scores, all_labels = [], []

    for item in ia_dataset:
        embs, timestamps = _audio_embeddings(
            item['audio'], model, device=device, #global_mean=global_mean, global_std=global_std,
            sr=sr, win_sec=win_sec, hop_sec=hop_sec, n_fft=n_fft, hop_length_fft=hop_length_fft
        )
        dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
        #dists = _apply_median_filter(dists, median_filter_kernel_size) if median_filter_kernel_size else dists
        if len(dists) == 0:
            continue

        insert_start = item['true_insert_start_s']
        insert_end = item['true_insert_end_s']

        for i, dist in enumerate(dists):
            t_prev, t_curr = timestamps[i], timestamps[i + 1]
            window_end_prev = t_prev + win_sec
            window_end_curr = t_curr + win_sec

            # A pair is impostor only if the source change (real <-> TTS) falls between
            # the two windows, i.e. the pair straddles at least one insert boundary.
            crosses_insert_start = t_prev < insert_start < window_end_curr
            crosses_insert_end = t_prev < insert_end < window_end_curr
            straddles_insert_boundary = crosses_insert_start or crosses_insert_end
            fully_outside_margin = (window_end_curr < insert_start - margin_s) or (t_prev > insert_end + margin_s)

            if straddles_insert_boundary:
                all_scores.append(dist)
                all_labels.append(0)   # IMPOSTOR_LABEL
            elif fully_outside_margin:
                all_scores.append(dist)
                all_labels.append(1)   # GENUINE_LABEL
            # else: ambiguous (pure-TTS pair inside the insert, or pair inside the margin buffer) — discarded

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if fixed_threshold is None:
        eer, optimal_threshold = calculate_eer(scores, labels)
    else:
        optimal_threshold = fixed_threshold
        eer, _ = calculate_eer(scores, labels)  # informative only

    far, frr = calculate_far_frr_at_threshold(scores, labels, optimal_threshold)
    conf_matrix = calculate_confusion_matrix(scores, labels, optimal_threshold)

    return {
        'eer': eer, 'optimal_threshold': optimal_threshold,
        'far_at_threshold': far, 'frr_at_threshold': frr,
        'scores': scores, 'labels': labels,
        'n_genuine': int((labels == 1).sum()), 'n_impostors': int((labels == 0).sum()),
        'confusion_matrix': conf_matrix,
    }

def eer_audio_level_ia_splicing(model, ia_dataset: list[dict], genuine_audios: list[dict],
                                 device=None, n_genuine_audios: int = None,
                                 sr: int = None, win_sec: float = None, hop_sec: float = None,
                                 n_fft: int = None, hop_length_fft: int = None,
                                 fixed_threshold: float = None,
                                 #median_filter_kernel_size: int = None,
                                 distance_metric: str = "euclidean", VI: np.ndarray = None) -> dict:
    """
    Audio-level EER for the AI-splicing evaluation set, mirroring
    eer_audio_level() but with genuine and impostor populations drawn from
    two different sources instead of on-the-fly synthetic splicing:

        - IMPOSTOR (label = 0, spliced)   : every audio in `ia_dataset`,
          already containing a TTS-based insert with known ground truth.
        - GENUINE  (label = 1, authentic) : audios sampled from
          `genuine_audios` (e.g. the pure single-device test set), scored
          identically (max consecutive-window embedding distance), so both
          populations remain directly comparable.

    `fixed_threshold` must be the already-frozen audio-level threshold —
    this set is never used to (re)calibrate it.
    """
    all_scores, all_labels = [], []

    # IMPOSTOR (spliced) audios — every sample in ia_dataset
    for item in ia_dataset:
        embs, _ = _audio_embeddings(
            item['audio'], model, device=device,
            sr=sr, win_sec=win_sec, hop_sec=hop_sec,
            n_fft=n_fft, hop_length_fft=hop_length_fft
        )
        dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
        if len(dists) == 0:
            continue
        all_scores.append(float(dists.max()))
        all_labels.append(0)   # spliced

    # GENUINE (authentic) audios
    n_genuine_audios = n_genuine_audios or len(ia_dataset)
    for audio_info in random.sample(genuine_audios, min(n_genuine_audios, len(genuine_audios))):
        embs, _ = _audio_embeddings(
            audio_info['audio'], model, device=device,
            sr=sr, win_sec=win_sec, hop_sec=hop_sec,
            n_fft=n_fft, hop_length_fft=hop_length_fft
        )
        dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
        if len(dists) == 0:
            continue
        all_scores.append(float(dists.max()))
        all_labels.append(1)   # authentic

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    if fixed_threshold is None:
        eer, optimal_threshold = calculate_eer(scores, labels)
    else:
        optimal_threshold = fixed_threshold
        eer, _ = calculate_eer(scores, labels)  # informative only, not used to set the threshold

    far, frr = calculate_far_frr_at_threshold(scores, labels, optimal_threshold)
    conf_matrix = calculate_confusion_matrix(scores, labels, optimal_threshold)

    return {
        'eer':               eer,
        'optimal_threshold': optimal_threshold,
        'far_at_threshold':  far,
        'frr_at_threshold':  frr,
        'scores':            scores,
        'labels':            labels,
        'n_authentic':       int((labels == 1).sum()),
        'n_spliced':         int((labels == 0).sum()),
        'confusion_matrix':  conf_matrix,
    }

def evaluate_ia_splicing_detection(model, ia_dataset: list[dict],
                                    #global_mean: float, global_std: float, 
                                    device=None,
                                    sr: int = None, win_sec: float = None, hop_sec: float = None,
                                    n_fft: int = None, hop_length_fft: int = None,
                                    fixed_threshold: float = None, tolerance_s: float = None,
                                    compute_entropy_comparison: bool = False,
                                    entropy_window: int = None, median_filter_kernel_size: int = None,
                                    distance_metric: str = "euclidean", VI: np.ndarray = None) -> dict:
    """
    
    Every sample in `ia_dataset` is, by construction, a spliced audio — there
    is no genuine/impostor balance at this level (unlike eer_audio_level()).
    Instead this reports:
        detection_rate    : fraction of audios where max distance > fixed_threshold
        localization_rate : fraction of detections whose flagged time falls
                             within [insert_start_s - tol, insert_end_s + tol]
    Localization strategies reported (all gated by the same `detected` flag):
            localization_rate                 : raw-distance argmax (baseline)
            localization_rate_median          : median-filtered-distance argmax
            localization_rate_entropy         : entropy-weighted score on raw distances
            localization_rate_median_entropy  : entropy-weighted score on median-filtered distances
    
    """
    per_sample = []

    for item in ia_dataset:
        embs, timestamps = _audio_embeddings(
            item['audio'], model, device=device,
            sr=sr, win_sec=win_sec, hop_sec=hop_sec, n_fft=n_fft, hop_length_fft=hop_length_fft
        )
        raw_dists = _consecutive_distances(embs, distance_metric=distance_metric, VI=VI)
        if len(raw_dists) == 0:
            continue

        dist_timestamps = timestamps[1:]
        max_idx = int(np.argmax(raw_dists))
        max_distance = float(raw_dists[max_idx])
        detected_time = dist_timestamps[max_idx]
        detected = max_distance > fixed_threshold

        insert_start, insert_end = item['true_insert_start_s'], item['true_insert_end_s']
        correct_localization = detected and (insert_start - tolerance_s) <= detected_time <= (insert_end + tolerance_s)

        # Median-filtered strategy 
        filtered_dists = None
        median_fields = {}
        if median_filter_kernel_size:
            filtered_dists = _apply_median_filter(raw_dists, median_filter_kernel_size)
            median_max_idx = int(np.argmax(filtered_dists))
            median_detected_time = dist_timestamps[median_max_idx]
            median_correct_localization = (
                detected
                and (insert_start - tolerance_s) <= median_detected_time <= (insert_end + tolerance_s)
            )
            median_fields = {
                'median_max_distance': float(filtered_dists[median_max_idx]),
                'median_detected_time_s': median_detected_time,
                'median_correct_localization': median_correct_localization,
            }

        # ── Entropy-weighted strategies (raw and median-filtered) ───────
        entropy_fields = {}
        median_entropy_fields = {}
        if compute_entropy_comparison:
            if entropy_window is None:
                raise ValueError(
                    "entropy_window must be provided when compute_entropy_comparison=True"
                )
            entropy_result = entropy_weighted_localization(
                raw_dists, dist_timestamps, entropy_window=entropy_window
            )
            entropy_correct_localization = (
                detected
                and (insert_start - tolerance_s) <= entropy_result['entropy_detected_time_s'] <= (insert_end + tolerance_s)
            )
            entropy_fields = {
                **entropy_result,
                'entropy_correct_localization': entropy_correct_localization,
            }

            if filtered_dists is not None:
                median_entropy_result = entropy_weighted_localization(
                    filtered_dists, dist_timestamps, entropy_window=entropy_window
                )
                median_entropy_correct_localization = (
                    detected
                    and (insert_start - tolerance_s) <= median_entropy_result['entropy_detected_time_s'] <= (insert_end + tolerance_s)
                )
                median_entropy_fields = {
                    'median_entropy_detected_time_s': median_entropy_result['entropy_detected_time_s'],
                    'median_entropy_max_distance': median_entropy_result['entropy_max_distance'],
                    'median_local_entropy_at_peak': median_entropy_result['local_entropy_at_peak'],
                    'median_entropy_correct_localization': median_entropy_correct_localization,
                }

        per_sample.append({
            'filename': item['filename'], 'device': item['device'], 'condition': item['condition'],
            'speaker': item.get('speaker'),
            'true_insert_start_s': insert_start, 'true_insert_end_s': insert_end,
            'max_distance': max_distance, 'detected_time_s': detected_time,
            'detected': detected, 'correct_localization': correct_localization,
            **median_fields, **entropy_fields, **median_entropy_fields,
        })

    n = len(per_sample)
    detection_rate = float(np.mean([s['detected'] for s in per_sample])) if n else float('nan')
    localization_rate = float(np.mean([s['correct_localization'] for s in per_sample])) if n else float('nan')

    localization_rate_median = None
    if median_filter_kernel_size and n:
        localization_rate_median = float(np.mean([s['median_correct_localization'] for s in per_sample]))

    localization_rate_entropy = None
    if compute_entropy_comparison and n:
        localization_rate_entropy = float(np.mean([s['entropy_correct_localization'] for s in per_sample]))

    localization_rate_median_entropy = None
    if compute_entropy_comparison and median_filter_kernel_size and n:
        localization_rate_median_entropy = float(
            np.mean([s['median_entropy_correct_localization'] for s in per_sample])
        )

    # ── Localization error (seconds) per strategy ──────────────────────────
    # Ground truth is an interval, so the error is zero inside the inserted
    # segment and the distance to the nearest boundary outside it. Reported
    # over all samples and, separately, over the detected subset only.
    strategy_time_fields = {
        'baseline':       'detected_time_s',
        'median':         'median_detected_time_s',
        'entropy':        'entropy_detected_time_s',
        'median_entropy': 'median_entropy_detected_time_s',
    }
    errors_by_strategy = {key: [] for key in LOCALIZATION_STRATEGY_KEYS}
    errors_by_strategy_detected = {key: [] for key in LOCALIZATION_STRATEGY_KEYS}
    for sample in per_sample:
        for strategy_key, time_field in strategy_time_fields.items():
            if time_field not in sample:
                continue
            error_s = _interval_localization_error_s(
                sample[time_field], sample['true_insert_start_s'], sample['true_insert_end_s']
            )
            sample[f"{strategy_key}_localization_error_s"] = error_s
            errors_by_strategy[strategy_key].append(error_s)
            if sample['detected']:
                errors_by_strategy_detected[strategy_key].append(error_s)

    localization_error_stats = _localization_error_statistics(errors_by_strategy)
    localization_error_stats_detected = _localization_error_statistics(errors_by_strategy_detected)

    return {
        'detection_rate': detection_rate,
        'localization_rate': localization_rate,
        'localization_rate_median': localization_rate_median,
        'localization_rate_entropy': localization_rate_entropy,
        'localization_rate_median_entropy': localization_rate_median_entropy,
        'localization_error_stats':          localization_error_stats,
        'localization_error_stats_detected': localization_error_stats_detected,
        'threshold_used': fixed_threshold,
        'tolerance_s': tolerance_s,
        'entropy_window': entropy_window if compute_entropy_comparison else None,
        'median_filter_kernel_size': median_filter_kernel_size,
        'n_samples': n,
        'per_sample': per_sample,
    }


def evaluate_ia_splicing_complete(model, ia_dataset: list[dict],
                                  #global_mean: float, global_std: float, 
                                  genuine_audios: list[dict] = None,          
                                   n_genuine_audios_eer: int = None,           
                                   device=None,
                                   sr: int = None, win_sec: float = None, hop_sec: float = None,
                                   n_fft: int = None, hop_length_fft: int = None,
                                   fixed_threshold: dict = None, tolerance_s: float = None,
                                   margin_s: float = None,
                                   save_pdf: bool = True, output_path: str = None,
                                   pdf_filename: str = None, run_name: str = None,
                                   class_names=("Spliced (impostor)", "Authentic (genuine)"),
                                   normalize=None, dpi: int = 400, median_filter_kernel_size: int = None,
                                   compute_entropy_comparison: bool = False,
                                   entropy_window: int = None, distance_metric: str = "euclidean", 
                                   VI: np.ndarray = None) -> dict:
    """
    Main entry point for the IA-splicing evaluation, mirroring
    evaluate_eer_complete()'s two-level structure. `fixed_threshold` must be
    the already-frozen dict {'window': float, 'audio': float}  this set is
    never used to (re)calibrate the threshold.
    """
    print(" Computing WINDOW-level detection (AI-splicing set) ")
    window_results = eer_window_level_ia_splicing(
        model, ia_dataset, #global_mean, global_std, 
        device=device,
        sr=sr, win_sec=win_sec, hop_sec=hop_sec, n_fft=n_fft, hop_length_fft=hop_length_fft,
        margin_s=margin_s, fixed_threshold=fixed_threshold['window'], #median_filter_kernel_size=median_filter_kernel_size,
        distance_metric=distance_metric, VI=VI,
    )
    print(f"   FIXED threshold used: {window_results['optimal_threshold']:.4f}  "
          f"| FAR: {window_results['far_at_threshold']*100:.4f}%  "
          f"| FRR: {window_results['frr_at_threshold']*100:.4f}%  "
          f"(reference EER for this sample: {window_results['eer']*100:.4f}%)")

    print(" Computing AUDIO-level detection (IA-splicing set) ")
    audio_results = evaluate_ia_splicing_detection(
        model, ia_dataset,
        device=device,
        sr=sr, win_sec=win_sec, hop_sec=hop_sec, n_fft=n_fft, hop_length_fft=hop_length_fft,
        fixed_threshold=fixed_threshold['audio'], tolerance_s=tolerance_s,
        compute_entropy_comparison=compute_entropy_comparison,
        entropy_window=entropy_window,
        median_filter_kernel_size=median_filter_kernel_size,   
        distance_metric=distance_metric, VI=VI,
    )
    print(f"   Detection rate:    {audio_results['detection_rate']*100:.4f}%")
    print(f"   Localization rate (baseline, argmax distance): {audio_results['localization_rate']*100:.4f}%  (±{tolerance_s}s)")
    if audio_results['localization_rate_median'] is not None:
        print(f"   Localization rate (median-filtered):            {audio_results['localization_rate_median']*100:.4f}%  (±{tolerance_s}s)")
    if audio_results['localization_rate_entropy'] is not None:
        print(f"   Localization rate (entropy-weighted):           {audio_results['localization_rate_entropy']*100:.4f}%  (±{tolerance_s}s)")
    if audio_results['localization_rate_median_entropy'] is not None:
        print(f"   Localization rate (median-filtered + entropy):  {audio_results['localization_rate_median_entropy']*100:.4f}%  (±{tolerance_s}s)")

    audio_eer_results = None
    if genuine_audios is not None:
        print(" Computing AUDIO-level EER (IA-splicing set) ")
        audio_eer_results = eer_audio_level_ia_splicing(
            model, ia_dataset, genuine_audios,
            device=device, n_genuine_audios=n_genuine_audios_eer,
            sr=sr, win_sec=win_sec, hop_sec=hop_sec, n_fft=n_fft, hop_length_fft=hop_length_fft,
            fixed_threshold=fixed_threshold['audio'], #median_filter_kernel_size=median_filter_kernel_size,
            distance_metric=distance_metric, VI=VI,
        )
        print(f"   FIXED threshold used: {audio_eer_results['optimal_threshold']:.4f}  "
              f"| FAR: {audio_eer_results['far_at_threshold']*100:.4f}%  "
              f"| FRR: {audio_eer_results['frr_at_threshold']*100:.4f}%  "
              f"(reference EER for this sample: {audio_eer_results['eer']*100:.4f}%)")
        
    if pdf_filename is None:
        pdf_filename = f"ia_splicing_report_{run_name}.pdf" if run_name else "ia_splicing_report.pdf"
    
    generate_ia_splicing_report(window_results, audio_results, save_pdf, output_path=output_path,
                                 pdf_filename=pdf_filename, class_names=class_names,
                                 normalize=normalize, dpi=dpi)

    return {'window': window_results, 'audio': audio_results, 'audio_eer': audio_eer_results}

def save_ia_splicing_results(results: dict, output_path: str, run_name: str = None,
                              config_extra: dict = None,
                              csv_name: str = "ia_splicing_history.csv") -> str:
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if run_name is None:
        run_name = datetime.now().strftime('run_%Y%m%d_%H%M%S')

    window_results = results['window']
    audio_results = results['audio']

    row = {
        "timestamp": timestamp, "run_name": run_name,
        "eer_window_pct": round(window_results['eer'] * 100, 3),
        "far_window_pct": round(window_results['far_at_threshold'] * 100, 3),
        "frr_window_pct": round(window_results['frr_at_threshold'] * 100, 3),
        "n_genuine_window": window_results['n_genuine'],
        "n_impostors_window": window_results['n_impostors'],
        "detection_rate_pct": round(audio_results['detection_rate'] * 100, 3),
        "localization_rate_pct": round(audio_results['localization_rate'] * 100, 3),
        "n_samples_audio": audio_results['n_samples'],
    }
    for results_key, population_label, _, _ in LOCALIZATION_ERROR_POPULATIONS:
        population_suffix = "all" if results_key.endswith("stats") else "detected"
        for strategy_key, strategy_stats in (audio_results.get(results_key) or {}).items():
            row[f"loc_error_mean_s_{strategy_key}_{population_suffix}"] = round(strategy_stats['mean_error_s'], 5)
            row[f"loc_error_std_s_{strategy_key}_{population_suffix}"] = round(strategy_stats['std_error_s'], 5)
            row[f"loc_error_n_{strategy_key}_{population_suffix}"] = strategy_stats['n_samples']
    audio_eer_results = results.get('audio_eer')
    if audio_eer_results is not None:
        row["eer_audio_pct"] = round(audio_eer_results['eer'] * 100, 3)
        row["threshold_audio"] = round(audio_eer_results['optimal_threshold'], 5)
        row["far_audio_pct"] = round(audio_eer_results['far_at_threshold'] * 100, 3)
        row["frr_audio_pct"] = round(audio_eer_results['frr_at_threshold'] * 100, 3)
        row["n_authentic_audio"] = audio_eer_results['n_authentic']
        row["n_spliced_audio"] = audio_eer_results['n_spliced']

    if audio_results.get('localization_rate_median') is not None:
        row["localization_rate_median_pct"] = round(audio_results['localization_rate_median'] * 100, 3)
    if audio_results.get('localization_rate_entropy') is not None:
        row["localization_rate_entropy_pct"] = round(audio_results['localization_rate_entropy'] * 100, 3)
    if audio_results.get('localization_rate_median_entropy') is not None:
        row["localization_rate_median_entropy_pct"] = round(audio_results['localization_rate_median_entropy'] * 100, 3)
    if config_extra:
        row.update(config_extra)

    csv_path = os.path.join(output_path, csv_name)
    _append_csv_row(csv_path, row)
    print(f"Result appended to {csv_path}")
    return csv_path