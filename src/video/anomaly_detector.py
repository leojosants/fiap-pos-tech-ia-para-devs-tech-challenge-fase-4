"""
anomaly_detector.py
===================
Detecta anomalias posturais no DataFrame produzido pelo video_processor.py.

Três métodos combinados (mesmo padrão da Etapa 1):

  1. Z-score sobre assimetria bilateral:
       Diferença entre ângulo esquerdo e direito.
       Em movimento simétrico normal, diff ≈ 0.
       Limiar: |z| > 2.0

  2. Z-score sobre velocidade angular (variação frame-a-frame):
       Mudança brusca de ângulo entre frames consecutivos.
       Hiperextensão aparece como aceleração anômala.
       Limiar: |z| > 2.5

  3. Regras clínicas fixas:
       - Assimetria     : |ângulo_esq - ângulo_dir| > 25°
       - Colapso tronco : z-score do centro horizontal dos ombros > 2.0

Combinação: OR dos três métodos → maximiza Recall.

Avaliação idêntica à Etapa 1 (precision/recall/F1 vs gabarito).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


# ── Configurações ─────────────────────────────────────────────────────────────
ZSCORE_ASYMMETRY_THRESHOLD  = 2.0   # z-score da diferença bilateral
ZSCORE_VELOCITY_THRESHOLD   = 2.5   # z-score da velocidade angular
ASYMMETRY_DEG_THRESHOLD     = 25.0  # graus de diferença direta entre ombros


# ── Método 1: Z-score de assimetria bilateral ─────────────────────────────────

def detect_zscore_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score sobre a diferença angular entre ombro esquerdo e direito.
    Captura padrões de assimetria e compensação unilateral.
    """
    df = df.copy()
    flag = pd.Series(False, index=df.index)

    if "angle_left_shoulder" in df.columns and "angle_right_shoulder" in df.columns:
        diff = (df["angle_left_shoulder"] - df["angle_right_shoulder"]).abs()
        valid = diff.notna()
        z = pd.Series(float("nan"), index=df.index)
        if valid.sum() > 1:
            mean_d = diff[valid].mean()
            std_d  = diff[valid].std()
            if std_d > 0:
                z[valid] = (diff[valid] - mean_d) / std_d
        df["zscore_shoulder_asymmetry"] = z.round(4)
        flag |= (np.abs(z) > ZSCORE_ASYMMETRY_THRESHOLD).fillna(False)
    else:
        df["zscore_shoulder_asymmetry"] = float("nan")

    df["anomaly_zscore_asymmetry"] = flag.astype(int)
    return df


# ── Método 2: Z-score de velocidade angular ───────────────────────────────────

def detect_zscore_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score sobre a variação frame-a-frame de cada ângulo de ombro.
    Captura mudanças bruscas típicas de hiperextensão e movimentos patológicos.
    """
    df = df.copy()
    flag = pd.Series(False, index=df.index)

    for col in ["angle_left_shoulder", "angle_right_shoulder"]:
        if col not in df.columns:
            df[f"velocity_{col}"] = float("nan")
            df[f"zscore_vel_{col}"] = float("nan")
            continue

        vel = df[col].diff().abs()
        df[f"velocity_{col}"] = vel.round(4)

        valid = vel.notna()
        z = pd.Series(float("nan"), index=df.index)
        if valid.sum() > 1:
            mean_v = vel[valid].mean()
            std_v  = vel[valid].std()
            if std_v > 0:
                z[valid] = (vel[valid] - mean_v) / std_v
        df[f"zscore_vel_{col}"] = z.round(4)
        flag |= (np.abs(z) > ZSCORE_VELOCITY_THRESHOLD).fillna(False)

    df["anomaly_zscore_velocity"] = flag.astype(int)
    return df


# ── Método 3: Regras clínicas ─────────────────────────────────────────────────

def detect_clinical_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regras biomecânicas fixas como complemento aos métodos estatísticos:

    Regra 1 — Assimetria direta:
        |ângulo_esq - ângulo_dir| > 25° detecta compensações laterais
        que podem não aparecer no z-score se forem graduais.

    Regra 2 — Colapso de tronco:
        Z-score do centro horizontal dos ombros detecta inclinação lateral
        do tronco (ambos os ombros migram para um lado).
    """
    df = df.copy()

    # Regra 1: Assimetria direta
    asym_flag = pd.Series(False, index=df.index)
    if "angle_left_shoulder" in df.columns and "angle_right_shoulder" in df.columns:
        diff = (df["angle_left_shoulder"] - df["angle_right_shoulder"]).abs()
        asym_flag = (diff > ASYMMETRY_DEG_THRESHOLD).fillna(False)
    df["rule_asymmetry"] = asym_flag.astype(int)

    # Regra 2: Colapso de tronco via z-score do centro horizontal
    trunk_flag = pd.Series(False, index=df.index)
    if "left_shoulder_x" in df.columns and "right_shoulder_x" in df.columns:
        center_x = (df["left_shoulder_x"] + df["right_shoulder_x"]) / 2
        valid = center_x.notna()
        z_center = pd.Series(float("nan"), index=df.index)
        if valid.sum() > 1:
            mean_c = center_x[valid].mean()
            std_c  = center_x[valid].std()
            if std_c > 0:
                z_center[valid] = (center_x[valid] - mean_c) / std_c
        df["zscore_trunk_center"] = z_center.round(4)
        trunk_flag = (np.abs(z_center) > 2.0).fillna(False)
    else:
        df["zscore_trunk_center"] = float("nan")
    df["rule_trunk_collapse"] = trunk_flag.astype(int)

    df["anomaly_clinical"] = (asym_flag | trunk_flag).astype(int)
    return df


# ── Combinação ────────────────────────────────────────────────────────────────

def combine_detections(df: pd.DataFrame) -> pd.DataFrame:
    """OR dos três métodos — maximiza Recall."""
    df = df.copy()
    df["anomaly_final"] = (
        (df["anomaly_zscore_asymmetry"] == 1) |
        (df["anomaly_zscore_velocity"]  == 1) |
        (df["anomaly_clinical"]         == 1)
    ).astype(int)

    # Flag unificada de z-score (para compatibilidade com plotter.py)
    df["anomaly_zscore"] = (
        (df["anomaly_zscore_asymmetry"] == 1) |
        (df["anomaly_zscore_velocity"]  == 1)
    ).astype(int)

    return df


# ── Avaliação ─────────────────────────────────────────────────────────────────

def evaluate_detection(df: pd.DataFrame) -> dict:
    """Avalia todos os métodos vs gabarito. Mesmo padrão da Etapa 1."""
    results: dict = {}
    methods = {
        "z-score_assimetria":  "anomaly_zscore_asymmetry",
        "z-score_velocidade":  "anomaly_zscore_velocity",
        "regras_clinicas":     "anomaly_clinical",
        "combinado":           "anomaly_final",
    }
    for method_name, col in methods.items():
        if col not in df.columns:
            continue
        report = classification_report(
            df["is_injected_anomaly"],
            df[col],
            target_names=["Normal", "Anomalia"],
            output_dict=True,
            zero_division=0,
        )
        results[method_name] = {
            "precisao":         round(report["Anomalia"]["precision"], 3),
            "recall":           round(report["Anomalia"]["recall"],    3),
            "f1_score":         round(report["Anomalia"]["f1-score"],  3),
            "total_detectadas": int(df[col].sum()),
            "total_injetadas":  int(df["is_injected_anomaly"].sum()),
        }
    return results


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_video_detection_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline completo:
        1. Z-score de assimetria bilateral
        2. Z-score de velocidade angular
        3. Regras clínicas
        4. Combinação OR
        5. Avaliação vs gabarito
    """
    df = detect_zscore_asymmetry(df)
    df = detect_zscore_velocity(df)
    df = detect_clinical_rules(df)
    df = combine_detections(df)
    metrics = evaluate_detection(df)
    return df, metrics


if __name__ == "__main__":
    from src.video.synthetic_video import generate_synthetic_video, ALL_ANOMALY_FRAMES
    from src.video.video_processor import process_video

    print("Gerando vídeo sintético...")
    frames, meta = generate_synthetic_video()

    print("Processando frames...")
    df_pose = process_video(frames, anomaly_frame_indices=ALL_ANOMALY_FRAMES)

    print("Detectando anomalias posturais...")
    df_result, metrics = run_video_detection_pipeline(df_pose)

    print("\n=== RESULTADOS DA DETECÇÃO DE POSE ===")
    for method, m in metrics.items():
        print(f"\n[{method.upper()}]")
        print(f"  Anomalias injetadas : {m['total_injetadas']}")
        print(f"  Anomalias detectadas: {m['total_detectadas']}")
        print(f"  Precisão            : {m['precisao']}")
        print(f"  Recall              : {m['recall']}")
        print(f"  F1-Score            : {m['f1_score']}")