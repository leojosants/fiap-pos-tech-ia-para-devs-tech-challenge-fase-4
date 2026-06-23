"""
anomaly_detector.py  (src/audio/)
==================================
Detecta anomalias de fala no DataFrame produzido pelo audio_processor.py.

Dois métodos combinados (mesmo padrão das Etapas 1 e 2):

  1. Z-score sobre velocidade de fala (palavras/segundo):
       Desvios estatísticos da velocidade média indicam fala anormalmente
       lenta (possível fadiga/confusão/disartria) ou rápida (possível
       agitação/ansiedade/taquilalia).

  2. Regras clínicas:
       - Fala lentificada : silence_ratio > limiar E words_per_second < limiar
       - Fala acelerada   : words_per_second > limiar (muito acima da média)

Avaliação idêntica às Etapas 1 e 2 (precisão/recall/F1 vs gabarito
is_injected_anomaly).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


# ── Configurações ─────────────────────────────────────────────────────────────
ZSCORE_THRESHOLD          = 1.5    # mais sensível — poucos segmentos no dataset
SILENCE_RATIO_THRESHOLD   = 0.40   # acima disso = pausas anormais
SLOW_SPEECH_WPS_THRESHOLD = 1.0    # palavras/segundo abaixo disso = fala lenta


# ── Método 1: Z-score sobre velocidade de fala ────────────────────────────────

def detect_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score sobre words_per_second. Desvios para baixo (fala lenta) e para
    cima (fala rápida) são ambos marcados — outliers em qualquer direção.
    """
    df = df.copy()
    col = "words_per_second"

    flag = pd.Series(False, index=df.index)
    z = pd.Series(float("nan"), index=df.index)

    if col in df.columns:
        valid = df[col].notna()
        if valid.sum() > 1:
            mean_v = df.loc[valid, col].mean()
            std_v  = df.loc[valid, col].std()
            if std_v > 0:
                z[valid] = (df.loc[valid, col] - mean_v) / std_v
        flag = (np.abs(z) > ZSCORE_THRESHOLD).fillna(False)

    df["zscore_speech_rate"] = z.round(4)
    df["anomaly_zscore"] = flag.astype(int)
    return df


# ── Método 2: Regras clínicas ─────────────────────────────────────────────────

def detect_clinical_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regras de limiar fixo, calibradas para o cenário de triagem médica:

    Regra 1 — Fala lentificada:
        silence_ratio alto (muitas pausas) E velocidade baixa.
        Combinação que diferencia "pausa natural entre frases" de
        "hesitação/lentidão patológica dentro da própria frase".

    Regra 2 — Fala acelerada:
        velocidade muito acima do padrão normal de fala em português
        (estimado empiricamente neste dataset).
    """
    df = df.copy()

    slow_flag = pd.Series(False, index=df.index)
    if "silence_ratio" in df.columns and "words_per_second" in df.columns:
        slow_flag = (
            (df["silence_ratio"] > SILENCE_RATIO_THRESHOLD) &
            (df["words_per_second"] < SLOW_SPEECH_WPS_THRESHOLD)
        ).fillna(False)
    df["rule_slow_speech"] = slow_flag.astype(int)

    fast_flag = pd.Series(False, index=df.index)
    if "words_per_second" in df.columns:
        valid = df["words_per_second"].notna()
        if valid.sum() > 1:
            mean_wps = df.loc[valid, "words_per_second"].mean()
            std_wps  = df.loc[valid, "words_per_second"].std()
            threshold = mean_wps + 1.0 * std_wps
            fast_flag = (df["words_per_second"] > threshold).fillna(False)
    df["rule_fast_speech"] = fast_flag.astype(int)

    df["anomaly_clinical"] = (slow_flag | fast_flag).astype(int)
    return df


# ── Combinação ────────────────────────────────────────────────────────────────

def combine_detections(df: pd.DataFrame) -> pd.DataFrame:
    """OR lógico — maximiza recall, mesma filosofia das etapas anteriores."""
    df = df.copy()
    df["anomaly_final"] = (
        (df["anomaly_zscore"]   == 1) |
        (df["anomaly_clinical"] == 1)
    ).astype(int)
    return df


# ── Avaliação ─────────────────────────────────────────────────────────────────

def evaluate_detection(df: pd.DataFrame) -> dict:
    """Avalia contra gabarito. Mesmo padrão das Etapas 1 e 2."""
    results: dict = {}
    methods = {
        "z-score":         "anomaly_zscore",
        "regras_clinicas": "anomaly_clinical",
        "combinado":       "anomaly_final",
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

def run_audio_detection_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline completo:
        1. Z-score de velocidade de fala
        2. Regras clínicas (fala lenta / fala rápida)
        3. Combinação OR
        4. Avaliação vs gabarito
    """
    df = detect_zscore(df)
    df = detect_clinical_rules(df)
    df = combine_detections(df)
    metrics = evaluate_detection(df)
    return df, metrics


if __name__ == "__main__":
    from pathlib import Path
    from src.audio.audio_processor import process_audio_dataset

    meta_path = Path("data/raw/audio_metadata.json")

    if not meta_path.exists():
        print(f"Metadados não encontrados em {meta_path}.")
        print("Execute primeiro: uv run python -m src.audio.synthetic_audio")
    else:
        print("Processando áudio com Whisper...")
        df_audio = process_audio_dataset(meta_path)

        print("Detectando anomalias de fala...")
        df_result, metrics = run_audio_detection_pipeline(df_audio)

        print("\n=== RESULTADOS DA DETECÇÃO DE ANOMALIAS DE FALA ===")
        for method, m in metrics.items():
            print(f"\n[{method.upper()}]")
            print(f"  Anomalias injetadas : {m['total_injetadas']}")
            print(f"  Anomalias detectadas: {m['total_detectadas']}")
            print(f"  Precisão            : {m['precisao']}")
            print(f"  Recall              : {m['recall']}")
            print(f"  F1-Score            : {m['f1_score']}")