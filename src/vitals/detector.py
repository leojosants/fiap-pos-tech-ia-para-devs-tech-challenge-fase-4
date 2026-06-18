
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# Colunas de sinais vitais que serão analisadas
VITAL_COLUMNS = [
    "heart_rate_bpm",
    "spo2_pct",
    "bp_systolic_mmhg",
    "bp_diastolic_mmhg"
]

# Limiar do z-score: pontos com |z| > 3 são considerados anomalias
ZSCORE_THRESHOLD = 3.0

# Configuração do Isolation Forest
ISOLATION_FOREST_CONTAMINATION = 0.015  # Espera ~5% de anomalias no sinal


def detect_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta anomalias usando z-score por coluna de sinal vital.

    Conceito: o z-score mede quantos desvios padrão um ponto está
    da média. Pontos com |z| > 3 são estatisticamente improváveis
    em uma distribuição normal (~0.3% de chance) — logo, anomalias.

    Retorna o DataFrame com colunas adicionais de z-score e flag de anomalia.
    """
    df = df.copy()

    zscore_flags = pd.Series(False, index=df.index)

    for col in VITAL_COLUMNS:
        mean = df[col].mean()
        std = df[col].std()

        # Evita divisão por zero em sinais constantes
        if std == 0:
            df[f"zscore_{col}"] = 0.0
            continue

        z = (df[col] - mean) / std
        df[f"zscore_{col}"] = np.round(z, 4)

        # Marca como anomalia se qualquer sinal ultrapassar o limiar
        zscore_flags |= (np.abs(z) > ZSCORE_THRESHOLD)

    df["anomaly_zscore"] = zscore_flags.astype(int)

    return df


def detect_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta anomalias usando Isolation Forest.

    Conceito: o Isolation Forest isola pontos anômalos construindo
    árvores de decisão aleatórias. Pontos que precisam de menos
    divisões para serem isolados são mais prováveis de serem anomalias.
    É eficaz para dados multivariados (múltiplos sinais ao mesmo tempo).

    Retorna o DataFrame com coluna adicional de anomalia pelo IF.
    """
    df = df.copy()

    # Normaliza os dados antes de aplicar o modelo
    scaler = StandardScaler()
    X = scaler.fit_transform(df[VITAL_COLUMNS])

    # Treina o modelo (não supervisionado — não usa o gabarito)
    model = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=42,
        n_estimators=100
    )
    predictions = model.fit_predict(X)

    # Isolation Forest retorna -1 para anomalias e 1 para normais
    # Convertemos para 1 = anomalia, 0 = normal (nosso padrão)
    df["anomaly_isolation_forest"] = (predictions == -1).astype(int)

    return df


def combine_detections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina os dois métodos de detecção em uma flag unificada.

    Estratégia: um ponto é marcado como anomalia final se
    pelo menos UM dos dois métodos o detectou.
    Isso maximiza a sensibilidade (recall) — em contexto médico,
    é preferível ter falsos positivos a perder uma anomalia real.
    """
    df = df.copy()
    df["anomaly_final"] = (
        (df["anomaly_zscore"] == 1) | (df["anomaly_isolation_forest"] == 1)
    ).astype(int)

    return df


def evaluate_detection(df: pd.DataFrame) -> dict:
    """
    Avalia a qualidade da detecção comparando com o gabarito
    (coluna is_injected_anomaly gerada pelo generator.py).

    Retorna um dicionário com métricas para cada método.
    """
    results = {}

    methods = {
        "z-score": "anomaly_zscore",
        "isolation_forest": "anomaly_isolation_forest",
        "combinado": "anomaly_final"
    }

    for method_name, col in methods.items():
        if col not in df.columns:
            continue

        report = classification_report(
            df["is_injected_anomaly"],
            df[col],
            target_names=["Normal", "Anomalia"],
            output_dict=True,
            zero_division=0
        )

        results[method_name] = {
            "precisao": round(report["Anomalia"]["precision"], 3),
            "recall": round(report["Anomalia"]["recall"], 3),
            "f1_score": round(report["Anomalia"]["f1-score"], 3),
            "total_detectadas": int(df[col].sum()),
            "total_injetadas": int(df["is_injected_anomaly"].sum())
        }

    return results


def run_detection_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Executa o pipeline completo de detecção:
    1. Z-score
    2. Isolation Forest
    3. Combinação dos resultados
    4. Avaliação vs gabarito

    Retorna o DataFrame enriquecido e as métricas de avaliação.
    """
    df = detect_zscore(df)
    df = detect_isolation_forest(df)
    df = combine_detections(df)
    metrics = evaluate_detection(df)

    return df, metrics


if __name__ == "__main__":
    from src.vitals.generator import generate_vitals_dataframe

    print("Gerando dados sintéticos...")
    df = generate_vitals_dataframe()

    print("Executando pipeline de detecção...")
    df_result, metrics = run_detection_pipeline(df)

    print("\n=== RESULTADOS DA DETECÇÃO ===")
    for method, m in metrics.items():
        print(f"\n[{method.upper()}]")
        print(f"  Anomalias injetadas : {m['total_injetadas']}")
        print(f"  Anomalias detectadas: {m['total_detectadas']}")
        print(f"  Precisão            : {m['precisao']}")
        print(f"  Recall              : {m['recall']}")
        print(f"  F1-Score            : {m['f1_score']}")