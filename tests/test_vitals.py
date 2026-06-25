"""
test_vitals.py
===============
Testes do módulo de sinais vitais (Etapa 1).

Cobre a lógica determinística principal: geração reprodutível de dados
sintéticos e a eficácia do detector combinado contra o gabarito conhecido.
Nenhuma chamada externa (rede, API) é necessária — testes rápidos e
100% offline.
"""

import pandas as pd

from src.vitals.generator import generate_vitals_dataframe
from src.vitals.detector import run_detection_pipeline


def test_generate_vitals_dataframe_is_reproducible():
    """
    A mesma seed deve sempre produzir o mesmo DataFrame — propriedade
    essencial para que o gabarito (is_injected_anomaly) seja confiável
    em toda execução do pipeline, incluindo no deploy.
    """
    df1 = generate_vitals_dataframe(seed=42)
    df2 = generate_vitals_dataframe(seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_generate_vitals_dataframe_has_expected_shape_and_anomalies():
    """
    Verifica a estrutura básica esperada pelo restante do pipeline:
    600 amostras, as 4 colunas de sinais vitais, e exatamente 6 anomalias
    injetadas (índices fixos documentados no relatório técnico).
    """
    df = generate_vitals_dataframe(seed=42)

    assert len(df) == 600
    for col in ["heart_rate_bpm", "spo2_pct", "bp_systolic_mmhg", "bp_diastolic_mmhg"]:
        assert col in df.columns
    assert df["is_injected_anomaly"].sum() == 6


def test_combined_detector_achieves_perfect_recall():
    """
    Regressão do resultado documentado no relatório técnico (Seção 5.2):
    o detector combinado (z-score OR Isolation Forest) deve manter
    recall de 1.0 — nenhuma anomalia real perdida — propriedade central
    da decisão de design deste módulo (preferir falso positivo a falso
    negativo em contexto clínico).
    """
    df = generate_vitals_dataframe(seed=42)
    df_result, metrics = run_detection_pipeline(df)

    assert metrics["combinado"]["recall"] == 1.0
    # Toda anomalia injetada deve estar contida no conjunto detectado
    injected_idx = set(df_result[df_result["is_injected_anomaly"] == 1].index)
    detected_idx = set(df_result[df_result["anomaly_final"] == 1].index)
    assert injected_idx.issubset(detected_idx)