"""
pages_vitals.py
================
Renderiza a aba de Sinais Vitais da interface Streamlit: métricas
resumidas, gráfico do painel completo (4 sinais) e tabela de detecção.

Consome os DataFrames já processados pelo pipeline da Etapa 1
(armazenados em st.session_state pelo main.py após a execução).
"""

import streamlit as st
import pandas as pd


def render_vitals_tab(df_vitals: pd.DataFrame, metrics: dict) -> None:
    """
    Renderiza a aba de Sinais Vitais.

    Parâmetros:
        df_vitals: DataFrame já processado por
                   src.vitals.detector.run_detection_pipeline()
        metrics  : dicionário de métricas (saída de evaluate_detection())
    """
    st.subheader("📊 Sinais Vitais")

    n_total = len(df_vitals)
    n_anomaly = int(df_vitals["anomaly_final"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Amostras monitoradas", n_total)
    col2.metric("Anomalias detectadas", n_anomaly)

    combinado = metrics.get("combinado", {})
    col3.metric("Precisão (combinado)", f"{combinado.get('precisao', 0):.1%}")
    col4.metric("Recall (combinado)", f"{combinado.get('recall', 0):.1%}")

    st.markdown("---")

    # ── Gráfico de linha: HR, SpO2, pressão ao longo do tempo ─────────────────
    chart_df = df_vitals.set_index("timestamp")[
        ["heart_rate_bpm", "spo2_pct", "bp_systolic_mmhg", "bp_diastolic_mmhg"]
    ]
    st.line_chart(chart_df, height=300)

    st.caption(
        "Linhas representam os 4 sinais vitais monitorados ao longo de 600 "
        "amostras (10 minutos, 1 amostra/segundo). Anomalias foram injetadas "
        "em 6 índices fixos e detectadas via z-score + Isolation Forest."
    )

    # ── Tabela de anomalias detectadas ────────────────────────────────────────
    with st.expander(f"Ver {n_anomaly} anomalia(s) detectada(s) em detalhe"):
        anomaly_rows = df_vitals[df_vitals["anomaly_final"] == 1]
        st.dataframe(
            anomaly_rows[
                ["timestamp", "heart_rate_bpm", "spo2_pct",
                 "bp_systolic_mmhg", "bp_diastolic_mmhg", "is_injected_anomaly"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    # ── Métricas por método de detecção ───────────────────────────────────────
    with st.expander("Ver métricas por método de detecção"):
        for method_name, m in metrics.items():
            st.markdown(
                f"**{method_name}**: Precisão={m['precisao']:.3f} | "
                f"Recall={m['recall']:.3f} | F1={m['f1_score']:.3f} "
                f"({m['total_detectadas']}/{m['total_injetadas']} detectadas)"
            )