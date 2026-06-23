"""
pages_audio.py
===============
Renderiza a aba de Análise de Fala (áudio) da interface Streamlit:
métricas resumidas, player de áudio por segmento, transcrição e
classificação de urgência via LLM.

Consome os dados já processados pelos pipelines das Etapas 3 e 4.
"""

from pathlib import Path

import streamlit as st
import pandas as pd


URGENCY_COLOR = {
    "baixo": "🟢",
    "medio": "🟡",
    "alto": "🔴",
}


def render_audio_tab(df_speech: pd.DataFrame, metrics: dict) -> None:
    """
    Renderiza a aba de Análise de Fala.

    Parâmetros:
        df_speech: DataFrame já processado pelos pipelines de
                   src.audio.anomaly_detector e
                   src.llm.text_analyzer.analyze_text_dataset()
        metrics  : dicionário de métricas (saída de evaluate_detection())
    """
    st.subheader("🎙️ Análise de Fala (Áudio)")

    n_total = len(df_speech)
    n_anomaly = int(df_speech["anomaly_final"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frases analisadas", n_total)
    col2.metric("Anomalias de fala", n_anomaly)

    combinado = metrics.get("combinado", {})
    col3.metric("Precisão (combinado)", f"{combinado.get('precisao', 0):.1%}")
    col4.metric("Recall (combinado)", f"{combinado.get('recall', 0):.1%}")

    st.markdown("---")

    # ── Gráfico de velocidade de fala por segmento ────────────────────────────
    if "words_per_second" in df_speech.columns:
        st.markdown("**Velocidade de fala por segmento**")
        chart_df = df_speech.set_index("segment_id")[["words_per_second"]]
        st.bar_chart(chart_df, height=220)

    st.markdown("---")
    st.markdown("**Transcrição e classificação por frase**")

    for _, row in df_speech.iterrows():
        urgency = row.get("nivel_urgencia", "baixo")
        icon = URGENCY_COLOR.get(urgency, "⚪")
        anomaly_flag = "⚠️" if row.get("anomaly_final", 0) else ""

        with st.container(border=True):
            cols = st.columns([0.08, 0.62, 0.15, 0.15])
            cols[0].markdown(f"### {icon}")
            cols[1].markdown(f"*\"{row.get('transcribed_text', '')}\"*")
            cols[2].markdown(f"Urgência: **{urgency}**")
            cols[3].markdown(f"{anomaly_flag} {row.get('words_per_second', 0):.2f} p/s")

            terms = row.get("termos_criticos", [])
            if isinstance(terms, list) and terms:
                st.caption(f"Termos críticos: {', '.join(terms)}")

            # Defensivo: file_path pode estar ausente, vazio ou apontar
            # para um caminho inexistente (ex.: dados de fallback sem
            # áudio real gerado) — nunca deixa a aplicação quebrar por
            # causa do player de áudio.
            file_path_value = row.get("file_path")
            if file_path_value:
                audio_path = Path(file_path_value)
                if audio_path.exists() and audio_path.is_file():
                    st.audio(str(audio_path))

    with st.expander("Ver métricas por método de detecção"):
        for method_name, m in metrics.items():
            st.markdown(
                f"**{method_name}**: Precisão={m['precisao']:.3f} | "
                f"Recall={m['recall']:.3f} | F1={m['f1_score']:.3f} "
                f"({m['total_detectadas']}/{m['total_injetadas']} detectadas)"
            )