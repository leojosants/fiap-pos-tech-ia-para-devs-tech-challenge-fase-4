"""
pages_video.py
===============
Renderiza a aba de Análise Postural (vídeo) da interface Streamlit:
métricas resumidas, vídeo sintético gerado, gráfico de ângulos
articulares e frames anotados representativos.

Consome os dados já processados pelo pipeline da Etapa 2.
"""

from pathlib import Path

import streamlit as st
import pandas as pd


def render_video_tab(
    df_posture: pd.DataFrame,
    metrics: dict,
    video_path: Path,
) -> None:
    """
    Renderiza a aba de Análise Postural.

    Parâmetros:
        df_posture: DataFrame já processado por
                    src.video.anomaly_detector.run_video_detection_pipeline()
        metrics   : dicionário de métricas (saída de evaluate_detection())
        video_path: caminho do vídeo sintético gerado (synthetic_pose.mp4)
    """
    st.subheader("🎥 Análise Postural (Vídeo)")

    n_total = len(df_posture)
    n_anomaly = int(df_posture["anomaly_final"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames analisados", n_total)
    col2.metric("Frames com anomalia", n_anomaly)

    combinado = metrics.get("combinado", {})
    col3.metric("Precisão (combinado)", f"{combinado.get('precisao', 0):.1%}")
    col4.metric("Recall (combinado)", f"{combinado.get('recall', 0):.1%}")

    st.markdown("---")

    col_video, col_chart = st.columns([1, 1.4])

    with col_video:
        st.markdown("**Vídeo sintético (sessão de fisioterapia)**")
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.info("Vídeo não disponível nesta execução.")
        st.caption(
            "Gerado matematicamente (NumPy/OpenCV) simulando exercício de "
            "elevação bilateral de braços, sem necessidade de câmera real."
        )

    with col_chart:
        st.markdown("**Ângulos articulares ao longo do tempo**")
        angle_cols = [
            c for c in
            ["angle_left_shoulder", "angle_right_shoulder",
             "angle_left_elbow", "angle_right_elbow"]
            if c in df_posture.columns
        ]
        if angle_cols:
            chart_df = df_posture.set_index("frame_idx")[angle_cols]
            st.line_chart(chart_df, height=280)
        st.caption(
            "Picos e assimetrias entre ângulos esquerdo/direito indicam "
            "possível hiperextensão ou compensação unilateral."
        )

    # ── Tabela de anomalias detectadas ────────────────────────────────────────
    with st.expander(f"Ver {n_anomaly} frame(s) com anomalia em detalhe"):
        anomaly_rows = df_posture[df_posture["anomaly_final"] == 1]
        display_cols = [
            c for c in
            ["frame_idx", "angle_left_shoulder", "angle_right_shoulder",
             "rule_asymmetry", "rule_trunk_collapse", "is_injected_anomaly"]
            if c in anomaly_rows.columns
        ]
        st.dataframe(
            anomaly_rows[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Ver métricas por método de detecção"):
        for method_name, m in metrics.items():
            st.markdown(
                f"**{method_name}**: Precisão={m['precisao']:.3f} | "
                f"Recall={m['recall']:.3f} | F1={m['f1_score']:.3f} "
                f"({m['total_detectadas']}/{m['total_injetadas']} detectadas)"
            )