"""
pages_alerts.py
================
Renderiza a aba de Alertas da interface Streamlit: nível de alerta
consolidado, status de cada modalidade, resumo clínico do LLM e
histórico de execuções anteriores.

Consome a FusionResult da Etapa 5 e o histórico persistido em JSON.
"""

import streamlit as st

from src.alerts.alert_rules import AlertLevel
from src.alerts.fusion_engine import FusionResult
from src.alerts.alert_dispatcher import load_alert_history


ALERT_STREAMLIT_TYPE = {
    AlertLevel.NORMAL: "success",
    AlertLevel.ATENCAO: "warning",
    AlertLevel.CRITICO: "error",
}

ALERT_EMOJI = {
    AlertLevel.NORMAL: "✅",
    AlertLevel.ATENCAO: "⚠️",
    AlertLevel.CRITICO: "🚨",
}


def render_alerts_tab(fusion_result: FusionResult) -> None:
    """
    Renderiza a aba de Alertas e Fusão Multimodal.

    Parâmetros:
        fusion_result: saída de src.alerts.fusion_engine.run_fusion_pipeline()
    """
    st.subheader("🚨 Alerta Consolidado")

    decision = fusion_result.alert_decision
    level = decision.level
    emoji = ALERT_EMOJI[level]
    alert_fn = getattr(st, ALERT_STREAMLIT_TYPE[level])

    alert_fn(f"{emoji} **NÍVEL DE ALERTA: {level.value.upper()}**")
    st.caption(decision.reasoning)

    st.markdown("---")

    cols = st.columns(len(decision.modalities))
    for col, modality in zip(cols, decision.modalities):
        with col:
            icon = "⚠️" if modality.has_anomaly else "✅"
            st.markdown(f"#### {icon} {modality.name.capitalize()}")
            st.caption(modality.detail)

    st.markdown("---")

    if fusion_result.clinical_summary:
        st.markdown("### 📋 Resumo Clínico Executivo")
        st.info(fusion_result.clinical_summary)
    else:
        st.caption(
            "Resumo clínico não gerado nesta execução "
            "(generate_clinical_summary=False)."
        )

    # ── Histórico de alertas ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🕓 Histórico de Alertas")

    history = load_alert_history()
    if not history:
        st.caption("Nenhum alerta registrado ainda nesta sessão do servidor.")
        return

    for entry in reversed(history[-10:]):  # 10 mais recentes, mais recente primeiro
        entry_level = entry["decision"]["level"]
        entry_emoji = ALERT_EMOJI.get(AlertLevel(entry_level), "⚪")
        timestamp = entry["timestamp"][:19].replace("T", " ")
        with st.expander(
            f"{entry_emoji} {timestamp} — {entry['patient_id']} "
            f"— {entry_level.upper()}"
        ):
            st.caption(entry["decision"]["reasoning"])
            if entry.get("clinical_summary"):
                st.write(entry["clinical_summary"])