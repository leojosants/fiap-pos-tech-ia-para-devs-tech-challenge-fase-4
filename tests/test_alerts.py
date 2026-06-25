"""
test_alerts.py
===============
Testes do motor de decisão de alerta (Etapa 5).

Cobre a regra de decisão central do projeto — combinação de contagem de
modalidades anômalas com o nível de urgência apontado pelo LLM — usando
os mesmos cenários validados manualmente durante o desenvolvimento
(Seção 10.2 do relatório técnico). Lógica 100% determinística, sem
dependências externas.
"""

from src.alerts.alert_rules import AlertLevel, ModalityStatus, decide_alert_level


def test_no_anomalies_and_low_urgency_results_in_normal_level():
    """Paciente estável: nenhuma modalidade com anomalia, urgência baixa."""
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", False),
            ModalityStatus("postura", False),
            ModalityStatus("fala", False),
        ],
        llm_urgency="baixo",
    )

    assert decision.level == AlertLevel.NORMAL


def test_single_modality_anomaly_results_in_atencao_level():
    """Exatamente uma modalidade com anomalia isolada → nível Atenção."""
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", False),
            ModalityStatus("postura", True, "assimetria detectada"),
            ModalityStatus("fala", False),
        ],
        llm_urgency="baixo",
    )

    assert decision.level == AlertLevel.ATENCAO
    assert decision.n_modalities_with_anomaly == 1


def test_two_or_more_modality_anomalies_result_in_critico_level():
    """Duas ou mais modalidades simultâneas → nível Crítico, por contagem."""
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", True, "bradicardia"),
            ModalityStatus("postura", True, "hiperextensão"),
            ModalityStatus("fala", False),
        ],
        llm_urgency="baixo",
    )

    assert decision.level == AlertLevel.CRITICO


def test_high_llm_urgency_alone_elevates_to_critico_even_without_detectors():
    """
    Caso central da decisão de design do projeto: mesmo sem nenhum
    detector estatístico disparando, uma urgência "alto" apontada pela
    análise de linguagem (paciente verbaliza sintoma grave) já é
    suficiente para elevar o alerta a Crítico — o sistema nunca rebaixa
    um alerta por falta de confirmação estatística, apenas eleva.
    """
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", False),
            ModalityStatus("postura", False),
            ModalityStatus("fala", False),
        ],
        llm_urgency="alto",
    )

    assert decision.level == AlertLevel.CRITICO
    assert decision.n_modalities_with_anomaly == 0  # confirma que veio do LLM, não da contagem


def test_reasoning_field_is_never_empty():
    """
    A justificativa textual da decisão (campo reasoning) deve sempre
    ser preenchida — requisito de auditabilidade do sistema (Seção 13
    do relatório técnico, Considerações Éticas).
    """
    decision = decide_alert_level(
        modalities=[ModalityStatus("vitais", False)],
        llm_urgency="baixo",
    )

    assert decision.reasoning.strip() != ""