"""
alert_rules.py
==============
Define os níveis de alerta do sistema e a lógica de decisão que combina
os resultados das três modalidades monitoradas (sinais vitais, postura,
fala) e da camada de LLM (Etapa 4) em um único nível de risco consolidado.

Este módulo NÃO realiza nenhuma detecção nova — ele consome os resultados
já produzidos pelos detectores das Etapas 1-3 e pela análise de texto da
Etapa 4, aplicando uma regra de decisão de mais alto nível: "dado tudo que
sabemos sobre este paciente neste momento, qual o nível de atenção
necessário da equipe médica?"

Níveis de alerta (alinhados a escalas de triagem clínica real, como
Manchester):
    NORMAL   : nenhuma anomalia detectada em nenhuma modalidade.
    ATENCAO  : uma única modalidade com anomalia, OU urgência "medio"
               segundo a análise de linguagem (Etapa 4).
    CRITICO  : duas ou mais modalidades com anomalia simultânea, OU
               urgência "alto" segundo a análise de linguagem.

A regra de "2+ modalidades simultâneas = crítico" reflete um princípio
clínico real: anomalias concorrentes em múltiplos sistemas (cardiovascular,
neuromuscular, comportamental) são mais preocupantes do que uma anomalia
isolada, mesmo que cada uma isoladamente pareça moderada.
"""

from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(str, Enum):
    """Níveis de alerta do sistema, em ordem crescente de gravidade."""
    NORMAL = "normal"
    ATENCAO = "atencao"
    CRITICO = "critico"


# Mapeamento de urgência do LLM (Etapa 4) para contribuição ao alerta
LLM_URGENCY_TO_LEVEL = {
    "baixo": AlertLevel.NORMAL,
    "medio": AlertLevel.ATENCAO,
    "alto": AlertLevel.CRITICO,
}

_LEVEL_ORDER = {AlertLevel.NORMAL: 0, AlertLevel.ATENCAO: 1, AlertLevel.CRITICO: 2}


@dataclass
class ModalityStatus:
    """Status de anomalia de uma única modalidade monitorada."""
    name: str                  # "vitais" | "postura" | "fala"
    has_anomaly: bool
    detail: str = ""           # breve descrição do achado, para o alerta


@dataclass
class AlertDecision:
    """Resultado da decisão de alerta consolidada."""
    level: AlertLevel
    n_modalities_with_anomaly: int
    modalities: list[ModalityStatus] = field(default_factory=list)
    llm_urgency: str = "baixo"
    reasoning: str = ""        # explicação textual da decisão, para auditoria


def _max_level(a: AlertLevel, b: AlertLevel) -> AlertLevel:
    """Retorna o nível de maior gravidade entre dois níveis."""
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b


def decide_alert_level(
    modalities: list[ModalityStatus],
    llm_urgency: str = "baixo",
) -> AlertDecision:
    """
    Decide o nível de alerta consolidado a partir do status de cada
    modalidade e do nível de urgência apontado pela análise de linguagem.

    Parâmetros:
        modalities : lista de ModalityStatus, uma por modalidade monitorada
                     (tipicamente 3: vitais, postura, fala)
        llm_urgency: "baixo" | "medio" | "alto" — saída agregada da Etapa 4
                     (ex.: maior nível de urgência entre as frases do paciente)

    Retorna AlertDecision com o nível final e a justificativa.

    Regra de decisão:
        1. Conta quantas modalidades têm anomalia (n_modalities_with_anomaly).
        2. Nível por contagem: 0 -> NORMAL, 1 -> ATENCAO, 2+ -> CRITICO.
        3. Nível por urgência do LLM: mapeado via LLM_URGENCY_TO_LEVEL.
        4. Nível final = o MAIOR entre os dois critérios acima — qualquer
           sinal de gravidade (estatístico ou linguístico) é suficiente
           para elevar o alerta, nunca para reduzi-lo (princípio de
           segurança: na dúvida, alerta mais, não menos).
    """
    n_anomaly = sum(1 for m in modalities if m.has_anomaly)

    if n_anomaly == 0:
        level_by_count = AlertLevel.NORMAL
    elif n_anomaly == 1:
        level_by_count = AlertLevel.ATENCAO
    else:
        level_by_count = AlertLevel.CRITICO

    level_by_llm = LLM_URGENCY_TO_LEVEL.get(llm_urgency, AlertLevel.NORMAL)

    final_level = _max_level(level_by_count, level_by_llm)

    reasoning_parts = [
        f"{n_anomaly} de {len(modalities)} modalidade(s) com anomalia "
        f"detectada (nível por contagem: {level_by_count.value})."
    ]
    if level_by_llm != AlertLevel.NORMAL:
        reasoning_parts.append(
            f"Análise de linguagem indicou urgência '{llm_urgency}' "
            f"(nível por LLM: {level_by_llm.value})."
        )
    if final_level != level_by_count:
        reasoning_parts.append(
            "Nível final elevado pela análise de linguagem, "
            "acima do que a contagem de modalidades indicaria isoladamente."
        )

    return AlertDecision(
        level=final_level,
        n_modalities_with_anomaly=n_anomaly,
        modalities=modalities,
        llm_urgency=llm_urgency,
        reasoning=" ".join(reasoning_parts),
    )


if __name__ == "__main__":
    print("=== Teste 1: paciente estável (nenhuma anomalia) ===")
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", False),
            ModalityStatus("postura", False),
            ModalityStatus("fala", False),
        ],
        llm_urgency="baixo",
    )
    print(f"Nível: {decision.level.value}")
    print(f"Justificativa: {decision.reasoning}\n")

    print("=== Teste 2: uma modalidade com anomalia isolada ===")
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", False),
            ModalityStatus("postura", True, "assimetria detectada"),
            ModalityStatus("fala", False),
        ],
        llm_urgency="baixo",
    )
    print(f"Nível: {decision.level.value}")
    print(f"Justificativa: {decision.reasoning}\n")

    print("=== Teste 3: duas modalidades simultâneas (crítico por contagem) ===")
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", True, "bradicardia"),
            ModalityStatus("postura", True, "hiperextensão"),
            ModalityStatus("fala", False),
        ],
        llm_urgency="medio",
    )
    print(f"Nível: {decision.level.value}")
    print(f"Justificativa: {decision.reasoning}\n")

    print("=== Teste 4: apenas LLM eleva o alerta (urgência alta isolada) ===")
    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", False),
            ModalityStatus("postura", False),
            ModalityStatus("fala", False),
        ],
        llm_urgency="alto",
    )
    print(f"Nível: {decision.level.value}")
    print(f"Justificativa: {decision.reasoning}")