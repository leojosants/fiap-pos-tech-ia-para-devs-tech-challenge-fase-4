"""
alert_dispatcher.py
====================
"Envia" o alerta consolidado pelo motor de fusão (fusion_engine.py) —
nesta versão do projeto, sem servidor de e-mail/SMS real, mas com:

    1. Log estruturado no console/arquivo de log (rastreável durante
       desenvolvimento e demonstração);
    2. Histórico persistente em JSON (data/processed/alert_history.json),
       permitindo que a Etapa 6 (interface Streamlit) exiba o histórico
       de alertas de cada "sessão" de monitoramento sem precisar
       reprocessar os dados brutos.

Esta separação entre "decidir o alerta" (fusion_engine.py + alert_rules.py)
e "despachar o alerta" (este módulo) segue o princípio de responsabilidade
única: trocar o canal de notificação no futuro (e-mail real, webhook,
SMS) exigiria alterar apenas este arquivo, sem tocar na lógica de decisão.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.alerts.alert_rules import AlertDecision, AlertLevel
from src.alerts.fusion_engine import FusionResult


# ── Configuração de logging ───────────────────────────────────────────────────
logger = logging.getLogger("medwatch.alerts")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)


DEFAULT_HISTORY_PATH = Path("data/processed/alert_history.json")

_LOG_LEVEL_MAP = {
    AlertLevel.NORMAL: logging.INFO,
    AlertLevel.ATENCAO: logging.WARNING,
    AlertLevel.CRITICO: logging.ERROR,
}


def _decision_to_dict(decision: AlertDecision) -> dict:
    """Converte AlertDecision (incluindo Enum e dataclasses aninhadas) em
    um dicionário serializável em JSON."""
    return {
        "level": decision.level.value,
        "n_modalities_with_anomaly": decision.n_modalities_with_anomaly,
        "llm_urgency": decision.llm_urgency,
        "reasoning": decision.reasoning,
        "modalities": [asdict(m) for m in decision.modalities],
    }


def log_alert(decision: AlertDecision, patient_id: str = "paciente-demo-01") -> None:
    """
    Registra o alerta no log estruturado, com nível de severidade do
    próprio logger (INFO/WARNING/ERROR) espelhando o nível clínico —
    facilita filtrar logs por gravidade em ferramentas de observabilidade.
    """
    log_level = _LOG_LEVEL_MAP[decision.level]
    message = (
        f"[{patient_id}] Nível: {decision.level.value.upper()} | "
        f"{decision.n_modalities_with_anomaly} modalidade(s) com anomalia | "
        f"Urgência LLM: {decision.llm_urgency} | {decision.reasoning}"
    )
    logger.log(log_level, message)


def append_to_history(
    decision: AlertDecision,
    patient_id: str = "paciente-demo-01",
    history_path: Path = DEFAULT_HISTORY_PATH,
    clinical_summary: str = "",
) -> dict:
    """
    Adiciona o alerta ao histórico persistente em JSON.

    Cada entrada do histórico contém timestamp, paciente, decisão completa
    e (se disponível) o resumo clínico gerado pelo LLM — permitindo que a
    interface Streamlit (Etapa 6) reconstrua a visão histórica sem
    reprocessar nada.

    Parâmetros:
        decision        : AlertDecision retornada por alert_rules.decide_alert_level
        patient_id       : identificador do paciente/sessão (sintético neste projeto)
        history_path     : caminho do arquivo JSON de histórico
        clinical_summary : texto do resumo clínico (Etapa 4), se disponível

    Retorna a entrada adicionada (dict), para conveniência do chamador.
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {"alerts": []}

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_id": patient_id,
        "decision": _decision_to_dict(decision),
        "clinical_summary": clinical_summary,
    }
    history["alerts"].append(entry)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return entry


def dispatch_alert(
    fusion_result: FusionResult,
    patient_id: str = "paciente-demo-01",
    history_path: Path = DEFAULT_HISTORY_PATH,
) -> dict:
    """
    Função de conveniência: registra o log E adiciona ao histórico em
    uma única chamada — o ponto de entrada típico após o motor de fusão.

    Parâmetros:
        fusion_result: saída de fusion_engine.run_fusion_pipeline()
        patient_id    : identificador do paciente/sessão
        history_path  : caminho do arquivo JSON de histórico

    Retorna a entrada de histórico adicionada.
    """
    log_alert(fusion_result.alert_decision, patient_id=patient_id)
    entry = append_to_history(
        fusion_result.alert_decision,
        patient_id=patient_id,
        history_path=history_path,
        clinical_summary=fusion_result.clinical_summary,
    )
    return entry


def load_alert_history(history_path: Path = DEFAULT_HISTORY_PATH) -> list[dict]:
    """
    Carrega o histórico completo de alertas já registrados — usado pela
    interface Streamlit (Etapa 6) para exibir a linha do tempo de alertas.

    Retorna lista vazia se o arquivo de histórico ainda não existir.
    """
    if not history_path.exists():
        return []

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    return history.get("alerts", [])


if __name__ == "__main__":
    from src.alerts.alert_rules import ModalityStatus, decide_alert_level
    from src.alerts.fusion_engine import FusionResult

    print("Testando dispatcher com alerta simulado (sem rodar pipelines completos)...\n")

    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", True, "2 anomalias de frequência cardíaca."),
            ModalityStatus("postura", True, "7 frames com assimetria detectada."),
            ModalityStatus("fala", False, "Padrão de fala normal."),
        ],
        llm_urgency="alto",
    )

    fusion_result = FusionResult(
        alert_decision=decision,
        vitals_summary="2 anomalias de frequência cardíaca.",
        posture_summary="7 frames com assimetria detectada.",
        speech_summary="Padrão de fala normal.",
        clinical_summary="[Resumo de teste — não chamou a API real.]",
    )

    test_history_path = Path("data/processed/alert_history_test.json")
    entry = dispatch_alert(fusion_result, patient_id="paciente-teste", history_path=test_history_path)

    print("\nEntrada registrada no histórico:")
    print(json.dumps(entry, ensure_ascii=False, indent=2))

    print(f"\nHistórico completo em {test_history_path}:")
    history = load_alert_history(test_history_path)
    print(f"Total de alertas no histórico: {len(history)}")