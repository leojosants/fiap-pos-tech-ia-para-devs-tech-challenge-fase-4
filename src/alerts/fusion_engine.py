"""
fusion_engine.py
================
Motor de fusão multimodal: orquestra os pipelines já implementados nas
Etapas 1-4 (sinais vitais, postura/vídeo, fala/áudio, análise de texto via
LLM) e produz uma única decisão de alerta consolidada por execução.

Este módulo representa o requisito central do desafio: "realizar a
análise e fusão de diferentes tipos de dados médicos (texto, áudio,
vídeo)" e "gerar alertas automáticos para a equipe médica com base nas
anomalias detectadas" (ver Seção 1 do relatório técnico).

Fluxo de execução:
    1. Roda (ou recebe já processados) os DataFrames de vitais, postura
       e fala, cada um já passado pelo seu detector de anomalias.
    2. Roda a análise de texto via LLM sobre as frases transcritas.
    3. Agrega cada modalidade em um ModalityStatus (anomalia sim/não).
    4. Aplica alert_rules.decide_alert_level() para o nível final.
    5. Opcionalmente, gera o resumo clínico executivo (Etapa 4) — mais
       custoso (chamada ao modelo gpt-oss-120b), por isso é um passo
       separado e opcional dentro do pipeline completo.
"""

from dataclasses import dataclass

import pandas as pd

from src.alerts.alert_rules import ModalityStatus, AlertDecision, decide_alert_level


@dataclass
class FusionResult:
    """Resultado completo de uma execução do motor de fusão."""
    alert_decision: AlertDecision
    vitals_summary: str
    posture_summary: str
    speech_summary: str
    clinical_summary: str = ""   # preenchido apenas se generate_summary=True


def _summarize_vitals_status(df: pd.DataFrame) -> ModalityStatus:
    """
    Resume o status da modalidade de sinais vitais a partir do DataFrame
    já processado pelo pipeline da Etapa 1 (coluna anomaly_final).
    """
    has_anomaly = bool(df.get("anomaly_final", pd.Series([0])).sum() > 0)
    n_anomaly = int(df.get("anomaly_final", pd.Series([0])).sum())
    detail = f"{n_anomaly} amostra(s) com anomalia em sinais vitais." if has_anomaly else "Sinais vitais dentro do padrão."
    return ModalityStatus(name="vitais", has_anomaly=has_anomaly, detail=detail)


def _summarize_posture_status(df: pd.DataFrame) -> ModalityStatus:
    """
    Resume o status da modalidade de postura a partir do DataFrame já
    processado pelo pipeline da Etapa 2 (coluna anomaly_final).
    """
    has_anomaly = bool(df.get("anomaly_final", pd.Series([0])).sum() > 0)
    n_anomaly = int(df.get("anomaly_final", pd.Series([0])).sum())
    detail = f"{n_anomaly} frame(s) com anomalia postural detectada." if has_anomaly else "Padrão de movimento dentro do esperado."
    return ModalityStatus(name="postura", has_anomaly=has_anomaly, detail=detail)


def _summarize_speech_status(df: pd.DataFrame) -> ModalityStatus:
    """
    Resume o status da modalidade de fala a partir do DataFrame já
    processado pelo pipeline da Etapa 3 (coluna anomaly_final).
    """
    has_anomaly = bool(df.get("anomaly_final", pd.Series([0])).sum() > 0)
    n_anomaly = int(df.get("anomaly_final", pd.Series([0])).sum())
    detail = f"{n_anomaly} segmento(s) de fala com anomalia de velocidade/pausa." if has_anomaly else "Padrão de fala dentro do esperado."
    return ModalityStatus(name="fala", has_anomaly=has_anomaly, detail=detail)


def _get_max_llm_urgency(df_speech_analyzed: pd.DataFrame) -> str:
    """
    Extrai o maior nível de urgência apontado pela análise de linguagem
    (Etapa 4) entre todas as frases do paciente nesta sessão.

    Retorna "alto" se qualquer frase foi classificada como tal, senão
    "medio" se houver ao menos uma, senão "baixo".
    """
    if "nivel_urgencia" not in df_speech_analyzed.columns:
        return "baixo"

    urgencies = set(df_speech_analyzed["nivel_urgencia"].dropna())
    if "alto" in urgencies:
        return "alto"
    if "medio" in urgencies:
        return "medio"
    return "baixo"


def run_fusion_pipeline(
    df_vitals: pd.DataFrame,
    df_posture: pd.DataFrame,
    df_speech: pd.DataFrame,
    df_speech_analyzed: pd.DataFrame,
    generate_clinical_summary: bool = False,
    vitals_summary_text: str = "",
    posture_summary_text: str = "",
    speech_summary_text: str = "",
) -> FusionResult:
    """
    Executa a fusão multimodal a partir dos DataFrames já processados
    pelos pipelines das Etapas 1-3 e pela análise de texto da Etapa 4.

    Parâmetros:
        df_vitals           : saída de src.vitals.detector.run_detection_pipeline
        df_posture           : saída de src.video.anomaly_detector.run_video_detection_pipeline
        df_speech            : saída de src.audio.anomaly_detector.run_audio_detection_pipeline
        df_speech_analyzed   : df_speech enriquecido por
                                src.llm.text_analyzer.analyze_text_dataset
        generate_clinical_summary: se True, chama a Groq API (gpt-oss-120b)
                                para gerar o resumo executivo (Etapa 4) —
                                passo mais custoso, opcional neste pipeline
        vitals_summary_text, posture_summary_text, speech_summary_text:
                                resumos textuais pré-calculados (Etapa 4);
                                se não fornecidos, são reconstruídos aqui
                                de forma simplificada

    Retorna FusionResult com a decisão de alerta consolidada.
    """
    vitals_status = _summarize_vitals_status(df_vitals)
    posture_status = _summarize_posture_status(df_posture)
    speech_status = _summarize_speech_status(df_speech)

    llm_urgency = _get_max_llm_urgency(df_speech_analyzed)

    decision = decide_alert_level(
        modalities=[vitals_status, posture_status, speech_status],
        llm_urgency=llm_urgency,
    )

    clinical_summary = ""
    if generate_clinical_summary:
        from src.llm.summarizer import generate_clinical_summary as llm_summarize

        result = llm_summarize(
            vitals_summary_text or vitals_status.detail,
            posture_summary_text or posture_status.detail,
            speech_summary_text or speech_status.detail,
        )
        clinical_summary = result.summary_text if result.success else (
            f"[Resumo indisponível: {result.error_message}]"
        )

    return FusionResult(
        alert_decision=decision,
        vitals_summary=vitals_summary_text or vitals_status.detail,
        posture_summary=posture_summary_text or posture_status.detail,
        speech_summary=speech_summary_text or speech_status.detail,
        clinical_summary=clinical_summary,
    )


if __name__ == "__main__":
    from pathlib import Path

    from src.vitals.generator import generate_vitals_dataframe
    from src.vitals.detector import run_detection_pipeline as run_vitals_pipeline

    from src.video.synthetic_video import generate_synthetic_video, ALL_ANOMALY_FRAMES
    from src.video.video_processor import process_video
    from src.video.anomaly_detector import run_video_detection_pipeline

    from src.audio.audio_processor import process_audio_dataset
    from src.audio.anomaly_detector import run_audio_detection_pipeline
    from src.llm.text_analyzer import analyze_text_dataset
    from src.llm.summarizer import build_vitals_summary_text, build_posture_summary_text
    from src.llm.text_analyzer import build_speech_summary_text

    print("=== Executando pipelines das Etapas 1-4 ===\n")

    print("[1/4] Sinais vitais...")
    df_vitals = generate_vitals_dataframe()
    df_vitals, vitals_metrics = run_vitals_pipeline(df_vitals)

    print("[2/4] Análise postural (vídeo)...")
    frames, _ = generate_synthetic_video()
    df_posture = process_video(frames, anomaly_frame_indices=ALL_ANOMALY_FRAMES)
    df_posture, posture_metrics = run_video_detection_pipeline(df_posture)

    print("[3/4] Análise de fala (áudio)...")
    meta_path = Path("data/raw/audio_metadata.json")
    df_speech = process_audio_dataset(meta_path)
    df_speech, speech_metrics = run_audio_detection_pipeline(df_speech)

    print("[4/4] Análise de texto via LLM...")
    df_speech_analyzed = analyze_text_dataset(df_speech)

    print("\n=== Executando motor de fusão ===\n")

    result = run_fusion_pipeline(
        df_vitals=df_vitals,
        df_posture=df_posture,
        df_speech=df_speech,
        df_speech_analyzed=df_speech_analyzed,
        generate_clinical_summary=False,  # True consome chamada à Groq API
        vitals_summary_text=build_vitals_summary_text(df_vitals, vitals_metrics),
        posture_summary_text=build_posture_summary_text(df_posture, posture_metrics),
        speech_summary_text=build_speech_summary_text(df_speech_analyzed),
    )

    print(f"NÍVEL DE ALERTA: {result.alert_decision.level.value.upper()}")
    print(f"Justificativa: {result.alert_decision.reasoning}\n")
    print("--- Detalhe por modalidade ---")
    for m in result.alert_decision.modalities:
        flag = "⚠️ " if m.has_anomaly else "✓ "
        print(f"{flag}{m.name}: {m.detail}")