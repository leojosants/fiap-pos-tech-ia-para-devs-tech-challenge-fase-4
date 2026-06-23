"""
summarizer.py
=============
Gera um resumo clínico executivo cruzando os achados das três
modalidades monitoradas pelo sistema (sinais vitais, postura, fala),
usando a Groq API (modelo gpt-oss-120b — maior capacidade de raciocínio,
necessária para sintetizar múltiplas fontes de evidência).

Esta é a peça central da camada de IA generativa: ao contrário dos
detectores estatísticos das Etapas 1-3 (que respondem "há uma anomalia
neste ponto?"), o summarizer responde "o que esses achados significam
em conjunto, e o que a equipe médica deveria fazer a respeito?" —
papel equivalente à sumarização de laudos mencionada no desafio original.

Entrada: resumos textuais agregados de cada modalidade (não os dados
brutos completos — manter o prompt conciso reduz custo e latência,
e evita exceder limites de contexto).

Saída: texto em português, até 150 palavras, com recomendação objetiva.
"""

from dataclasses import dataclass

import pandas as pd

from src.llm.groq_client import call_groq, MODEL_SMART
from src.llm.prompts import build_clinical_summary_prompt


@dataclass
class ClinicalSummaryResult:
    """Resultado da sumarização clínica."""
    success: bool
    summary_text: str
    error_message: str = ""


def build_vitals_summary_text(df: pd.DataFrame, metrics: dict) -> str:
    """
    Constrói um resumo textual conciso dos achados de sinais vitais
    (Etapa 1), a partir do DataFrame de detecção e das métricas de
    avaliação já calculadas por src.vitals.detector.

    Parâmetros:
        df     : DataFrame com colunas de sinais vitais e anomaly_final
        metrics: dicionário de métricas (saída de evaluate_detection())
    """
    n_total = len(df)
    n_anomaly = int(df.get("anomaly_final", pd.Series([0] * n_total)).sum())

    anomaly_rows = df[df.get("anomaly_final", 0) == 1] if "anomaly_final" in df.columns else pd.DataFrame()

    lines = [
        f"Total de amostras monitoradas: {n_total} (1 amostra/segundo).",
        f"Anomalias detectadas: {n_anomaly}.",
    ]

    if not anomaly_rows.empty:
        for col, label in [
            ("heart_rate_bpm", "frequência cardíaca"),
            ("spo2_pct", "SpO2"),
            ("bp_systolic_mmhg", "pressão sistólica"),
        ]:
            if col in anomaly_rows.columns:
                vmin, vmax = anomaly_rows[col].min(), anomaly_rows[col].max()
                lines.append(f"  Faixa de {label} nos pontos anômalos: {vmin:.0f}–{vmax:.0f}.")

    combinado = metrics.get("combinado", {})
    if combinado:
        lines.append(
            f"Desempenho do detector combinado: precisão={combinado.get('precisao')}, "
            f"recall={combinado.get('recall')}."
        )

    return "\n".join(lines)


def build_posture_summary_text(df: pd.DataFrame, metrics: dict) -> str:
    """
    Constrói um resumo textual conciso dos achados de análise postural
    (Etapa 2), a partir do DataFrame de detecção e métricas.
    """
    n_total = len(df)
    n_anomaly = int(df.get("anomaly_final", pd.Series([0] * n_total)).sum())

    anomaly_types_detected = []
    if "rule_asymmetry" in df.columns and df["rule_asymmetry"].sum() > 0:
        anomaly_types_detected.append("assimetria bilateral entre braços")
    if "rule_trunk_collapse" in df.columns and df["rule_trunk_collapse"].sum() > 0:
        anomaly_types_detected.append("inclinação anômala de tronco")
    if "anomaly_zscore_velocity" in df.columns and df["anomaly_zscore_velocity"].sum() > 0:
        anomaly_types_detected.append("aceleração articular brusca (possível hiperextensão)")

    lines = [
        f"Total de frames de vídeo analisados: {n_total} (sessão de fisioterapia).",
        f"Frames com anomalia postural detectada: {n_anomaly}.",
        f"Padrões identificados: {', '.join(anomaly_types_detected) if anomaly_types_detected else 'nenhum'}.",
    ]

    combinado = metrics.get("combinado", {})
    if combinado:
        lines.append(
            f"Desempenho do detector combinado: precisão={combinado.get('precisao')}, "
            f"recall={combinado.get('recall')}."
        )

    return "\n".join(lines)


def generate_clinical_summary(
    vitals_summary: str,
    posture_summary: str,
    speech_summary: str,
) -> ClinicalSummaryResult:
    """
    Gera o resumo clínico executivo final, cruzando as três modalidades.

    Parâmetros:
        vitals_summary  : texto de build_vitals_summary_text()
        posture_summary : texto de build_posture_summary_text()
        speech_summary  : texto de text_analyzer.build_speech_summary_text()

    Retorna ClinicalSummaryResult com o texto gerado ou mensagem de erro.
    """
    messages = build_clinical_summary_prompt(
        vitals_summary, posture_summary, speech_summary
    )

    # max_tokens generoso: cobre o orçamento de "reasoning" interno do
    # modelo gpt-oss-120b somado a uma resposta de até ~150 palavras em
    # português (a resposta em português consome mais tokens por palavra
    # que o inglês). O modelo de 120B observou-se consumir uma fração de
    # reasoning maior que o de 20B (Etapa 4), exigindo um limite mais alto.
    response = call_groq(messages, model=MODEL_SMART, max_tokens=1200)

    if not response.success:
        return ClinicalSummaryResult(
            success=False,
            summary_text="",
            error_message=response.error_message,
        )

    if not response.content:
        # Mesmo modo de falha silenciosa documentado na Etapa 4: chamada
        # com sucesso, mas conteúdo vazio por esgotamento do orçamento de
        # tokens em reasoning. Uma tentativa extra com limite ainda maior.
        response = call_groq(messages, model=MODEL_SMART, max_tokens=1600)

    if not response.success or not response.content:
        return ClinicalSummaryResult(
            success=False,
            summary_text="",
            error_message=response.error_message or
                "Resposta vazia da API mesmo após nova tentativa com mais tokens.",
        )

    # Verifica se a resposta parece truncada no meio de uma frase (não
    # termina em pontuação final) — sintoma de truncamento por limite de
    # tokens mesmo quando o conteúdo não veio vazio. Loga como aviso para
    # facilitar diagnóstico, sem impedir o uso do texto parcial.
    final_chars = ('.', '!', '?', '"', "'")
    if response.content and response.content[-1] not in final_chars:
        import warnings
        warnings.warn(
            "Resumo clínico pode estar truncado (não termina em pontuação "
            "final). Considere aumentar max_tokens se isso persistir."
        )

    return ClinicalSummaryResult(
        success=True,
        summary_text=response.content,
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
    from src.llm.text_analyzer import analyze_text_dataset, build_speech_summary_text

    print("=== Coletando achados das três modalidades ===\n")

    print("[1/3] Sinais vitais...")
    df_vitals = generate_vitals_dataframe()
    df_vitals, vitals_metrics = run_vitals_pipeline(df_vitals)
    vitals_text = build_vitals_summary_text(df_vitals, vitals_metrics)

    print("[2/3] Análise postural (vídeo)...")
    frames, _ = generate_synthetic_video()
    df_pose = process_video(frames, anomaly_frame_indices=ALL_ANOMALY_FRAMES)
    df_pose, posture_metrics = run_video_detection_pipeline(df_pose)
    posture_text = build_posture_summary_text(df_pose, posture_metrics)

    print("[3/3] Análise de fala (áudio)...")
    meta_path = Path("data/raw/audio_metadata.json")
    df_audio = process_audio_dataset(meta_path)
    df_audio, audio_metrics = run_audio_detection_pipeline(df_audio)
    df_audio = analyze_text_dataset(df_audio)
    speech_text = build_speech_summary_text(df_audio)

    print("\n=== RESUMOS POR MODALIDADE ===")
    print("\n--- Sinais Vitais ---")
    print(vitals_text)
    print("\n--- Postura ---")
    print(posture_text)
    print("\n--- Fala ---")
    print(speech_text)

    print("\n\nGerando resumo clínico executivo com Groq API (gpt-oss-120b)...")
    result = generate_clinical_summary(vitals_text, posture_text, speech_text)

    print("\n=== RESUMO CLÍNICO EXECUTIVO ===")
    if result.success:
        print(result.summary_text)
    else:
        print(f"[ERRO] {result.error_message}")