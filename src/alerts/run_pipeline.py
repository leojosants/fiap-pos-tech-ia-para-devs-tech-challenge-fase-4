"""
run_pipeline.py  (src/alerts/)
================================
Ponto de entrada único da Etapa 5: executa o pipeline completo de fusão
multimodal com dados REAIS (não os dados de exemplo usados nos testes
isolados de cada módulo) e produz os três artefatos finais:

    1. Decisão de alerta consolidada (fusion_engine.py)
    2. Dashboard visual com os dados desta execução (plotter.py)
    3. Log + histórico persistido (alert_dispatcher.py)

Por que este arquivo existe:
    Cada módulo da Etapa 5 (alert_rules.py, fusion_engine.py,
    alert_dispatcher.py, plotter.py) tem um bloco `if __name__ ==
    "__main__"` próprio, usado durante o desenvolvimento para testar
    cada peça isoladamente com dados de exemplo fixos — útil para
    validar a lógica sem depender de toda a cadeia anterior. Esses
    blocos de teste NÃO substituem a execução ponta a ponta: rodar
    `python -m src.alerts.plotter` isoladamente gera um dashboard com
    dados de exemplo, não com os resultados desta sessão. Este módulo
    é o único ponto de entrada que conecta as Etapas 1-5 com dados
    reais de uma única execução consistente.
"""

from pathlib import Path

from src.vitals.generator import generate_vitals_dataframe
from src.vitals.detector import run_detection_pipeline as run_vitals_pipeline

from src.video.synthetic_video import generate_synthetic_video, ALL_ANOMALY_FRAMES
from src.video.video_processor import process_video
from src.video.anomaly_detector import run_video_detection_pipeline

from src.audio.audio_processor import process_audio_dataset
from src.audio.anomaly_detector import run_audio_detection_pipeline
from src.llm.text_analyzer import analyze_text_dataset, build_speech_summary_text
from src.llm.summarizer import build_vitals_summary_text, build_posture_summary_text

from src.alerts.fusion_engine import run_fusion_pipeline
from src.alerts.alert_dispatcher import dispatch_alert
from src.alerts.plotter import plot_fusion_dashboard


def run_full_pipeline(
    patient_id: str = "paciente-demo-01",
    generate_clinical_summary: bool = True,
    dashboard_path: Path = Path("data/processed/fusion_dashboard.png"),
    history_path: Path = Path("data/processed/alert_history.json"),
):
    """
    Executa o pipeline completo das Etapas 1-5 em uma única chamada
    consistente, e produz os artefatos finais com os dados desta execução.

    Parâmetros:
        patient_id                : identificador do paciente/sessão
        generate_clinical_summary : se True, chama a Groq API (gpt-oss-120b)
                                     para o resumo executivo — mais lento
        dashboard_path             : caminho do PNG do dashboard
        history_path                : caminho do JSON de histórico de alertas
    """
    print("=== ETAPA 1: Sinais vitais ===")
    df_vitals = generate_vitals_dataframe()
    df_vitals, vitals_metrics = run_vitals_pipeline(df_vitals)
    vitals_text = build_vitals_summary_text(df_vitals, vitals_metrics)
    print(f"  {vitals_text.splitlines()[1]}")

    print("\n=== ETAPA 2: Análise postural (vídeo) ===")
    frames, _ = generate_synthetic_video()
    df_posture = process_video(frames, anomaly_frame_indices=ALL_ANOMALY_FRAMES)
    df_posture, posture_metrics = run_video_detection_pipeline(df_posture)
    posture_text = build_posture_summary_text(df_posture, posture_metrics)
    print(f"  {posture_text.splitlines()[1]}")

    print("\n=== ETAPA 3: Análise de fala (áudio) ===")
    meta_path = Path("data/raw/audio_metadata.json")
    df_speech = process_audio_dataset(meta_path)
    df_speech, speech_metrics = run_audio_detection_pipeline(df_speech)

    print("\n=== ETAPA 4: Análise de texto via LLM ===")
    df_speech_analyzed = analyze_text_dataset(df_speech)
    speech_text = build_speech_summary_text(df_speech_analyzed)
    print(f"  {speech_text.splitlines()[1]}")

    print("\n=== ETAPA 5: Motor de fusão multimodal ===")
    fusion_result = run_fusion_pipeline(
        df_vitals=df_vitals,
        df_posture=df_posture,
        df_speech=df_speech,
        df_speech_analyzed=df_speech_analyzed,
        generate_clinical_summary=generate_clinical_summary,
        vitals_summary_text=vitals_text,
        posture_summary_text=posture_text,
        speech_summary_text=speech_text,
    )

    decision = fusion_result.alert_decision
    print(f"\n  NÍVEL DE ALERTA: {decision.level.value.upper()}")
    print(f"  Justificativa: {decision.reasoning}")
    for m in decision.modalities:
        flag = "⚠️ " if m.has_anomaly else "✓ "
        print(f"  {flag}{m.name}: {m.detail}")

    if generate_clinical_summary and fusion_result.clinical_summary:
        print(f"\n  Resumo clínico:\n  {fusion_result.clinical_summary}")

    print("\n=== Gerando dashboard e despachando alerta ===")
    plot_fusion_dashboard(
        decision,
        clinical_summary=fusion_result.clinical_summary,
        save_path=dashboard_path,
    )
    entry = dispatch_alert(fusion_result, patient_id=patient_id, history_path=history_path)
    print(f"Alerta despachado e registrado no histórico ({history_path}).")

    return fusion_result, entry


if __name__ == "__main__":
    run_full_pipeline()