"""
main.py
=======
Ponto de entrada da interface Streamlit do MedWatch — sistema de
monitoramento multimodal de pacientes (Tech Challenge Fase 4).

Fluxo da interface:
    1. Tela inicial com visão geral do sistema e botão único
       "Executar Monitoramento Completo".
    2. Ao clicar, executa as Etapas 1-5 em sequência, com barra de
       progresso mostrando a etapa atual (sinais vitais → vídeo →
       áudio → análise de texto → fusão e alerta).
    3. Resultado populado em 4 abas: Vitais, Vídeo, Áudio, Alertas.

Os resultados de cada execução são mantidos em st.session_state,
permitindo navegar entre as abas sem reprocessar os dados.
"""

from pathlib import Path

import streamlit as st

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

from src.dashboard.pages_vitals import render_vitals_tab
from src.dashboard.pages_video import render_video_tab
from src.dashboard.pages_audio import render_audio_tab
from src.dashboard.pages_alerts import render_alerts_tab


st.set_page_config(
    page_title="MedWatch — Monitoramento Multimodal",
    page_icon="🏥",
    layout="wide",
)


AUDIO_METADATA_PATH = Path("data/raw/audio_metadata.json")
VIDEO_PATH = Path("data/processed/synthetic_pose.mp4")


def run_monitoring_pipeline(generate_clinical_summary: bool = True) -> None:
    """
    Executa o pipeline completo (Etapas 1-5) com barra de progresso,
    armazenando todos os resultados em st.session_state.

    Parâmetros:
        generate_clinical_summary: se True, chama a Groq API
                                    (gpt-oss-120b) para o resumo executivo
    """
    progress = st.progress(0, text="Iniciando monitoramento...")

    progress.progress(10, text="Processando sinais vitais...")
    df_vitals = generate_vitals_dataframe()
    df_vitals, vitals_metrics = run_vitals_pipeline(df_vitals)
    vitals_text = build_vitals_summary_text(df_vitals, vitals_metrics)

    progress.progress(30, text="Analisando vídeo (postura)...")
    frames, _ = generate_synthetic_video(output_video_path=VIDEO_PATH)
    df_posture = process_video(frames, anomaly_frame_indices=ALL_ANOMALY_FRAMES)
    df_posture, posture_metrics = run_video_detection_pipeline(df_posture)
    posture_text = build_posture_summary_text(df_posture, posture_metrics)

    progress.progress(50, text="Transcrevendo e analisando áudio (Whisper)...")
    # Os arquivos de áudio sintético (data/raw/audio_segments/*.wav) são
    # gerados localmente via pyttsx3 (motor de TTS do sistema operacional,
    # SAPI5 no Windows) e COMMITADOS no repositório — não regenerados em
    # tempo de execução no servidor de deploy (Streamlit Community Cloud,
    # Linux/Debian), onde o SAPI5 não está disponível. Esta verificação
    # só dispara a geração via TTS quando os arquivos realmente não
    # existem (ex.: primeira execução em ambiente de desenvolvimento local
    # ainda sem os áudios gerados) — ver Seção de Deploy do relatório
    # técnico para detalhes desta decisão.
    if not AUDIO_METADATA_PATH.exists():
        try:
            from src.audio.synthetic_audio import generate_synthetic_audio_dataset
            generate_synthetic_audio_dataset(
                Path("data/raw/audio_segments"), AUDIO_METADATA_PATH
            )
        except Exception as e:
            st.error(
                "Não foi possível gerar o áudio sintético neste ambiente "
                f"(motor de TTS indisponível: {e}). Os arquivos de áudio "
                "deveriam estar pré-gerados e commitados no repositório."
            )
            st.stop()
    df_speech = process_audio_dataset(AUDIO_METADATA_PATH)
    df_speech, speech_metrics = run_audio_detection_pipeline(df_speech)

    progress.progress(70, text="Analisando texto e urgência clínica (Groq LLM)...")
    df_speech_analyzed = analyze_text_dataset(df_speech)
    speech_text = build_speech_summary_text(df_speech_analyzed)

    # O pipeline de detecção (Etapa 3) não propaga a coluna file_path —
    # ela existe apenas no JSON de metadados original. Reconectamos aqui
    # para que a aba de áudio (pages_audio.py) consiga localizar e
    # reproduzir cada segmento de áudio via st.audio().
    import json
    with open(AUDIO_METADATA_PATH, encoding="utf-8") as f:
        audio_meta = json.load(f)
    path_map = {s["segment_id"]: s["file_path"] for s in audio_meta["segments"]}
    df_speech_analyzed["file_path"] = df_speech_analyzed["segment_id"].map(path_map)

    progress.progress(85, text="Executando motor de fusão multimodal...")
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

    progress.progress(95, text="Registrando alerta...")
    dispatch_alert(fusion_result, patient_id="paciente-demo-01")

    progress.progress(100, text="Concluído!")
    progress.empty()

    st.session_state["df_vitals"] = df_vitals
    st.session_state["vitals_metrics"] = vitals_metrics
    st.session_state["df_posture"] = df_posture
    st.session_state["posture_metrics"] = posture_metrics
    st.session_state["df_speech"] = df_speech_analyzed
    st.session_state["speech_metrics"] = speech_metrics
    st.session_state["fusion_result"] = fusion_result
    st.session_state["pipeline_executed"] = True


def render_debug_secrets() -> None:
    """
    BLOCO TEMPORÁRIO DE DIAGNÓSTICO — remover após resolver o problema
    de GROQ_API_KEY não encontrada no deploy do Streamlit Community Cloud.
    Não expõe a chave real, apenas tamanho e primeiros caracteres.
    """
    with st.expander("🔍 Diagnóstico temporário de Secrets (remover depois)"):
        st.write("**1. st.secrets — chaves disponíveis:**")
        try:
            keys = list(st.secrets.keys())
            st.write(f"Chaves: {keys}")
            if "GROQ_API_KEY" in st.secrets:
                val = st.secrets["GROQ_API_KEY"]
                st.success(f"GROQ_API_KEY em st.secrets! Tamanho={len(val)}, início={val[:6]}...")
            else:
                st.error("GROQ_API_KEY NÃO está em st.secrets.")
        except Exception as e:
            st.error(f"Erro ao acessar st.secrets: {type(e).__name__}: {e}")

        st.write("**2. os.environ — variável GROQ_API_KEY:**")
        env_val = os.environ.get("GROQ_API_KEY")
        if env_val:
            st.success(f"GROQ_API_KEY em os.environ! Tamanho={len(env_val)}, início={env_val[:6]}...")
        else:
            st.error("GROQ_API_KEY NÃO está em os.environ.")

        st.write("**3. Variáveis de ambiente contendo 'GROQ':**")
        groq_vars = {k: v for k, v in os.environ.items() if "GROQ" in k.upper()}
        if groq_vars:
            for k, v in groq_vars.items():
                st.write(f"- {k}: tamanho={len(v)}, início={v[:6]}...")
        else:
            st.warning("Nenhuma variável de ambiente contendo 'GROQ' encontrada.")

        st.write("**4. Arquivo .streamlit/secrets.toml:**")
        secrets_path = ".streamlit/secrets.toml"
        st.write(f"Caminho absoluto: {os.path.abspath(secrets_path)}")
        st.write(f"Existe: {os.path.exists(secrets_path)}")


def render_overview() -> None:
    """Renderiza a tela inicial com visão geral do sistema."""
    st.title("🏥 MedWatch — Monitoramento Multimodal de Pacientes")
    st.markdown(
        "Sistema de monitoramento contínuo que analisa **sinais vitais**, "
        "**postura (vídeo)** e **fala (áudio)** de um paciente simulado, "
        "cruzando os achados com um modelo de linguagem (Groq API) para "
        "gerar um **alerta consolidado** para a equipe médica."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📊 Sinais Vitais**")
        st.caption("Frequência cardíaca, SpO2 e pressão arterial, com detecção via z-score + Isolation Forest.")
    with col2:
        st.markdown("**🎥 Postura (Vídeo)**")
        st.caption("Ângulos articulares em sessão de fisioterapia simulada, com detecção de assimetria e hiperextensão.")
    with col3:
        st.markdown("**🎙️ Fala (Áudio)**")
        st.caption("Transcrição via Whisper local + classificação de urgência via LLM (Groq).")

    st.markdown("---")

    st.info(
        "Todo o processamento de vídeo, áudio e sinais vitais ocorre "
        "**100% localmente**. A Groq API é usada exclusivamente para "
        "interpretação de linguagem natural sobre o texto já transcrito."
    )


def main() -> None:
    render_debug_secrets()
    render_overview()

    if st.button("▶️ Executar Monitoramento Completo", type="primary", use_container_width=False):
        run_monitoring_pipeline(generate_clinical_summary=True)

    if not st.session_state.get("pipeline_executed", False):
        st.caption(
            "Clique no botão acima para gerar dados sintéticos, processar "
            "as três modalidades e produzir o alerta consolidado."
        )
        return

    st.markdown("---")
    tab_alerts, tab_vitals, tab_video, tab_audio = st.tabs(
        ["🚨 Alertas", "📊 Vitais", "🎥 Vídeo", "🎙️ Áudio"]
    )

    with tab_alerts:
        render_alerts_tab(st.session_state["fusion_result"])

    with tab_vitals:
        render_vitals_tab(
            st.session_state["df_vitals"], st.session_state["vitals_metrics"]
        )

    with tab_video:
        render_video_tab(
            st.session_state["df_posture"],
            st.session_state["posture_metrics"],
            VIDEO_PATH,
        )

    with tab_audio:
        render_audio_tab(
            st.session_state["df_speech"], st.session_state["speech_metrics"]
        )


if __name__ == "__main__":
    main()