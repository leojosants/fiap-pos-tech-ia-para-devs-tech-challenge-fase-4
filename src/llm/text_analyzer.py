"""
text_analyzer.py
================
Analisa as frases transcritas (saída da Etapa 3 — Whisper) usando a
Groq API (modelo gpt-oss-20b) para extrair:

    - termos críticos mencionados (sintomas, partes do corpo)
    - sentimento geral da frase
    - nível de urgência clínica (baixo/médio/alto)

Esta é a substituição local-friendly do Azure Text Analytics / AWS
Comprehend sugeridos no desafio original (ver Seção 2 do relatório
técnico) — usamos um LLM via Groq API em vez de um serviço de NLP
gerenciado em nuvem proprietária.

Entrada esperada: DataFrame produzido por
    src.audio.audio_processor.process_audio_dataset()
(colunas: segment_id, transcribed_text, anomaly_type, is_injected_anomaly, ...)

Saída: o mesmo DataFrame enriquecido com colunas de análise de texto.
"""

import json
import re

import pandas as pd

from src.llm.groq_client import call_groq, MODEL_FAST
from src.llm.prompts import build_text_analysis_prompt


def _parse_json_response(raw_content: str) -> dict:
    """
    Extrai e parseia o JSON da resposta do LLM, tolerando pequenas
    variações comuns (texto extra antes/depois, markdown fences).

    Retorna dicionário com valores padrão em caso de falha de parsing,
    para que o pipeline não trave por uma única resposta malformada.
    """
    default = {
        "termos_criticos": [],
        "sentimento": "neutro",
        "nivel_urgencia": "baixo",
        "justificativa": "",
    }

    if not raw_content:
        return default

    # Remove markdown fences (```json ... ```) se presentes
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip())

    # Tenta extrair o primeiro objeto JSON válido, mesmo se houver texto extra
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return default

    try:
        parsed = json.loads(match.group(0))
        return {
            "termos_criticos": parsed.get("termos_criticos", []),
            "sentimento": parsed.get("sentimento", "neutro"),
            "nivel_urgencia": parsed.get("nivel_urgencia", "baixo"),
            "justificativa": parsed.get("justificativa", ""),
        }
    except (json.JSONDecodeError, AttributeError):
        return default


def analyze_text_segment(transcribed_text: str) -> dict:
    """
    Analisa uma única frase transcrita via Groq API.

    Retorna dicionário com termos_criticos, sentimento, nivel_urgencia,
    justificativa, e metadados de execução (sucesso, modelo usado).
    """
    if not transcribed_text or not transcribed_text.strip():
        return {
            "termos_criticos": [],
            "sentimento": "neutro",
            "nivel_urgencia": "baixo",
            "justificativa": "",
            "llm_success": False,
            "llm_error": "Texto vazio — análise não realizada.",
        }

    messages = build_text_analysis_prompt(transcribed_text)

    # max_tokens generoso: modelos gpt-oss consomem uma parcela variável
    # (observada entre ~30 e ~130 tokens neste projeto) em raciocínio
    # interno ("reasoning") antes de produzir a resposta JSON visível.
    # Um limite baixo pode esgotar o orçamento todo em reasoning e
    # retornar conteúdo vazio mesmo com a chamada tecnicamente bem-sucedida
    # — comportamento observado e documentado no relatório técnico.
    response = call_groq(messages, model=MODEL_FAST, max_tokens=400)

    if not response.success:
        return {
            "termos_criticos": [],
            "sentimento": "neutro",
            "nivel_urgencia": "baixo",
            "justificativa": "",
            "llm_success": False,
            "llm_error": response.error_message,
        }

    if not response.content:
        # Chamada teve sucesso (sem erro de API), mas a resposta visível
        # ficou vazia — sintoma de orçamento de tokens insuficiente.
        # Uma única tentativa extra com max_tokens ainda maior resolve
        # a grande maioria dos casos sem mascarar falhas reais.
        response = call_groq(messages, model=MODEL_FAST, max_tokens=600)

    if not response.success or not response.content:
        return {
            "termos_criticos": [],
            "sentimento": "neutro",
            "nivel_urgencia": "baixo",
            "justificativa": "",
            "llm_success": False,
            "llm_error": response.error_message or
                "Resposta vazia da API mesmo após nova tentativa com mais tokens.",
        }

    parsed = _parse_json_response(response.content)
    parsed["llm_success"] = True
    parsed["llm_error"] = ""
    return parsed


def analyze_text_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica analyze_text_segment() a cada linha do DataFrame de áudio
    (saída do audio_processor.py), enriquecendo com as colunas de análise.

    Parâmetros:
        df: DataFrame com coluna 'transcribed_text' (saída da Etapa 3)

    Retorna o mesmo DataFrame com colunas adicionais:
        termos_criticos, sentimento, nivel_urgencia, justificativa,
        llm_success, llm_error
    """
    df = df.copy()

    results = [
        analyze_text_segment(text)
        for text in df["transcribed_text"]
    ]

    df["termos_criticos"] = [r["termos_criticos"] for r in results]
    df["sentimento"] = [r["sentimento"] for r in results]
    df["nivel_urgencia"] = [r["nivel_urgencia"] for r in results]
    df["justificativa"] = [r["justificativa"] for r in results]
    df["llm_success"] = [r["llm_success"] for r in results]
    df["llm_error"] = [r["llm_error"] for r in results]

    return df


def build_speech_summary_text(df: pd.DataFrame) -> str:
    """
    Constrói um resumo textual conciso dos achados da análise de fala,
    para ser usado como entrada do summarizer.py (Etapa 4b).

    Agrega: quantas frases tiveram anomalia de velocidade detectada,
    quais termos críticos foram mencionados, e a distribuição de
    níveis de urgência segundo o LLM.
    """
    n_total = len(df)
    n_anomaly = int(df.get("anomaly_final", pd.Series([0] * n_total)).sum())

    all_terms = []
    for terms in df.get("termos_criticos", []):
        if isinstance(terms, list):
            all_terms.extend(terms)
    unique_terms = sorted(set(all_terms))

    urgency_counts = (
        df["nivel_urgencia"].value_counts().to_dict()
        if "nivel_urgencia" in df.columns else {}
    )

    lines = [
        f"Total de frases analisadas: {n_total}",
        f"Frases com anomalia de fala detectada (velocidade/pausas): {n_anomaly}",
        f"Termos críticos mencionados: {', '.join(unique_terms) if unique_terms else 'nenhum'}",
        f"Distribuição de urgência (segundo análise de linguagem): {urgency_counts}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path
    from src.audio.audio_processor import process_audio_dataset

    meta_path = Path("data/raw/audio_metadata.json")

    if not meta_path.exists():
        print(f"Metadados não encontrados em {meta_path}.")
        print("Execute primeiro as Etapas 3: synthetic_audio e audio_processor.")
    else:
        print("Carregando transcrições da Etapa 3...")
        df_audio = process_audio_dataset(meta_path)

        print("Analisando texto com Groq API (modelo gpt-oss-20b)...")
        df_analyzed = analyze_text_dataset(df_audio)

        print("\n=== RESULTADO DA ANÁLISE DE TEXTO ===")
        cols = ["segment_id", "transcribed_text", "sentimento",
                "nivel_urgencia", "termos_criticos", "llm_success"]
        print(df_analyzed[cols].to_string(index=False))

        n_failed = (~df_analyzed["llm_success"]).sum()
        if n_failed > 0:
            print(f"\n[AVISO] {n_failed} segmento(s) falharam na análise LLM:")
            for _, row in df_analyzed[~df_analyzed["llm_success"]].iterrows():
                print(f"  Segmento {row['segment_id']}: {row['llm_error']}")

        print("\n=== RESUMO PARA O SUMMARIZER ===")
        print(build_speech_summary_text(df_analyzed))