"""
prompts.py
==========
Prompts centralizados usados pelos módulos de LLM do projeto.

Centralizar os prompts em um único arquivo facilita auditoria, versionamento
e ajuste fino — qualquer alteração no comportamento do LLM fica rastreável
em um só lugar, em vez de espalhada entre vários módulos.

Convenção: cada prompt é uma função que recebe os dados variáveis e
retorna a lista de `messages` pronta para a Groq API (formato OpenAI).
"""


def build_text_analysis_prompt(transcribed_text: str) -> list[dict]:
    """
    Prompt para extração de termos críticos, sentimento e nível de
    urgência a partir de uma frase transcrita de triagem médica.

    Pede resposta em formato JSON estrito, para parsing confiável
    posterior (ver text_analyzer.py).
    """
    system_prompt = (
        "Você é um assistente de triagem médica. Sua tarefa é analisar uma "
        "frase dita por um paciente e extrair informações estruturadas. "
        "Responda APENAS com um objeto JSON válido, sem nenhum texto "
        "adicional antes ou depois, no seguinte formato exato:\n"
        '{"termos_criticos": ["termo1", "termo2"], '
        '"sentimento": "positivo|neutro|negativo", '
        '"nivel_urgencia": "baixo|medio|alto", '
        '"justificativa": "breve explicação em até 15 palavras"}\n\n'
        "Regras:\n"
        "- termos_criticos: liste apenas sintomas, partes do corpo ou "
        "condições mencionadas (vazio [] se não houver nenhum).\n"
        "- sentimento: o tom emocional geral da frase.\n"
        "- nivel_urgencia: 'alto' para sintomas potencialmente graves "
        "(dor no peito, falta de ar, fraqueza súbita), 'medio' para "
        "queixas moderadas, 'baixo' para frases neutras ou positivas."
    )

    user_prompt = f'Frase do paciente: "{transcribed_text}"'

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_clinical_summary_prompt(
    vitals_summary: str,
    posture_summary: str,
    speech_summary: str,
) -> list[dict]:
    """
    Prompt para gerar um resumo clínico executivo cruzando as três
    modalidades monitoradas (sinais vitais, postura, fala).

    Parâmetros:
        vitals_summary  : texto resumindo achados da Etapa 1 (vitais)
        posture_summary : texto resumindo achados da Etapa 2 (vídeo/postura)
        speech_summary  : texto resumindo achados da Etapa 3 (áudio/fala)

    Retorna messages para a Groq API. Usa MODEL_SMART (gpt-oss-120b) por
    exigir raciocínio sobre múltiplas fontes de evidência simultaneamente.
    """
    system_prompt = (
        "Você é um assistente clínico que auxilia a equipe médica a "
        "interpretar dados de monitoramento multimodal de um paciente "
        "(sinais vitais, análise postural por vídeo, e análise de fala). "
        "Gere um resumo executivo claro, objetivo e factual, em português, "
        "com no máximo 150 palavras, destacando:\n"
        "1. Se há indícios de risco clínico que mereçam atenção prioritária;\n"
        "2. Quais modalidades apresentaram anomalias e sua possível relação;\n"
        "3. Uma recomendação objetiva de próximo passo para a equipe médica.\n\n"
        "Baseie-se EXCLUSIVAMENTE nos dados fornecidos. Não invente sintomas "
        "ou achados que não estejam explicitamente nos dados de entrada."
    )

    user_prompt = (
        f"DADOS DE SINAIS VITAIS:\n{vitals_summary}\n\n"
        f"DADOS DE ANÁLISE POSTURAL (VÍDEO):\n{posture_summary}\n\n"
        f"DADOS DE ANÁLISE DE FALA (ÁUDIO):\n{speech_summary}\n\n"
        "Gere o resumo executivo conforme as instruções."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]