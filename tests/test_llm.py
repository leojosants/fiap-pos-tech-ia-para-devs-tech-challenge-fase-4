"""
test_llm.py
============
Testes do módulo de integração com a Groq API (Etapa 4).

Cobre o parser de JSON tolerante (núcleo determinístico, testável sem
rede) e o fluxo de análise de texto com a chamada à Groq API mockada —
nenhum teste aqui consome créditos de API ou depende de conectividade.
"""

from unittest.mock import patch

from src.llm.text_analyzer import _parse_json_response, analyze_text_segment
from src.llm.groq_client import GroqResponse


def test_parse_json_response_handles_clean_json():
    """Caso feliz: resposta do LLM é um JSON puro, sem ruído ao redor."""
    raw = (
        '{"termos_criticos": ["dor no peito"], "sentimento": "negativo", '
        '"nivel_urgencia": "alto", "justificativa": "sintoma grave"}'
    )
    result = _parse_json_response(raw)

    assert result["termos_criticos"] == ["dor no peito"]
    assert result["nivel_urgencia"] == "alto"


def test_parse_json_response_strips_markdown_fences_and_extra_text():
    """
    LLMs ocasionalmente envolvem a resposta em blocos de código markdown
    ou adicionam texto explicativo antes/depois do JSON — o parser deve
    extrair o objeto JSON corretamente em ambos os casos.
    """
    raw_with_fences = (
        '```json\n{"termos_criticos": [], "sentimento": "positivo", '
        '"nivel_urgencia": "baixo", "justificativa": "ok"}\n```'
    )
    raw_with_preamble = (
        'Aqui está a análise: {"termos_criticos": ["tremor"], '
        '"sentimento": "negativo", "nivel_urgencia": "medio", '
        '"justificativa": "ansiedade"} Espero ter ajudado!'
    )

    result1 = _parse_json_response(raw_with_fences)
    result2 = _parse_json_response(raw_with_preamble)

    assert result1["sentimento"] == "positivo"
    assert result2["termos_criticos"] == ["tremor"]


def test_parse_json_response_falls_back_safely_on_malformed_input():
    """
    Entradas sem JSON válido (string vazia, texto sem estrutura) não
    devem levantar exceção — o parser retorna valores neutros padrão,
    permitindo que o pipeline de fusão (Etapa 5) continue funcionando
    mesmo se uma única chamada de LLM retornar algo inesperado.
    """
    assert _parse_json_response("")["nivel_urgencia"] == "baixo"
    assert _parse_json_response("resposta sem nenhum json")["termos_criticos"] == []


def test_analyze_text_segment_uses_mocked_groq_response():
    """
    Verifica a integração entre analyze_text_segment e o parser, com a
    chamada real à Groq API substituída por um mock — nenhuma rede ou
    custo de API envolvido neste teste.
    """
    mocked_response = GroqResponse(
        success=True,
        content='{"termos_criticos": ["dor no peito"], "sentimento": "negativo", '
                '"nivel_urgencia": "alto", "justificativa": "sintoma grave"}',
        model="openai/gpt-oss-20b",
    )

    with patch("src.llm.text_analyzer.call_groq", return_value=mocked_response):
        result = analyze_text_segment("Estou sentindo dor no peito")

    assert result["llm_success"] is True
    assert result["nivel_urgencia"] == "alto"
    assert "dor no peito" in result["termos_criticos"]


def test_analyze_text_segment_handles_empty_text_without_calling_api():
    """
    Texto vazio (ex.: transcrição falhou) não deve gerar uma chamada à
    API — economiza custo e evita erro de prompt vazio.
    """
    with patch("src.llm.text_analyzer.call_groq") as mock_call:
        result = analyze_text_segment("")

    mock_call.assert_not_called()
    assert result["llm_success"] is False