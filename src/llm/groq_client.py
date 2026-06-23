"""
groq_client.py
==============
Cliente centralizado para a Groq API — substituto local-friendly da
OpenAI API sugerida no desafio original (ver Seção 2 do relatório técnico).

A Groq API é compatível com o SDK oficial da OpenAI em termos de
estrutura de chamada (client.chat.completions.create), facilitando a
migração caso o projeto precise trocar de provedor no futuro.

Modelos utilizados (ver nota técnica — Seção do relatório sobre
deprecação de modelos):
    - openai/gpt-oss-20b  : tarefas estruturadas e simples (extração de
                              termos, classificação) — rápido e barato.
    - openai/gpt-oss-120b : tarefas que exigem raciocínio mais complexo
                              (sumarização cruzando múltiplas modalidades).

Nota sobre tokens de raciocínio:
    Os modelos da família gpt-oss são modelos de raciocínio — parte do
    `max_tokens` é consumida internamente em um campo "reasoning" antes
    da resposta final visível. Por isso, este projeto usa valores de
    max_tokens generosos (≥150) em todas as chamadas, validados
    empiricamente durante o desenvolvimento (ver relatório técnico).
"""

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq, APIError, APIConnectionError, RateLimitError


load_dotenv()

# ── Modelos configuráveis via .env ────────────────────────────────────────────
# Os nomes dos modelos NÃO são fixos no código — são lidos do .env para que
# uma futura depreciação de modelo pela Groq (já ocorrida uma vez durante o
# desenvolvimento deste projeto, ver relatório técnico) exija apenas a
# alteração de uma linha de configuração, sem tocar em código-fonte.
#
# GROQ_MODEL_FAST  : modelo para tarefas estruturadas/simples (extração,
#                    classificação) — prioriza velocidade e custo.
# GROQ_MODEL_SMART : modelo para tarefas que exigem raciocínio mais complexo
#                    (sumarização cruzando múltiplas fontes de evidência).
#
# Valores padrão (fallback) caso as variáveis não estejam definidas no .env —
# refletem os modelos vigentes na Groq API no momento do desenvolvimento.
MODEL_FAST = os.environ.get("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
MODEL_SMART = os.environ.get("GROQ_MODEL_SMART", "openai/gpt-oss-120b")

DEFAULT_MAX_TOKENS = 250
DEFAULT_TEMPERATURE = 0.3   # baixa — queremos respostas consistentes/factuais
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


@dataclass
class GroqResponse:
    """Resultado padronizado de uma chamada à Groq API."""
    success: bool
    content: str
    model: str
    error_message: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


def _get_client() -> Groq:
    """
    Cria o cliente Groq a partir da chave configurada em .env.

    Levanta erro claro e acionável se a variável de ambiente não existir,
    em vez de deixar o SDK lançar um erro genérico de autenticação.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY não encontrada nas variáveis de ambiente. "
            "Verifique se o arquivo .env existe na raiz do projeto e contém "
            "GROQ_API_KEY=sua_chave_aqui (veja .env.example)."
        )
    return Groq(api_key=api_key)


def call_groq(
    messages: list[dict],
    model: str = MODEL_FAST,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> GroqResponse:
    """
    Wrapper robusto para chamadas à Groq API, com retry automático em
    caso de rate limit ou erro transitório de conexão.

    Parâmetros:
        messages   : lista de mensagens no formato OpenAI
                     (ex.: [{"role": "system", "content": "..."},
                            {"role": "user", "content": "..."}])
        model      : MODEL_FAST ou MODEL_SMART (ver constantes do módulo)
        max_tokens : limite de tokens gerados (inclui tokens de raciocínio
                     para modelos gpt-oss — usar valores generosos)
        temperature: controla aleatoriedade da resposta (0 = determinístico)

    Retorna GroqResponse com success=False e error_message preenchida em
    caso de falha — nunca levanta exceção, para que o pipeline de fusão
    multimodal (Etapa 5) possa degradar graciosamente sem travar todo o
    sistema caso a API esteja indisponível.
    """
    client = _get_client()

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            choice = response.choices[0]
            usage = response.usage

            reasoning_tokens = 0
            if usage and usage.completion_tokens_details:
                reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0

            return GroqResponse(
                success=True,
                content=(choice.message.content or "").strip(),
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                reasoning_tokens=reasoning_tokens,
            )

        except RateLimitError as e:
            last_error = f"Rate limit excedido: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

        except APIConnectionError as e:
            last_error = f"Erro de conexão com a Groq API: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

        except APIError as e:
            # Erros de API (modelo inválido, payload malformado) não se
            # beneficiam de retry — falham imediatamente com mensagem clara.
            last_error = f"Erro da Groq API: {e}"
            break

        except Exception as e:
            last_error = f"Erro inesperado: {e}"
            break

    return GroqResponse(
        success=False,
        content="",
        model=model,
        error_message=last_error,
    )


if __name__ == "__main__":
    print("Testando conexão com a Groq API...")
    print(f"  Modelo rápido : {MODEL_FAST}")
    print(f"  Modelo robusto: {MODEL_SMART}")
    print()

    for model in [MODEL_FAST, MODEL_SMART]:
        result = call_groq(
            messages=[{"role": "user", "content": "Responda apenas a palavra OK"}],
            model=model,
            max_tokens=100,
        )
        if result.success:
            print(f"[{model}] OK -> '{result.content}' "
                  f"(prompt={result.prompt_tokens}, "
                  f"completion={result.completion_tokens}, "
                  f"reasoning={result.reasoning_tokens})")
        else:
            print(f"[{model}] FALHOU -> {result.error_message}")