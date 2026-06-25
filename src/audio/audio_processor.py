"""
audio_processor.py
===================
Transcreve os segmentos de áudio com Whisper (modelo local, sem internet)
e extrai métricas de fala relevantes para triagem clínica:

    - duração total do áudio
    - velocidade de fala (palavras por segundo, estimada via transcrição)
    - número de palavras transcritas
    - confiança média da transcrição (probabilidade do Whisper)
    - presença de silêncios longos (proxy de pausas/hesitação)

Modelo Whisper:
    Usamos o modelo "base" — equilíbrio entre velocidade e precisão para
    frases curtas em português, sem exigir GPU. Roda 100% localmente,
    sem enviar áudio a serviços externos (alinhado à decisão arquitetural
    de privacidade de dados clínicos, ver Seção 2 do relatório técnico).

Saída:
    DataFrame com uma linha por segmento de áudio, no mesmo padrão das
    Etapas 1 e 2: métricas + coluna de gabarito is_injected_anomaly.
"""

import os
import json
import wave
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd


# ── Configuração do modelo Whisper ────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium | large
WHISPER_LANGUAGE = "pt"        # força português — evita autodetecção incorreta

_whisper_model = None   # cache do modelo carregado (evita reload por segmento)


def _ensure_ffmpeg_on_path() -> None:
    """
    Garante que o executável ffmpeg esteja acessível no PATH do processo.

    O Whisper depende do binário ffmpeg (linha de comando) para decodificar
    áudio, mas não o instala como dependência Python. Em vez de exigir uma
    instalação manual no sistema (Chocolatey/Scoop/download), usamos o
    pacote imageio-ffmpeg — que empacota um executável ffmpeg via PyPI —
    e o expomos no PATH apenas para a duração deste processo Python.

    Isso preserva a filosofia 100% local do projeto: nenhuma instalação
    manual de software fora do ecossistema gerenciado pelo uv.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        original_dir = os.path.dirname(ffmpeg_path)

        # O Whisper chama especificamente o comando "ffmpeg" — em alguns
        # ambientes o executável do imageio-ffmpeg tem nome diferente
        # (ex.: ffmpeg-win64-v4.2.2.exe), exigindo uma cópia com nome
        # compatível. Essa cópia é feita em um diretório temporário
        # gravável (tempfile.gettempdir()), e NÃO dentro da pasta de
        # instalação do pacote (site-packages/imageio_ffmpeg/binaries/),
        # pois em ambientes de deploy com permissões restritas (ex.:
        # Streamlit Community Cloud, usuário "adminuser" sem permissão
        # de escrita em site-packages) a cópia no local original falha
        # com PermissionError — comportamento observado e corrigido
        # durante o deploy deste projeto (ver relatório técnico, nota
        # sobre a Etapa 8).
        import tempfile
        import shutil

        target_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        target_dir = os.path.join(tempfile.gettempdir(), "medwatch_ffmpeg")
        target_path = os.path.join(target_dir, target_name)

        if not os.path.exists(target_path):
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(ffmpeg_path, target_path)
            os.chmod(target_path, 0o755)  # garante permissão de execução

        # Coloca o diretório temporário (com o binário já nomeado
        # corretamente) na FRENTE do PATH, para garantir que o comando
        # "ffmpeg" resolva para ele antes de qualquer outra ocorrência.
        if target_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = target_dir + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        # imageio-ffmpeg não instalado — assume que ffmpeg já está no PATH
        # do sistema (ex.: instalado manualmente, via Chocolatey/Scoop, ou
        # via packages.txt no Streamlit Community Cloud)
        pass


def _get_whisper_model():
    """
    Carrega o modelo Whisper uma única vez (lazy loading) e reaproveita
    entre chamadas — carregar o modelo é a operação mais custosa do pipeline.
    """
    global _whisper_model
    if _whisper_model is None:
        _ensure_ffmpeg_on_path()
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


def _get_audio_duration(file_path: Path) -> float:
    """Retorna a duração do arquivo .wav em segundos, via cabeçalho WAV."""
    with contextlib.closing(wave.open(str(file_path), "r")) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate > 0 else 0.0


def _detect_silence_ratio(file_path: Path, silence_threshold: float = 0.02) -> float:
    """
    Calcula a proporção do áudio que está em silêncio (energia abaixo do
    limiar), como proxy de pausas/hesitação na fala.

    Retorna valor entre 0.0 (sem silêncio) e 1.0 (áudio inteiro em silêncio).
    """
    with contextlib.closing(wave.open(str(file_path), "r")) as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) == 0:
        return 1.0

    # Energia em janelas de 50ms
    window_size = max(1, int(0.05 * wf.getframerate()))
    n_windows = len(audio) // window_size
    if n_windows == 0:
        return 0.0

    energies = [
        np.sqrt(np.mean(audio[i * window_size:(i + 1) * window_size] ** 2))
        for i in range(n_windows)
    ]
    silent_windows = sum(1 for e in energies if e < silence_threshold)
    return silent_windows / n_windows


def transcribe_segment(file_path: Path) -> dict:
    """
    Transcreve um único segmento de áudio com Whisper e extrai métricas.

    Retorna dicionário com:
        transcribed_text   : texto transcrito
        avg_logprob         : log-probabilidade média (proxy de confiança)
        n_words_transcribed : número de palavras no texto transcrito
        duration_seconds     : duração do áudio em segundos
        words_per_second     : velocidade de fala estimada
        silence_ratio         : proporção de silêncio no áudio
    """
    model = _get_whisper_model()

    result = model.transcribe(
        str(file_path),
        language=WHISPER_LANGUAGE,
        fp16=False,   # CPU-only, sem GPU — evita warning/erro de precisão
    )

    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    avg_logprob = (
        float(np.mean([seg.get("avg_logprob", 0.0) for seg in segments]))
        if segments else float("nan")
    )

    duration = _get_audio_duration(file_path)
    n_words = len(text.split()) if text else 0
    words_per_second = (n_words / duration) if duration > 0 else 0.0
    silence_ratio = _detect_silence_ratio(file_path)

    return {
        "transcribed_text": text,
        "avg_logprob": round(avg_logprob, 4),
        "n_words_transcribed": n_words,
        "duration_seconds": round(duration, 3),
        "words_per_second": round(words_per_second, 4),
        "silence_ratio": round(silence_ratio, 4),
    }


def process_audio_dataset(
    metadata_path: Path,
) -> pd.DataFrame:
    """
    Processa todos os segmentos descritos no arquivo de metadados,
    transcrevendo cada um e extraindo métricas de fala.

    Parâmetros:
        metadata_path: caminho do audio_metadata.json gerado por
                        synthetic_audio.generate_synthetic_audio_dataset()

    Retorna DataFrame com uma linha por segmento — mesmo padrão das
    Etapas 1 e 2 (métricas + is_injected_anomaly).
    """
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    rows: list[dict] = []

    for seg in meta["segments"]:
        # Normaliza separadores de caminho para "/" antes de criar o Path —
        # rede de segurança contra arquivos audio_metadata.json antigos,
        # gerados no Windows com str(Path) (que usa "\"), antes da correção
        # em synthetic_audio.py (.as_posix()). Em Linux, uma barra invertida
        # não separa diretórios — é um caractere literal do nome do arquivo,
        # causando falha silenciosa de "arquivo não encontrado" (ver nota
        # de deploy no relatório técnico, Etapa 8).
        normalized_path = seg["file_path"].replace("\\", "/")
        file_path = Path(normalized_path)

        row = {
            "segment_id": seg["segment_id"],
            "original_text": seg["text"],
            "anomaly_type": seg["anomaly_type"],
            "is_injected_anomaly": seg["is_injected_anomaly"],
            "tts_rate": seg["tts_rate"],
            "tts_pause_ms": seg["tts_pause_ms"],
        }

        if file_path.exists():
            metrics = transcribe_segment(file_path)
            row.update(metrics)
        else:
            row.update({
                "transcribed_text": "",
                "avg_logprob": float("nan"),
                "n_words_transcribed": 0,
                "duration_seconds": float("nan"),
                "words_per_second": float("nan"),
                "silence_ratio": float("nan"),
            })

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    meta_path = Path("data/raw/audio_metadata.json")

    if not meta_path.exists():
        print(f"Metadados não encontrados em {meta_path}.")
        print("Execute primeiro: uv run python -m src.audio.synthetic_audio")
    else:
        print("Transcrevendo segmentos com Whisper (modelo 'base')...")
        print("(primeira execução pode demorar — baixa o modelo localmente)")

        df = process_audio_dataset(meta_path)

        print(f"\n=== RESULTADO DA TRANSCRIÇÃO ===")
        print(f"  Segmentos processados : {len(df)}")
        print(f"  Velocidade média (palavras/s): {df['words_per_second'].mean():.2f}")
        print(f"  Silêncio médio          : {df['silence_ratio'].mean():.2%}")

        cols = ["segment_id", "anomaly_type", "transcribed_text",
                 "words_per_second", "silence_ratio", "is_injected_anomaly"]
        print("\nResultado por segmento:")
        print(df[cols].to_string(index=False))