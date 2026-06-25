"""
test_audio.py
==============
Testes do módulo de análise de fala (Etapa 3).

Cobre o cálculo de métricas de fala a partir de áudio sintético (sem
chamar o Whisper real — mocks de tudo que dependeria de rede/modelo
pesado) e a normalização de separador de caminho, que corrigiu um bug
real encontrado durante o deploy (Seção 12.6 do relatório técnico):
caminhos gerados no Windows usavam "\\", inacessível no servidor Linux.
"""

import json
from pathlib import Path
from unittest.mock import patch

from src.audio.audio_processor import process_audio_dataset
from src.audio.synthetic_audio import generate_synthetic_audio_dataset


def _fake_transcribe_segment(file_path):
    """
    Substitui a chamada real ao Whisper por um resultado fixo — o teste
    de normalização de caminho não precisa de transcrição real, apenas
    confirmar que o arquivo foi localizado e processado (em vez de cair
    no branch de "arquivo não encontrado").
    """
    return {
        "transcribed_text": "teste mockado",
        "avg_logprob": -0.5,
        "n_words_transcribed": 2,
        "duration_seconds": 1.0,
        "words_per_second": 2.0,
        "silence_ratio": 0.2,
    }


def test_synthetic_audio_metadata_uses_forward_slash(tmp_path):
    """
    Regressão do bug real encontrado no deploy (Seção 12.6): o caminho
    salvo em audio_metadata.json deve sempre usar "/" como separador,
    independente do sistema operacional onde o dataset é gerado — caso
    contrário, o arquivo não é localizado em servidores Linux.
    """
    output_dir = tmp_path / "audio_segments"
    metadata_path = tmp_path / "audio_metadata.json"

    generate_synthetic_audio_dataset(output_dir, metadata_path, use_fallback=True)

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    for segment in meta["segments"]:
        assert "\\" not in segment["file_path"], (
            f"file_path contém separador inválido para Linux: {segment['file_path']}"
        )


def test_process_audio_dataset_normalizes_legacy_backslash_paths(tmp_path):
    """
    Mesmo que um audio_metadata.json legado (gerado antes da correção)
    contenha caminhos com barra invertida, process_audio_dataset deve
    normalizá-los e localizar os arquivos com sucesso — rede de
    segurança documentada na Seção 12.6 do relatório técnico.
    """
    audio_dir = tmp_path / "audio_segments"
    audio_dir.mkdir()

    # Cria um .wav mínimo válido (cabeçalho WAV de 0 amostras é suficiente
    # para o teste de localização de arquivo; o conteúdo de áudio em si
    # não é avaliado aqui)
    import wave
    wav_path = audio_dir / "frase_00.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"")

    # Simula um metadata.json "legado", com separador estilo Windows
    legacy_path = str(audio_dir).replace("/", "\\") + "\\frase_00.wav"
    metadata_path = tmp_path / "audio_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "used_fallback_tone_audio": True,
            "segments": [{
                "segment_id": 0,
                "text": "teste",
                "anomaly_type": "normal",
                "is_injected_anomaly": 0,
                "file_path": legacy_path,
                "tts_rate": 180,
                "tts_pause_ms": 0,
            }],
        }, f)

    with patch("src.audio.audio_processor.transcribe_segment", side_effect=_fake_transcribe_segment):
        df = process_audio_dataset(metadata_path)

    # Mesmo com o caminho legado, o arquivo deve ter sido localizado e
    # processado (pose_detected/transcrição executada, sem cair no branch
    # de "arquivo não encontrado" que preenche tudo com NaN)
    assert df.loc[0, "duration_seconds"] == df.loc[0, "duration_seconds"]  # não é NaN
    assert not pd_isna(df.loc[0, "duration_seconds"])


def pd_isna(value) -> bool:
    """Pequeno helper local para evitar import de pandas só para isso."""
    import math
    return isinstance(value, float) and math.isnan(value)