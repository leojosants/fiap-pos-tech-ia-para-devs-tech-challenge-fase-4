"""
synthetic_audio.py
===================
Gera áudio sintético de triagem médica (paciente narrando sintomas) usando
TTS local via pyttsx3 — sem internet, sem custo, sem dados enviados a terceiros.

Cenário clínico simulado:
    Paciente responde a 12 frases curtas de triagem (sintomas + estado geral).
    Anomalias de fala são injetadas alterando a velocidade do motor de TTS
    e inserindo pausas extras entre palavras — simulando:

      - Fala lentificada/pausada: possível confusão mental, fadiga extrema,
        ou sintoma neurológico (ex.: disartria pós-AVC).
      - Fala acelerada: possível agitação, ansiedade, taquilalia.

Requisito de ambiente:
    pyttsx3 depende do motor de voz do sistema operacional (SAPI5 no
    Windows). Em ambientes sem motor de TTS disponível (ex.: containers
    Linux sem espeak), a geração de voz real falha — nesse caso, use
    generate_fallback_tone_audio() para obter áudio de teste (tons, sem
    fala real) que ainda permite validar o pipeline de métricas e detecção,
    mas SEM transcrição válida pelo Whisper.

Saída:
    data/raw/audio_segments/frase_NN.wav  — um arquivo por frase
    data/raw/audio_metadata.json          — gabarito (frase, tipo, anomalia)
"""

import json
import wave
import struct
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np


# ── Frases de triagem médica ──────────────────────────────────────────────────
# Cada frase tem um "tipo" (normal ou tipo de anomalia) usado como gabarito.
TRIAGE_PHRASES: list[dict] = [
    {"text": "Estou me sentindo bem hoje",                  "anomaly": "normal"},
    {"text": "Minha pressão está controlada",                "anomaly": "normal"},
    {"text": "Não sinto nenhuma dor",                        "anomaly": "normal"},
    {"text": "Dormi bem essa noite",                         "anomaly": "normal"},
    {"text": "Estou sentindo dor no peito",                  "anomaly": "fala_lenta"},
    {"text": "Minha visão está embaçada",                    "anomaly": "fala_lenta"},
    {"text": "Sinto uma fraqueza no braço",                  "anomaly": "fala_lenta"},
    {"text": "Já tomei a medicação prescrita",                "anomaly": "normal"},
    {"text": "Estou muito ansioso e agitado",                 "anomaly": "fala_rapida"},
    {"text": "Não consigo parar de tremer",                   "anomaly": "fala_rapida"},
    {"text": "Preciso de ajuda agora mesmo",                  "anomaly": "fala_rapida"},
    {"text": "Estou calmo e tranquilo agora",                 "anomaly": "normal"},
]

# Configurações do motor TTS por tipo de anomalia
# rate: palavras por minuto (padrão pyttsx3 ~200); pause_ms: pausa extra entre palavras
TTS_PROFILES = {
    "normal":      {"rate": 180, "pause_ms": 0},
    "fala_lenta":  {"rate": 90,  "pause_ms": 350},   # metade da velocidade + pausas longas
    "fala_rapida": {"rate": 320, "pause_ms": 0},      # quase o dobro da velocidade normal
}

SAMPLE_RATE = 22050   # Hz — padrão suficiente para fala, compatível com Whisper


@dataclass
class AudioSegmentInfo:
    """Metadados de um segmento de áudio gerado — funciona como gabarito."""
    segment_id: int
    text: str
    anomaly_type: str          # "normal" | "fala_lenta" | "fala_rapida"
    is_injected_anomaly: int   # 0 ou 1 — coluna de gabarito (padrão do projeto)
    file_path: str
    tts_rate: int
    tts_pause_ms: int


def _generate_with_pyttsx3(
    text: str,
    output_path: Path,
    rate: int,
    pause_ms: int = 0,
) -> bool:
    """
    Gera áudio de fala real usando pyttsx3 (motor de TTS do sistema).

    Parâmetros:
        text      : texto a ser falado
        output_path: caminho do arquivo .wav de saída
        rate      : velocidade da fala (palavras por minuto)
        pause_ms  : pausa extra entre palavras, em milissegundos
                    (simulada inserindo pontuação de pausa no texto)

    Retorna True se a geração teve sucesso, False caso o motor de TTS
    não esteja disponível no ambiente (ex.: Linux sem espeak instalado).
    """
    try:
        import pyttsx3
    except ImportError:
        return False

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)

        # Simula pausas longas inserindo "..." entre palavras
        # (o mecanismo de pausa varia entre motores SAPI5/NSSpeech/espeak;
        # "..." é a forma mais portável de forçar uma pausa perceptível)
        if pause_ms > 0:
            words = text.split()
            text_with_pauses = "... ".join(words)
        else:
            text_with_pauses = text

        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text_with_pauses, str(output_path))
        engine.runAndWait()
        engine.stop()
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _generate_fallback_tone_audio(
    text: str,
    output_path: Path,
    rate: int,
    pause_ms: int = 0,
) -> None:
    """
    Gera áudio sintético de TOM (não é fala real) como alternativa quando
    o motor de TTS não está disponível no ambiente (ex.: Linux sem espeak).

    Útil para validar o pipeline de métricas/detecção de anomalias mesmo
    sem voz real — mas o Whisper NÃO conseguirá transcrever texto coerente
    a partir desse áudio. Use apenas para testes de infraestrutura.

    Estratégia: gera um "beep" por palavra, com duração proporcional a
    1/rate (palavras mais lentas = tons mais longos) e pausas entre tons
    proporcionais a pause_ms — preservando a MESMA assinatura temporal
    que a fala real teria, para fins de teste do detector de anomalias.
    """
    words = text.split()
    n_words = len(words)

    # Duração de cada "palavra" (tom) em segundos, baseada no rate (palavras/min)
    word_duration = 60.0 / rate if rate > 0 else 0.3
    pause_duration = pause_ms / 1000.0

    samples: list[float] = []
    freq = 220.0  # tom base (A3)

    for i, _ in enumerate(words):
        t = np.linspace(0, word_duration, int(SAMPLE_RATE * word_duration), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
        samples.extend(tone.tolist())

        # Pausa entre palavras (silêncio)
        gap_duration = 0.08 + pause_duration   # pequena pausa natural + pausa extra
        gap = [0.0] * int(SAMPLE_RATE * gap_duration)
        samples.extend(gap)

    audio_array = np.array(samples, dtype=np.float32)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def generate_synthetic_audio_dataset(
    output_dir: Path,
    metadata_path: Path,
    use_fallback: bool = False,
) -> list[AudioSegmentInfo]:
    """
    Gera o dataset completo de áudio sintético — um arquivo .wav por frase
    de triagem, mais um arquivo JSON com os metadados (gabarito).

    Parâmetros:
        output_dir   : diretório onde salvar os arquivos .wav
        metadata_path: caminho do arquivo .json de metadados
        use_fallback : se True, força o uso de tons sintéticos em vez de
                       TTS real (útil em ambientes sem motor de voz, como
                       containers Linux para fins de teste de pipeline)

    Retorna lista de AudioSegmentInfo (um por frase gerada).
    """
    segments: list[AudioSegmentInfo] = []
    used_fallback = use_fallback

    for idx, phrase in enumerate(TRIAGE_PHRASES):
        anomaly_type = phrase["anomaly"]
        profile = TTS_PROFILES[anomaly_type]
        file_path = output_dir / f"frase_{idx:02d}.wav"

        success = False
        if not use_fallback:
            success = _generate_with_pyttsx3(
                phrase["text"], file_path, profile["rate"], profile["pause_ms"]
            )
            if not success:
                used_fallback = True

        if use_fallback or not success:
            _generate_fallback_tone_audio(
                phrase["text"], file_path, profile["rate"], profile["pause_ms"]
            )

        segments.append(AudioSegmentInfo(
            segment_id=idx,
            text=phrase["text"],
            anomaly_type=anomaly_type,
            is_injected_anomaly=0 if anomaly_type == "normal" else 1,
            file_path=str(file_path),
            tts_rate=profile["rate"],
            tts_pause_ms=profile["pause_ms"],
        ))

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "used_fallback_tone_audio": used_fallback,
                "segments": [asdict(s) for s in segments],
            },
            f, ensure_ascii=False, indent=2,
        )

    return segments


if __name__ == "__main__":
    out_dir = Path("data/raw/audio_segments")
    meta_path = Path("data/raw/audio_metadata.json")

    print("Gerando dataset de áudio sintético de triagem médica...")
    segments = generate_synthetic_audio_dataset(out_dir, meta_path)

    n_normal = sum(1 for s in segments if s.anomaly_type == "normal")
    n_lenta  = sum(1 for s in segments if s.anomaly_type == "fala_lenta")
    n_rapida = sum(1 for s in segments if s.anomaly_type == "fala_rapida")

    print(f"\n=== DATASET GERADO ===")
    print(f"  Total de frases  : {len(segments)}")
    print(f"  Normais          : {n_normal}")
    print(f"  Fala lenta       : {n_lenta}")
    print(f"  Fala rápida      : {n_rapida}")
    print(f"  Arquivos em      : {out_dir}")
    print(f"  Metadados em     : {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta["used_fallback_tone_audio"]:
        print(
            "\n[AVISO] O motor de TTS do sistema não foi detectado — "
            "áudio gerado com tons sintéticos (sem fala real).\n"
            "         No Windows, isso normalmente não ocorre (SAPI5 nativo)."
        )