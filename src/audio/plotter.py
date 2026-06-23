"""
plotter.py  (src/audio/)
=========================
Gera visualizações do pipeline de áudio:

    - Painel de métricas de fala (velocidade, silêncio) por segmento,
      com marcadores de anomalia — mesmo estilo das Etapas 1 e 2.
    - Waveform anotado de segmentos representativos (normal vs anômalo).

Sem display gráfico — tudo salvo em arquivo.

Saídas:
    data/processed/audio_metrics_panel.png
    data/processed/audio_waveforms/waveform_segment_NN.png
"""

import wave
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Paleta (consistente com Etapas 1 e 2) ─────────────────────────────────────
COLORS = {
    "normal":     "#2196F3",
    "anomaly":    "#F44336",
    "injected":   "#FF9800",
    "background": "#F8F9FA",
    "grid":       "#E0E0E0",
}


def _read_wav_samples(file_path: Path) -> tuple[np.ndarray, int]:
    """Lê um .wav e retorna (amostras normalizadas [-1,1], sample_rate)."""
    with contextlib.closing(wave.open(str(file_path), "r")) as wf:
        n_frames = wf.getnframes()
        rate = wf.getframerate()
        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def plot_metrics_panel(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Painel com 2 subplots: velocidade de fala e proporção de silêncio
    por segmento, com marcadores de anomalia detectada e gabarito.

    Mesmo estilo visual das Etapas 1 e 2.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.patch.set_facecolor(COLORS["background"])
    fig.suptitle(
        "Painel de Análise de Fala — Métricas por Segmento com Detecção de Anomalias",
        fontsize=14, fontweight="bold", y=1.02
    )

    x = df["segment_id"].values

    plot_config = [
        ("words_per_second", "Velocidade de Fala\n(palavras/seg)", "#1976D2"),
        ("silence_ratio",     "Proporção de Silêncio\n(%)",         "#388E3C"),
    ]

    for ax, (col, label, color) in zip(axes, plot_config):
        ax.set_facecolor(COLORS["background"])

        if col not in df.columns:
            ax.set_ylabel(label, fontsize=9)
            continue

        y = df[col].values
        ax.plot(x, y, color=color, linewidth=1.5, marker="o", markersize=5, alpha=0.8)

        if "anomaly_final" in df.columns:
            mask_det = df["anomaly_final"] == 1
            ax.scatter(x[mask_det], y[mask_det],
                       color=COLORS["anomaly"], s=120, zorder=5, marker="X",
                       label="Anomalia detectada")

        if "is_injected_anomaly" in df.columns:
            mask_inj = df["is_injected_anomaly"] == 1
            ax.scatter(x[mask_inj], y[mask_inj],
                       color=COLORS["injected"], s=160, zorder=4,
                       marker="o", facecolors="none", linewidths=2,
                       label="Gabarito (injetada)")

        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.8)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("ID do Segmento (frase)", fontsize=10)
    axes[-1].set_xticks(x)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Painel de métricas salvo em: {save_path}")

    return fig


def plot_waveform(
    file_path: Path,
    segment_info: pd.Series,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plota o waveform (forma de onda) de um segmento de áudio, anotado
    com o texto transcrito e a classificação de anomalia.
    """
    samples, rate = _read_wav_samples(file_path)
    duration = len(samples) / rate if rate > 0 else 0
    t = np.linspace(0, duration, len(samples))

    is_anomaly = bool(segment_info.get("anomaly_final", 0))
    color = COLORS["anomaly"] if is_anomaly else COLORS["normal"]

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    ax.plot(t, samples, color=color, linewidth=0.6)
    ax.fill_between(t, samples, color=color, alpha=0.2)

    text = segment_info.get("transcribed_text", segment_info.get("original_text", ""))
    wps = segment_info.get("words_per_second", float("nan"))
    title = f"Segmento {segment_info.get('segment_id', '?')} — \"{text}\""
    subtitle = f"Velocidade: {wps:.2f} palavras/s"
    if is_anomaly:
        subtitle += "  [ANOMALIA DETECTADA]"

    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("Tempo (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


def save_sample_waveforms(
    df: pd.DataFrame,
    output_dir: Path,
    n_samples: int = 4,
) -> list[Path]:
    """
    Salva waveforms de segmentos representativos: alguns normais e
    alguns anômalos, para inspeção visual no relatório técnico.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_ids = df[df.get("anomaly_final", 0) == 1]["segment_id"].tolist()
    normal_ids  = df[df.get("anomaly_final", 0) == 0]["segment_id"].tolist()

    selected = anomaly_ids[:2] + normal_ids[:2]
    saved: list[Path] = []

    for seg_id in selected:
        row = df[df["segment_id"] == seg_id].iloc[0]
        file_path = Path(row.get("file_path", ""))
        if not file_path.exists():
            # tenta reconstruir caminho padrão
            file_path = Path(f"data/raw/audio_segments/frase_{seg_id:02d}.wav")
        if not file_path.exists():
            continue

        out_path = output_dir / f"waveform_segment_{seg_id:02d}.png"
        plot_waveform(file_path, row, save_path=out_path)
        saved.append(out_path)
        print(f"  Waveform salvo: {out_path}")

    return saved


if __name__ == "__main__":
    from src.audio.audio_processor import process_audio_dataset
    from src.audio.anomaly_detector import run_audio_detection_pipeline

    meta_path = Path("data/raw/audio_metadata.json")

    if not meta_path.exists():
        print(f"Metadados não encontrados em {meta_path}.")
        print("Execute primeiro: uv run python -m src.audio.synthetic_audio")
    else:
        print("Processando áudio com Whisper...")
        df_audio = process_audio_dataset(meta_path)

        print("Detectando anomalias...")
        df_result, metrics = run_audio_detection_pipeline(df_audio)

        # adiciona file_path de volta para o plotter localizar os .wav
        import json
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        path_map = {s["segment_id"]: s["file_path"] for s in meta["segments"]}
        df_result["file_path"] = df_result["segment_id"].map(path_map)

        print("Gerando visualizações...")
        plot_metrics_panel(
            df_result,
            save_path=Path("data/processed/audio_metrics_panel.png")
        )
        save_sample_waveforms(
            df_result,
            output_dir=Path("data/processed/audio_waveforms/"),
        )

        print("\nVisualização concluída!")
        print("\n=== MÉTRICAS FINAIS ===")
        for method, m in metrics.items():
            print(f"[{method}] Precisão={m['precisao']} | Recall={m['recall']} | F1={m['f1_score']}")