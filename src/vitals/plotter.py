
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# Paleta de cores consistente em todo o projeto
COLORS = {
    "normal": "#2196F3",       # Azul — pontos normais
    "anomaly": "#F44336",      # Vermelho — anomalias detectadas
    "injected": "#FF9800",     # Laranja — anomalias injetadas (gabarito)
    "background": "#F8F9FA",   # Cinza claro — fundo dos gráficos
    "grid": "#E0E0E0"          # Cinza — grid
}

# Configurações dos sinais vitais para os gráficos
VITALS_CONFIG = {
    "heart_rate_bpm": {
        "label": "Frequência Cardíaca",
        "unit": "bpm",
        "normal_range": (60, 100),
        "color": "#1976D2"
    },
    "spo2_pct": {
        "label": "Saturação de Oxigênio (SpO2)",
        "unit": "%",
        "normal_range": (95, 100),
        "color": "#388E3C"
    },
    "bp_systolic_mmhg": {
        "label": "Pressão Sistólica",
        "unit": "mmHg",
        "normal_range": (90, 120),
        "color": "#7B1FA2"
    },
    "bp_diastolic_mmhg": {
        "label": "Pressão Diastólica",
        "unit": "mmHg",
        "normal_range": (60, 80),
        "color": "#F57C00"
    }
}


def plot_vital_signal(
    df: pd.DataFrame,
    column: str,
    anomaly_column: str = "anomaly_final",
    save_path: Path | None = None
) -> plt.Figure:
    """
    Gera gráfico de um sinal vital com anomalias destacadas.

    Parâmetros:
        df: DataFrame com os sinais vitais e flags de anomalia
        column: nome da coluna do sinal a plotar
        anomaly_column: coluna com as flags de anomalia detectada
        save_path: se informado, salva a figura nesse caminho

    Retorna a figura matplotlib (compatível com st.pyplot() no Streamlit).
    """
    config = VITALS_CONFIG.get(column, {})
    label = config.get("label", column)
    unit = config.get("unit", "")
    normal_min, normal_max = config.get("normal_range", (None, None))
    color = config.get("color", COLORS["normal"])

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Separa pontos normais e anômalos
    mask_anomaly = df[anomaly_column] == 1
    mask_normal = ~mask_anomaly

    # Plota linha contínua do sinal
    ax.plot(
        df["timestamp"],
        df[column],
        color=color,
        linewidth=1.2,
        alpha=0.7,
        label=f"{label} ({unit})",
        zorder=2
    )

    # Destaca pontos anômalos detectados
    ax.scatter(
        df.loc[mask_anomaly, "timestamp"],
        df.loc[mask_anomaly, column],
        color=COLORS["anomaly"],
        s=80,
        zorder=5,
        label="Anomalia detectada",
        marker="X"
    )

    # Destaca pontos do gabarito (anomalias injetadas)
    if "is_injected_anomaly" in df.columns:
        mask_injected = df["is_injected_anomaly"] == 1
        ax.scatter(
            df.loc[mask_injected, "timestamp"],
            df.loc[mask_injected, column],
            color=COLORS["injected"],
            s=120,
            zorder=4,
            label="Anomalia injetada (gabarito)",
            marker="o",
            facecolors="none",
            linewidths=2
        )

    # Faixa de normalidade (área sombreada)
    if normal_min is not None and normal_max is not None:
        ax.axhspan(
            normal_min, normal_max,
            alpha=0.08,
            color=color,
            label=f"Faixa normal ({normal_min}–{normal_max} {unit})"
        )
        ax.axhline(normal_min, color=color, linewidth=0.8, linestyle="--", alpha=0.4)
        ax.axhline(normal_max, color=color, linewidth=0.8, linestyle="--", alpha=0.4)

    # Formatação dos eixos
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax.set_title(f"{label} — Monitoramento com Detecção de Anomalias", fontsize=13, pad=12)
    ax.set_xlabel("Tempo", fontsize=10)
    ax.set_ylabel(f"{label} ({unit})", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.8)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")

    return fig


def plot_all_vitals(
    df: pd.DataFrame,
    anomaly_column: str = "anomaly_final",
    save_path: Path | None = None
) -> plt.Figure:
    """
    Gera um painel com todos os 4 sinais vitais em subplots empilhados.
    Ideal para o relatório técnico e para a aba de overview no Streamlit.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.patch.set_facecolor(COLORS["background"])
    fig.suptitle(
        "Painel de Monitoramento — Sinais Vitais com Detecção de Anomalias",
        fontsize=14,
        fontweight="bold",
        y=1.01
    )

    for ax, (col, config) in zip(axes, VITALS_CONFIG.items()):
        label = config["label"]
        unit = config["unit"]
        color = config["color"]
        normal_min, normal_max = config["normal_range"]

        mask_anomaly = df[anomaly_column] == 1

        ax.set_facecolor(COLORS["background"])
        ax.plot(
            df["timestamp"],
            df[col],
            color=color,
            linewidth=1.2,
            alpha=0.75
        )
        ax.scatter(
            df.loc[mask_anomaly, "timestamp"],
            df.loc[mask_anomaly, col],
            color=COLORS["anomaly"],
            s=60,
            zorder=5,
            marker="X",
            label="Anomalia detectada"
        )

        # Gabarito — anomalias injetadas
        if "is_injected_anomaly" in df.columns:
            mask_injected = df["is_injected_anomaly"] == 1
            ax.scatter(
                df.loc[mask_injected, "timestamp"],
                df.loc[mask_injected, col],
                color=COLORS["injected"],
                s=100,
                zorder=4,
                marker="o",
                facecolors="none",
                linewidths=2,
                label="Gabarito (injetada)"
            )
        ax.axhspan(normal_min, normal_max, alpha=0.07, color=color)
        ax.set_ylabel(f"{label}\n({unit})", fontsize=9)
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.8)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    axes[-1].set_xlabel("Tempo", fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Painel salvo em: {save_path}")

    return fig


if __name__ == "__main__":
    from src.vitals.generator import generate_vitals_dataframe
    from src.vitals.detector import run_detection_pipeline

    df = generate_vitals_dataframe()
    df_result, _ = run_detection_pipeline(df)

    # Salva o painel completo em data/processed/
    output_path = Path("data/processed/vitals_panel.png")
    plot_all_vitals(df_result, save_path=output_path)
    print("Painel gerado com sucesso!")