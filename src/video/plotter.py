"""
plotter.py  (src/video/)
========================
Gera imagens anotadas com keypoints e um painel de análise
de ângulos articulares ao longo do tempo.

Sem display gráfico — tudo salvo em arquivo, igual à Etapa 1.

Saídas:
    data/processed/pose_annotated_frame_NNN.png  — frames selecionados anotados
    data/processed/pose_angles_panel.png         — painel de ângulos vs tempo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path


# ── Paleta de cores (mesma da Etapa 1) ────────────────────────────────────────
COLORS = {
    "normal":     "#2196F3",   # azul
    "anomaly":    "#F44336",   # vermelho
    "injected":   "#FF9800",   # laranja (gabarito)
    "background": "#F8F9FA",   # cinza claro
    "grid":       "#E0E0E0",   # grade
}

# Conexões para desenhar o esqueleto nos frames anotados (BGR para OpenCV)
SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
]

# Ângulos a plotar no painel temporal
ANGLE_PLOT_CONFIG = {
    "angle_left_shoulder":  {"label": "Ombro Esquerdo",  "color": "#1976D2"},
    "angle_right_shoulder": {"label": "Ombro Direito",   "color": "#7B1FA2"},
    "angle_left_elbow":     {"label": "Cotovelo Esq",    "color": "#388E3C"},
    "angle_right_elbow":    {"label": "Cotovelo Dir",    "color": "#F57C00"},
}


# ── Anotação de frames individuais ────────────────────────────────────────────

def annotate_frame(
    frame_bgr: np.ndarray,
    df_row: pd.Series,
    df_full: pd.DataFrame,
) -> np.ndarray:
    """
    Desenha keypoints e ângulos sobre um frame BGR.

    Parâmetros:
        frame_bgr : array BGR do frame original
        df_row    : linha do DataFrame correspondente ao frame
        df_full   : DataFrame completo (para buscar coordenadas dos landmarks)

    Retorna frame anotado (array BGR).
    """
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    is_anomaly   = bool(df_row.get("anomaly_final", 0))
    is_injected  = bool(df_row.get("is_injected_anomaly", 0))

    # Cor dos elementos: vermelho em anomalia, azul em normal
    color_bgr = (0, 0, 220) if is_anomaly else (180, 80, 0)

    # Keypoints disponíveis no DataFrame
    kp_names = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
    ]

    kp_px: dict[str, tuple[int, int]] = {}
    for kp in kp_names:
        x_col, y_col = f"{kp}_x", f"{kp}_y"
        if x_col in df_row and y_col in df_row:
            x, y = df_row[x_col], df_row[y_col]
            if not (np.isnan(x) or np.isnan(y)):
                kp_px[kp] = (int(x * w), int(y * h))

    # Desenha conexões (esqueleto)
    for a_name, b_name in SKELETON_CONNECTIONS:
        if a_name in kp_px and b_name in kp_px:
            cv2.line(frame, kp_px[a_name], kp_px[b_name], color_bgr, 2, cv2.LINE_AA)

    # Desenha keypoints
    for kp, px in kp_px.items():
        cv2.circle(frame, px, 5, color_bgr, -1, cv2.LINE_AA)

    # Ângulos sobre o frame
    angle_texts = []
    for col in ["angle_left_shoulder", "angle_right_shoulder"]:
        val = df_row.get(col, float("nan"))
        if not np.isnan(val):
            label = col.replace("angle_", "").replace("_", " ").title()
            angle_texts.append(f"{label}: {val:.1f}°")

    for i, text in enumerate(angle_texts):
        cv2.putText(frame, text, (10, 45 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv2.LINE_AA)

    # Marcadores de anomalia
    if is_anomaly:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 200), 3)
        cv2.putText(frame, "ANOMALIA", (w // 2 - 60, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

    if is_injected:
        cv2.putText(frame, "[GABARITO]", (w - 110, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    # Índice do frame
    frame_idx = int(df_row.get("frame_idx", -1))
    cv2.putText(frame, f"Frame {frame_idx:03d}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

    return frame


def save_annotated_frames(
    frames: list[np.ndarray],
    df: pd.DataFrame,
    output_dir: Path,
    n_samples: int = 6,
) -> list[Path]:
    """
    Salva N frames representativos como PNG:
        - frames anômalos (primeiro de cada grupo)
        - frames normais (distribuídos uniformemente)

    Retorna lista de caminhos salvos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seleciona frames para salvar
    anomaly_idx = df[df["anomaly_final"] == 1]["frame_idx"].tolist()
    normal_idx  = df[df["anomaly_final"] == 0]["frame_idx"].tolist()

    # Primeiros 3 anômalos detectados + 3 normais espaçados
    selected_anomaly = anomaly_idx[:3] if anomaly_idx else []
    step = max(1, len(normal_idx) // 3)
    selected_normal  = normal_idx[::step][:3]
    selected = sorted(set(selected_anomaly + selected_normal))

    saved_paths: list[Path] = []
    for fidx in selected:
        if fidx >= len(frames):
            continue
        row = df[df["frame_idx"] == fidx].iloc[0]
        annotated = annotate_frame(frames[fidx], row, df)
        out_path = output_dir / f"pose_annotated_frame_{fidx:03d}.png"
        cv2.imwrite(str(out_path), annotated)
        saved_paths.append(out_path)
        print(f"  Frame anotado salvo: {out_path}")

    return saved_paths


# ── Painel de ângulos ao longo do tempo ───────────────────────────────────────

def plot_angles_panel(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Gera painel com 4 subplots — um por ângulo articular,
    com anomalias e gabarito marcados.

    Mesmo estilo visual da Etapa 1 (plotter.py dos vitais).
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.patch.set_facecolor(COLORS["background"])
    fig.suptitle(
        "Painel de Análise Postural — Ângulos Articulares com Detecção de Anomalias",
        fontsize=14, fontweight="bold", y=1.01
    )

    x = df["frame_idx"].values

    for ax, (col, config) in zip(axes, ANGLE_PLOT_CONFIG.items()):
        ax.set_facecolor(COLORS["background"])

        if col not in df.columns:
            ax.set_ylabel(config["label"], fontsize=9)
            ax.text(0.5, 0.5, "Dados não disponíveis",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        y = df[col].values

        # Linha principal
        ax.plot(x, y, color=config["color"], linewidth=1.2, alpha=0.75, label=config["label"])

        # Anomalias detectadas
        mask_det = df["anomaly_final"] == 1
        ax.scatter(x[mask_det], y[mask_det],
                   color=COLORS["anomaly"], s=60, zorder=5, marker="X",
                   label="Anomalia detectada")

        # Gabarito
        if "is_injected_anomaly" in df.columns:
            mask_inj = df["is_injected_anomaly"] == 1
            ax.scatter(x[mask_inj], y[mask_inj],
                       color=COLORS["injected"], s=100, zorder=4,
                       marker="o", facecolors="none", linewidths=2,
                       label="Gabarito (injetada)")

        # Limiar clínico de hiperextensão (apenas para ângulos de ombro)
        if "shoulder" in col:
            ax.axhline(160, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
                       label="Limite clínico (160°)")

        ax.set_ylabel(f"{config['label']}\n(graus)", fontsize=9)
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.8)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frame", fontsize=10)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Painel de ângulos salvo em: {save_path}")

    return fig


if __name__ == "__main__":
    from src.video.synthetic_video import generate_synthetic_video, ALL_ANOMALY_FRAMES
    from src.video.video_processor import process_video
    from src.video.anomaly_detector import run_video_detection_pipeline

    print("Gerando vídeo sintético...")
    frames, meta = generate_synthetic_video(
        output_video_path=Path("data/processed/synthetic_pose.mp4")
    )

    print("Processando com MediaPipe...")
    df_pose = process_video(frames, anomaly_frame_indices=ALL_ANOMALY_FRAMES)

    print("Detectando anomalias posturais...")
    df_result, metrics = run_video_detection_pipeline(df_pose)

    print("Gerando visualizações...")

    # Painel de ângulos
    plot_angles_panel(
        df_result,
        save_path=Path("data/processed/pose_angles_panel.png")
    )

    # Frames anotados
    save_annotated_frames(
        frames,
        df_result,
        output_dir=Path("data/processed/pose_frames/"),
    )

    print("\nVisualização concluída!")
    print("\n=== MÉTRICAS FINAIS ===")
    for method, m in metrics.items():
        print(f"[{method}] Precisão={m['precisao']} | Recall={m['recall']} | F1={m['f1_score']}")
