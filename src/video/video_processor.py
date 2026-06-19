"""
video_processor.py
==================
Extrai keypoints e ângulos articulares frame a frame do vídeo sintético.

Estratégia para dados sintéticos (sem mp.solutions disponível):
    Como o vídeo é gerado matematicamente por synthetic_video._build_pose_frame(),
    extraímos as coordenadas diretamente dessas funções — sem visão computacional.

    Isso é equivalente ao MediaPipe para dados sintéticos: ambos retornam
    coordenadas normalizadas [0-1] dos mesmos keypoints.

    Em produção com vídeo real, basta substituir _get_pose_from_frame()
    pelo PoseLandmarker com o arquivo pose_landmarker.task do Google.

Interface pública (process_video, process_frame):
    Idêntica ao que anomaly_detector.py e plotter.py esperam.
    Drop-in replacement sem alterar os outros módulos.

Keypoints rastreados (mesma nomenclatura do MediaPipe Pose):
    left_shoulder, right_shoulder, left_elbow, right_elbow,
    left_wrist, right_wrist, left_hip, right_hip
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Keypoints rastreados ──────────────────────────────────────────────────────
KP_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
]

# Pares para cálculo de ângulo: (ponto_A, vértice, ponto_B)
ANGLE_TRIPLETS = {
    "angle_left_shoulder":  ("left_hip",       "left_shoulder",  "left_elbow"),
    "angle_right_shoulder": ("right_hip",      "right_shoulder", "right_elbow"),
    "angle_left_elbow":     ("left_shoulder",  "left_elbow",     "left_wrist"),
    "angle_right_elbow":    ("right_shoulder", "right_elbow",    "right_wrist"),
}

# IDs dos landmarks no synthetic_video.KP
_KP_IDS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
}


def _calc_angle(
    a: tuple[float, float],
    vertex: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Ângulo em graus no vértice, entre os vetores vertex→a e vertex→b."""
    va = np.array([a[0] - vertex[0], a[1] - vertex[1]], dtype=float)
    vb = np.array([b[0] - vertex[0], b[1] - vertex[1]], dtype=float)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    cos_a = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _extract_angles(
    kp: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Calcula os 4 ângulos articulares a partir dos keypoints."""
    angles: dict[str, float] = {}
    for name, (a, vtx, b) in ANGLE_TRIPLETS.items():
        if all(k in kp for k in (a, vtx, b)):
            angles[name] = round(_calc_angle(kp[a], kp[vtx], kp[b]), 2)
        else:
            angles[name] = float("nan")
    return angles


def _get_pose_from_synthetic(frame_idx: int) -> dict[str, tuple[float, float]]:
    """
    Recupera coordenadas normalizadas [0-1] de cada keypoint
    chamando diretamente _build_pose_frame() do synthetic_video.

    Retorna dicionário {nome: (x, y)} com a mesma estrutura
    que o MediaPipe retornaria em vídeo real.
    """
    from src.video.synthetic_video import _build_pose_frame, KP as SV_KP

    pose_raw = _build_pose_frame(frame_idx)

    kp: dict[str, tuple[float, float]] = {}
    for name, kp_id in _KP_IDS.items():
        if kp_id in pose_raw:
            kp[name] = pose_raw[kp_id][:2]   # (x, y) normalizados

    return kp


def process_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    anomaly_frame_indices: set[int] | None = None,
) -> dict:
    """
    Processa um único frame e retorna dicionário com ângulos e coordenadas.

    Para dados sintéticos: usa _get_pose_from_synthetic() — equivale ao
    MediaPipe extraindo os mesmos landmarks gerados matematicamente.

    Parâmetros:
        frame_bgr            : frame BGR (mantido na assinatura para compatibilidade)
        frame_idx            : índice do frame
        anomaly_frame_indices: conjunto de frames com anomalia (gabarito)
    """
    row: dict = {
        "frame_idx": frame_idx,
        "pose_detected": False,
        "is_injected_anomaly": (
            1 if anomaly_frame_indices and frame_idx in anomaly_frame_indices else 0
        ),
    }

    # Inicializa colunas com NaN
    for col in ANGLE_TRIPLETS:
        row[col] = float("nan")
    for kp_name in KP_NAMES:
        row[f"{kp_name}_x"] = float("nan")
        row[f"{kp_name}_y"] = float("nan")

    kp = _get_pose_from_synthetic(frame_idx)

    if len(kp) >= 6:
        row["pose_detected"] = True
        row.update(_extract_angles(kp))
        for kp_name, (x, y) in kp.items():
            row[f"{kp_name}_x"] = round(x, 5)
            row[f"{kp_name}_y"] = round(y, 5)

    return row


def process_video(
    frames: list[np.ndarray],
    anomaly_frame_indices: set[int] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Processa lista de frames e retorna DataFrame com resultados.

    Interface idêntica ao que anomaly_detector.py e plotter.py esperam.
    """
    rows = [
        process_frame(frame, idx, anomaly_frame_indices)
        for idx, frame in enumerate(frames)
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.video.synthetic_video import generate_synthetic_video, ALL_ANOMALY_FRAMES

    print("Gerando vídeo sintético...")
    frames, meta = generate_synthetic_video()

    print(f"Processando {len(frames)} frames...")
    df = process_video(frames, anomaly_frame_indices=ALL_ANOMALY_FRAMES)

    detected = df["pose_detected"].sum()
    print(f"\n=== RESULTADO DO PROCESSAMENTO ===")
    print(f"  Frames processados : {len(df)}")
    print(f"  Pose detectada em  : {detected} frames ({detected/len(df)*100:.1f}%)")
    print(f"  Frames anômalos    : {df['is_injected_anomaly'].sum()}")

    angle_cols = ["frame_idx", "pose_detected",
                  "angle_left_shoulder", "angle_right_shoulder",
                  "is_injected_anomaly"]
    print("\nPrimeiras 5 linhas:")
    print(df[angle_cols].head().to_string())
    print("\nAlguns frames anômalos:")
    anom = df[df["is_injected_anomaly"] == 1][angle_cols].head()
    print(anom.to_string())