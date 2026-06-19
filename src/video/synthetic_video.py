"""
synthetic_video.py
==================
Gera um vídeo sintético de sessão de fisioterapia usando coordenadas
de pose construídas matematicamente via NumPy.

NÃO requer câmera, display gráfico nem arquivo de vídeo externo.

Contexto clínico simulado:
    Paciente realiza exercício de elevação bilateral dos braços.
    Anomalias injetadas representam movimentos patológicos:
      - Hiperextensão (ângulo acima do esperado)
      - Assimetria entre braços (lado esquerdo vs direito)
      - Colapso de tronco (inclinação excessiva)

Saída:
    data/processed/synthetic_pose.mp4  — vídeo com stickman animado
    data/raw/synthetic_pose_frames.npy — array de frames (backup sem cv2)
"""

import numpy as np
import cv2
from pathlib import Path


# ── Configurações do vídeo ────────────────────────────────────────────────────
WIDTH = 640
HEIGHT = 480
FPS = 15
DURATION_SECONDS = 20          # 300 frames a 15 fps
N_FRAMES = DURATION_SECONDS * FPS

# ── Índices de frames com anomalias injetadas ─────────────────────────────────
# Mantemos o padrão da Etapa 1: índices fixos e conhecidos para avaliação
ANOMALY_FRAMES = {
    "hiperextensao_direita": list(range(60, 75)),    # braço direito sobe demais
    "assimetria":            list(range(130, 145)),  # braços em alturas diferentes
    "colapso_tronco":        list(range(200, 215)),  # tronco inclinado
    "hiperextensao_esquerda": list(range(250, 265)), # braço esquerdo sobe demais
}
ALL_ANOMALY_FRAMES: set[int] = set()
for frames in ANOMALY_FRAMES.values():
    ALL_ANOMALY_FRAMES.update(frames)


# ── Keypoints do MediaPipe Pose (33 landmarks) ────────────────────────────────
# Usamos apenas os relevantes para o exercício de braço:
#   0  = nariz
#   11 = ombro esquerdo   12 = ombro direito
#   13 = cotovelo esq     14 = cotovelo dir
#   15 = pulso esq        16 = pulso dir
#   23 = quadril esq      24 = quadril dir
KP = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
}


def _build_pose_frame(frame_idx: int) -> dict[int, tuple[float, float]]:
    """
    Calcula as coordenadas (x, y) normalizadas [0-1] de cada keypoint
    para o frame dado, simulando o movimento de elevação de braços.

    Retorna dicionário {landmark_id: (x, y)}.
    """
    t = frame_idx / N_FRAMES          # progresso 0→1
    phase = (frame_idx % 60) / 60.0  # fase do ciclo de movimento (0→1, repete)

    # ── Corpo fixo (centro da cena) ───────────────────────────────────────────
    cx = 0.5   # centro horizontal

    # Cabeça/nariz
    nose_y = 0.18
    pose: dict[int, tuple[float, float]] = {
        KP["nose"]: (cx, nose_y),
    }

    # Tronco — inclinação padrão zero; anomalia de colapso adiciona desvio
    trunk_tilt = 0.0
    if frame_idx in ALL_ANOMALY_FRAMES and frame_idx in range(200, 215):
        # Colapso de tronco: inclina o corpo para a esquerda
        trunk_tilt = 0.07 * np.sin(np.pi * (frame_idx - 200) / 15)

    # Ombros
    shoulder_y = 0.35
    pose[KP["left_shoulder"]]  = (cx - 0.10 + trunk_tilt, shoulder_y + trunk_tilt * 0.5)
    pose[KP["right_shoulder"]] = (cx + 0.10 + trunk_tilt, shoulder_y + trunk_tilt * 0.5)

    # Quadris
    hip_y = 0.62
    pose[KP["left_hip"]]  = (cx - 0.08 + trunk_tilt, hip_y)
    pose[KP["right_hip"]] = (cx + 0.08 + trunk_tilt, hip_y)

    # ── Braços: movimento de elevação senoidal ────────────────────────────────
    # Ângulo base: 0° (ao lado do corpo) → 90° (horizontal) → 150° (normal max)
    base_angle_deg = 30 + 110 * np.abs(np.sin(np.pi * phase))  # 30°–140°

    # Ângulos individuais com ou sem anomalia
    angle_right = base_angle_deg
    angle_left  = base_angle_deg

    if frame_idx in ALL_ANOMALY_FRAMES:
        if frame_idx in range(60, 75):
            # Hiperextensão direita: ângulo sobe além de 160°
            angle_right = base_angle_deg + 35 * np.sin(np.pi * (frame_idx - 60) / 15)
        if frame_idx in range(130, 145):
            # Assimetria: braço esquerdo não levanta junto
            angle_left = base_angle_deg * 0.3
        if frame_idx in range(250, 265):
            # Hiperextensão esquerda
            angle_left = base_angle_deg + 35 * np.sin(np.pi * (frame_idx - 250) / 15)

    def arm_keypoints(
        shoulder: tuple[float, float],
        angle_deg: float,
        side: str
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calcula cotovelo e pulso a partir do ombro e ângulo."""
        sign = -1 if side == "left" else 1
        rad = np.radians(angle_deg)
        arm_len = 0.12    # comprimento do segmento (normalizado)
        forearm_len = 0.11

        elbow_x = shoulder[0] + sign * arm_len * np.cos(rad)
        elbow_y = shoulder[1] - arm_len * np.sin(rad)

        wrist_x = elbow_x + sign * forearm_len * np.cos(rad + np.radians(10))
        wrist_y = elbow_y - forearm_len * np.sin(rad + np.radians(10))

        return (elbow_x, elbow_y), (wrist_x, wrist_y)

    elbow_l, wrist_l = arm_keypoints(pose[KP["left_shoulder"]],  angle_left,  "left")
    elbow_r, wrist_r = arm_keypoints(pose[KP["right_shoulder"]], angle_right, "right")

    pose[KP["left_elbow"]]  = elbow_l
    pose[KP["left_wrist"]]  = wrist_l
    pose[KP["right_elbow"]] = elbow_r
    pose[KP["right_wrist"]] = wrist_r

    return pose


def _render_frame(
    pose: dict[int, tuple[float, float]],
    frame_idx: int,
    is_anomaly: bool
) -> np.ndarray:
    """
    Renderiza um frame BGR (numpy array H×W×3) com o stickman
    desenhado a partir das coordenadas de pose.
    """
    frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 240  # fundo cinza claro

    # Converte coordenadas normalizadas para pixels
    def to_px(xy: tuple[float, float]) -> tuple[int, int]:
        x, y = xy
        return (int(x * WIDTH), int(y * HEIGHT))

    # Cor das conexões: verde normal / vermelho em anomalia
    conn_color = (0, 0, 200) if is_anomaly else (50, 150, 50)
    kp_color   = (0, 0, 255) if is_anomaly else (0, 200, 0)

    # ── Conexões (segmentos do corpo) ─────────────────────────────────────────
    connections = [
        (KP["left_shoulder"],  KP["right_shoulder"]),  # tronco superior
        (KP["left_shoulder"],  KP["left_hip"]),         # lado esquerdo
        (KP["right_shoulder"], KP["right_hip"]),        # lado direito
        (KP["left_hip"],       KP["right_hip"]),        # quadril
        (KP["left_shoulder"],  KP["left_elbow"]),       # braço esq
        (KP["left_elbow"],     KP["left_wrist"]),       # antebraço esq
        (KP["right_shoulder"], KP["right_elbow"]),      # braço dir
        (KP["right_elbow"],    KP["right_wrist"]),      # antebraço dir
        (KP["nose"],           KP["left_shoulder"]),    # pescoço esq
        (KP["nose"],           KP["right_shoulder"]),   # pescoço dir
    ]

    for a, b in connections:
        if a in pose and b in pose:
            cv2.line(frame, to_px(pose[a]), to_px(pose[b]), conn_color, 3, cv2.LINE_AA)

    # ── Keypoints ─────────────────────────────────────────────────────────────
    for kp_id, xy in pose.items():
        cv2.circle(frame, to_px(xy), 6, kp_color, -1, cv2.LINE_AA)

    # ── Cabeça (círculo) ──────────────────────────────────────────────────────
    if KP["nose"] in pose:
        cv2.circle(frame, to_px(pose[KP["nose"]]), 18, conn_color, 2, cv2.LINE_AA)

    # ── Overlay de informação ─────────────────────────────────────────────────
    frame_text = f"Frame {frame_idx:03d}/{N_FRAMES}"
    cv2.putText(frame, frame_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)

    if is_anomaly:
        cv2.putText(frame, "!! ANOMALIA DETECTADA !!", (WIDTH // 2 - 150, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)
        # Borda vermelha
        cv2.rectangle(frame, (3, 3), (WIDTH - 3, HEIGHT - 3), (0, 0, 200), 3)

    return frame


def generate_synthetic_video(
    output_video_path: Path | None = None,
    output_frames_path: Path | None = None,
    seed: int = 42
) -> tuple[list[np.ndarray], dict]:
    """
    Gera o vídeo sintético e retorna:
        frames: lista de arrays BGR (um por frame)
        metadata: informações sobre o vídeo gerado

    Parâmetros:
        output_video_path : caminho para salvar o .mp4 (opcional)
        output_frames_path: caminho para salvar frames em .npy (opcional)
        seed              : semente para reprodutibilidade

    O vídeo é gerado inteiramente via NumPy/OpenCV — sem câmera.
    """
    np.random.seed(seed)

    frames: list[np.ndarray] = []

    for frame_idx in range(N_FRAMES):
        is_anomaly = frame_idx in ALL_ANOMALY_FRAMES
        pose = _build_pose_frame(frame_idx)
        frame = _render_frame(pose, frame_idx, is_anomaly)
        frames.append(frame)

    # ── Salva como .mp4 ───────────────────────────────────────────────────────
    if output_video_path is not None:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_video_path), fourcc, FPS, (WIDTH, HEIGHT)
        )
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"Vídeo salvo em: {output_video_path}")

    # ── Salva frames como numpy array (útil para testes sem cv2 writer) ──────
    if output_frames_path is not None:
        output_frames_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_frames_path), np.array(frames))
        print(f"Frames salvos em: {output_frames_path}")

    metadata = {
        "n_frames": N_FRAMES,
        "fps": FPS,
        "duration_seconds": DURATION_SECONDS,
        "width": WIDTH,
        "height": HEIGHT,
        "total_anomaly_frames": len(ALL_ANOMALY_FRAMES),
        "anomaly_groups": {k: len(v) for k, v in ANOMALY_FRAMES.items()},
        "anomaly_frame_indices": sorted(ALL_ANOMALY_FRAMES),
    }

    return frames, metadata


if __name__ == "__main__":
    video_path  = Path("data/processed/synthetic_pose.mp4")
    frames_path = Path("data/raw/synthetic_pose_frames.npy")

    print("Gerando vídeo sintético de fisioterapia...")
    frames, meta = generate_synthetic_video(
        output_video_path=video_path,
        output_frames_path=frames_path,
    )

    print(f"\n=== METADADOS DO VÍDEO ===")
    print(f"  Frames totais   : {meta['n_frames']}")
    print(f"  FPS             : {meta['fps']}")
    print(f"  Duração         : {meta['duration_seconds']}s")
    print(f"  Frames anômalos : {meta['total_anomaly_frames']}")
    print(f"  Grupos de anomalia:")
    for grupo, qtd in meta["anomaly_groups"].items():
        print(f"    - {grupo}: {qtd} frames")
