"""
test_video.py
==============
Testes do módulo de análise postural (Etapa 2).

Cobre a geometria de cálculo de ângulos (núcleo determinístico do
módulo) e a estrutura básica dos dados de pose sintética — sem
depender de codecs de vídeo ou ffmpeg (testados manualmente durante o
deploy, Seção 12 do relatório técnico).
"""

import numpy as np

from src.video.synthetic_video import (
    _build_pose_frame,
    ALL_ANOMALY_FRAMES,
    ANOMALY_FRAMES,
    N_FRAMES,
)


def test_build_pose_frame_returns_all_expected_keypoints():
    """
    Cada frame de pose deve conter os 9 keypoints usados pelo restante
    do pipeline (nariz, ombros, cotovelos, pulsos, quadris) — contrato
    básico consumido por video_processor.py.
    """
    pose = _build_pose_frame(frame_idx=0)

    expected_landmark_ids = {0, 11, 12, 13, 14, 15, 16, 23, 24}
    assert expected_landmark_ids.issubset(pose.keys())

    for x, y in pose.values():
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_anomaly_frame_indices_are_internally_consistent():
    """
    ALL_ANOMALY_FRAMES deve ser exatamente a união dos 4 grupos definidos
    em ANOMALY_FRAMES, e todos os índices devem estar dentro do range
    válido de frames do vídeo (0 a N_FRAMES-1) — garante que o gabarito
    usado na avaliação dos detectores está bem formado.
    """
    union_of_groups = set()
    for frames in ANOMALY_FRAMES.values():
        union_of_groups.update(frames)

    assert ALL_ANOMALY_FRAMES == union_of_groups
    assert all(0 <= idx < N_FRAMES for idx in ALL_ANOMALY_FRAMES)


def _shoulder_angle(pose: dict, side: str) -> float:
    """
    Replica o cálculo de ângulo do ombro feito por video_processor.py
    (vértice no ombro, entre o quadril e o cotovelo do mesmo lado) —
    mesma geometria usada pelo detector de anomalias real.
    """
    hip_id, shoulder_id, elbow_id = (
        (23, 11, 13) if side == "left" else (24, 12, 14)
    )
    hip = np.array(pose[hip_id])
    shoulder = np.array(pose[shoulder_id])
    elbow = np.array(pose[elbow_id])

    v1 = hip - shoulder
    v2 = elbow - shoulder
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def test_hyperextension_anomaly_produces_higher_angular_velocity():
    """
    Verifica que o mecanismo de injeção de anomalia produz um sinal
    mensurável na métrica que o detector real de fato utiliza:
    velocidade angular (variação do ângulo entre frames consecutivos),
    não a magnitude absoluta do ângulo.

    Esta distinção é deliberada e documentada no relatório técnico
    (Seção 7.5): o próprio movimento normal de elevação do braço já
    atinge picos de até ~179°, tornando um limiar de magnitude absoluta
    incapaz de diferenciar movimento saudável de patológico. Por isso, o
    detector real (anomaly_detector.py) usa a velocidade angular como
    sinal — uma subida anormalmente rápida do ângulo, característica da
    hiperextensão, independente do valor absoluto atingido.
    """
    def max_angular_velocity(frame_range):
        angles = [_shoulder_angle(_build_pose_frame(i), "right") for i in frame_range]
        velocities = [abs(angles[i] - angles[i - 1]) for i in range(1, len(angles))]
        return max(velocities)

    normal_velocity = max_angular_velocity(range(60))
    anomaly_velocity = max_angular_velocity(ANOMALY_FRAMES["hiperextensao_direita"])

    assert anomaly_velocity > normal_velocity