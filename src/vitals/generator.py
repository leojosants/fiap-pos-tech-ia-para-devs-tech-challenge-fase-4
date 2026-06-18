
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# Configurações baseadas no padrão MIT-BIH (PhysioNet)
# Sampling rate original: 360 Hz — aqui usamos 1 leitura/segundo (escala de monitoramento)
SAMPLING_RATE_HZ = 1
DURATION_MINUTES = 10
N_SAMPLES = DURATION_MINUTES * 60 * SAMPLING_RATE_HZ  # 600 amostras


def generate_heart_rate(n_samples: int, anomaly_indices: list[int]) -> np.ndarray:
    """
    Gera série temporal de frequência cardíaca (bpm).
    Faixa normal: 60–100 bpm (adulto em repouso).
    Anomalia: pico acima de 130 bpm (taquicardia) ou abaixo de 45 bpm (bradicardia).
    """
    # Sinal base com ruído gaussiano leve (simula variação natural)
    signal = np.random.normal(loc=72, scale=4, size=n_samples)

    # Injeta anomalias de forma controlada
    for idx in anomaly_indices:
        if idx < n_samples:
            anomaly_type = np.random.choice(["taquicardia", "bradicardia"])
            if anomaly_type == "taquicardia":
                signal[idx] = np.random.uniform(130, 160)
            else:
                signal[idx] = np.random.uniform(35, 44)

    return np.round(signal, 2)


def generate_spo2(n_samples: int, anomaly_indices: list[int]) -> np.ndarray:
    """
    Gera série temporal de saturação de oxigênio (SpO2 em %).
    Faixa normal: 95–100%.
    Anomalia: queda abaixo de 90% (hipóxia).
    """
    signal = np.random.normal(loc=98, scale=0.5, size=n_samples)
    signal = np.clip(signal, 94, 100)  # Limita ao range fisiológico normal

    for idx in anomaly_indices:
        if idx < n_samples:
            signal[idx] = np.random.uniform(82, 89)

    return np.round(signal, 1)


def generate_blood_pressure(n_samples: int, anomaly_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera séries temporais de pressão arterial sistólica e diastólica (mmHg).
    Faixa normal: sistólica 90–120 mmHg / diastólica 60–80 mmHg.
    Anomalia: sistólica > 160 mmHg (hipertensão grave) ou < 85 mmHg (hipotensão).
    """
    systolic = np.random.normal(loc=115, scale=5, size=n_samples)
    diastolic = np.random.normal(loc=75, scale=3, size=n_samples)

    for idx in anomaly_indices:
        if idx < n_samples:
            anomaly_type = np.random.choice(["hipertensao", "hipotensao"])
            if anomaly_type == "hipertensao":
                systolic[idx] = np.random.uniform(160, 185)
                diastolic[idx] = np.random.uniform(100, 115)
            else:
                systolic[idx] = np.random.uniform(70, 84)
                diastolic[idx] = np.random.uniform(45, 59)

    return np.round(systolic, 1), np.round(diastolic, 1)


def generate_vitals_dataframe(seed: int = 42) -> pd.DataFrame:
    """
    Gera um DataFrame completo de sinais vitais sintéticos com anomalias injetadas.
    Retorna um DataFrame pronto para análise e detecção de anomalias.
    """
    np.random.seed(seed)  # Reprodutibilidade

    # Índices de anomalia: posições fixas e conhecidas (para validação)
    anomaly_indices = [60, 150, 250, 380, 480, 550]

    # Gera timestamps (1 leitura por segundo, a partir de agora)
    start_time = datetime.now().replace(microsecond=0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(N_SAMPLES)]

    # Gera cada sinal
    heart_rate = generate_heart_rate(N_SAMPLES, anomaly_indices)
    spo2 = generate_spo2(N_SAMPLES, anomaly_indices)
    systolic, diastolic = generate_blood_pressure(N_SAMPLES, anomaly_indices)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "heart_rate_bpm": heart_rate,
        "spo2_pct": spo2,
        "bp_systolic_mmhg": systolic,
        "bp_diastolic_mmhg": diastolic,
        "is_injected_anomaly": [1 if i in anomaly_indices else 0 for i in range(N_SAMPLES)]
    })

    return df


if __name__ == "__main__":
    df = generate_vitals_dataframe()
    print(f"Dataset gerado: {len(df)} amostras")
    print(f"Anomalias injetadas: {df['is_injected_anomaly'].sum()} pontos")
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print("\nEstatísticas básicas:")
    print(df.describe())