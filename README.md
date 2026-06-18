# 🏥 MedWatch - Sistema de Monitoramento Multimodal de Pacientes

### FIAP | Pós Tech IA para Devs - Tech Challenge Fase 4

Este projeto apresenta um sistema de monitoramento contínuo de pacientes utilizando análise **multimodal de dados** (vídeo, áudio e sinais vitais), com detecção de anomalias em tempo real e geração automática de alertas para a equipe médica. A solução é executada **100% localmente**, garantindo privacidade e conformidade com dados sensíveis de saúde (LGPD).

---

## 🚀 Visão Geral

O **MedWatch** integra três fontes de dados clínicos em um único pipeline de monitoramento:

- **Vídeo:** Análise de sessões de fisioterapia e cirurgias com detecção de padrões anômalos de movimento (MediaPipe Pose).
- **Áudio:** Transcrição e análise de consultas médicas para identificar sinais de fadiga e alterações vocais (Whisper local).
- **Sinais Vitais:** Detecção de anomalias em séries temporais de ECG, frequência cardíaca, SpO2 e pressão arterial (scikit-learn).

### Principais Diferenciais

- **Execução local (local-first):** Sem dependência de serviços em nuvem para processamento de dados sensíveis.
- **Análise multimodal integrada:** Fusão dos três canais de dados em um motor unificado de alertas.
- **LLM via Groq API:** Extração de termos críticos, análise de sentimento e sumarização de laudos.
- **Dados sintéticos baseados em padrões reais:** Sinais vitais gerados com base no formato MIT-BIH (PhysioNet), com anomalias injetadas de forma controlada.
- **Alertas estruturados:** Geração automática de alertas em JSON para a equipe médica.

> **Nota sobre adaptação arquitetural:** O desafio sugeria o uso de serviços Azure (Speech-to-Text, Text Analytics) e AWS (Comprehend). Optamos por alternativas locais e Groq API por razões de privacidade de dados clínicos, custo zero e independência de conectividade. A seção de Relatório Técnico detalha essa decisão.

---

## 🛠️ Estrutura do Projeto

´´´md
├── data/
│   ├── raw/               # Dados sintéticos gerados (vídeo, áudio, sinais vitais)
│   └── processed/         # Saídas processadas (transcrições, CSVs de anomalias)
├── docs/
│   └── relatorio_tecnico.md   # Relatório técnico completo
├── notebook/              # Notebooks exploratórios
├── src/
│   ├── video/             # Módulo de análise de vídeo (MediaPipe Pose)
│   ├── audio/             # Módulo de transcrição e análise vocal (Whisper)
│   ├── vitals/            # Módulo de sinais vitais e detecção de anomalias
│   ├── alerts/            # Motor de fusão e geração de alertas
│   └── llm/               # Integração com Groq API
├── tests/                 # Testes por módulo
├── .env.example           # Modelo de variáveis de ambiente
├── main.py                # Ponto de entrada principal
├── pyproject.toml
└── README.md
´´´
---

## ⚙️ Como Executar

### Pré-requisitos

- **Python 3.12+** (via `uv`)
- **Chave Groq API** gratuita em [console.groq.com](https://console.groq.com)

### Passo 1: Clonar o Repositório

```bash
git clone git@github.com:leojosants/fiap-pos-tech-ia-para-devs-tech-challenge-fase-4.git
cd fiap-pos-tech-ia-para-devs-tech-challenge-fase-4
```

### Passo 2: Configurar Variáveis de Ambiente

```bash
cp .env.example .env
# Edite o .env e adicione sua GROQ_API_KEY
```

### Passo 3: Instalar Dependências

```bash
uv sync
```

### Passo 4: Executar o Sistema

```bash
uv run python main.py
```

---

## 🧪 Módulos e Funcionalidades

| Módulo | Tecnologia | Função |
|--------|-----------|--------|
| `src/video/` | MediaPipe Pose | Detecção de keypoints e padrões anômalos de movimento |
| `src/audio/` | Whisper local | Transcrição de consultas e análise de alterações vocais |
| `src/vitals/` | scikit-learn | Detecção de anomalias em sinais vitais (z-score + Isolation Forest) |
| `src/llm/` | Groq API | Análise de termos críticos, sentimento e sumarização |
| `src/alerts/` | Python | Fusão multimodal e geração de alertas estruturados |

---

## 📊 Datasets

Os dados utilizados neste projeto são **sintéticos**, gerados com base em padrões reais de referência:

- **Sinais vitais:** Baseados no formato MIT-BIH Arrhythmia Database ([PhysioNet](https://physionet.org/)) — 360 Hz, faixa mV, padrão de batimento cardíaco.
- **Áudio:** Gravações sintéticas simulando consultas médicas.
- **Vídeo:** Vídeos simulando sessões de fisioterapia com movimentos normais e anômalos.

---

## 🎓 Instituição

**FIAP - Pós Tech IA para Devs**
**Tech Challenge - Fase 4**
**Autor:** [leojosants](https://github.com/leojosants)

---

> Este projeto foi desenvolvido como prova de conceito para monitoramento multimodal de pacientes com IA aplicada à saúde.
