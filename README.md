# 🏥 MedWatch — Sistema de Monitoramento Multimodal de Pacientes

### FIAP | Pós Tech em IA para Devs — Tech Challenge Fase 4

Sistema de monitoramento contínuo de pacientes hospitalares por meio de
análise **multimodal de dados** — vídeo, áudio e sinais vitais — com
detecção de anomalias e geração automática de alertas para a equipe
médica. A solução é executada **100% localmente** sempre que possível,
garantindo privacidade e conformidade com dados sensíveis de saúde
(LGPD); a única exceção é a camada de interpretação de linguagem
natural, que usa a Groq API exclusivamente sobre texto já transcrito.

---

## 📄 Relatório Técnico e Aplicação Publicada

> **Este projeto não inclui vídeo de demonstração gravado**, por
> limitação de hardware já justificada perante a coordenação do curso.
> Em substituição, a entrega é composta por dois artefatos
> complementares — detalhados na Seção 1.1 do relatório técnico:

- 📘 **[Relatório técnico completo (PDF)](docs/relatorio_tecnico.pdf)** —
  escrito para funcionar como a apresentação do projeto: decisões de
  design, problemas reais encontrados durante o desenvolvimento e o
  deploy, e resultados obtidos em cada etapa.
- 🌐 **[Aplicação publicada (Streamlit Cloud)](https://medwatch-fiap-fase4.streamlit.app/)** —
  permite executar o sistema interativamente, com um único clique, sem
  depender de nenhuma narração gravada.

---

## 🚀 Visão Geral

O MedWatch integra três fontes de dados clínicos em um único pipeline
de monitoramento, cruzando os achados em um motor de fusão que produz
um alerta consolidado:

- **Sinais vitais** — detecção de anomalias em séries temporais de
  frequência cardíaca, SpO2 e pressão arterial, via z-score +
  Isolation Forest (`scikit-learn`).
- **Vídeo** — análise de uma sessão de fisioterapia sintética, com
  detecção de assimetria bilateral, hiperextensão e colapso de tronco.
- **Áudio** — transcrição local via Whisper de frases de triagem
  médica sintetizadas por TTS, com detecção de padrões anômalos de
  velocidade de fala (lentificada/acelerada).
- **LLM (Groq API)** — extração de termos críticos, classificação de
  urgência e sumarização clínica a partir do texto já transcrito.
- **Motor de fusão e alertas** — combina os quatro sinais acima em um
  nível de alerta único (Normal / Atenção / Crítico), com justificativa
  textual auditável.

### Principais diferenciais

- **Local-first:** todo processamento de vídeo, áudio e sinais vitais
  ocorre localmente; nenhum dado bruto sensível é enviado a serviços
  externos.
- **Dados sintéticos com gabarito conhecido:** todas as anomalias são
  injetadas de forma controlada, permitindo avaliação quantitativa
  (precisão/recall/F1) de cada detector contra um resultado esperado.
- **Interface interativa (Streamlit):** um único botão executa as
  cinco etapas do pipeline e popula um dashboard com quatro abas.
- **Testes automatizados (`pytest`):** 18 testes cobrindo a lógica
  determinística de cada módulo, com chamadas a Whisper/Groq mockadas.

> **Nota sobre adaptação arquitetural:** o desafio sugeria serviços
> Azure (Speech-to-Text, Text Analytics) e AWS (Comprehend). Optou-se
> por alternativas locais e pela Groq API, por razões de privacidade de
> dados clínicos, custo zero e independência de conectividade — decisão
> detalhada na Seção 2 do relatório técnico.

---

## 🛠️ Estrutura do Projeto

```
fiap-pos-tech-ia-para-devs-tech-challenge-fase-4/
├── data/
│   ├── raw/
│   │   ├── audio_segments/        # 12 .wav sintéticos de triagem (Etapa 3)
│   │   └── audio_metadata.json    # gabarito das frases (texto, tipo de anomalia)
│   └── processed/                 # artefatos gerados em runtime
│       ├── audio_waveforms/       # waveforms anotados por segmento
│       └── pose_frames/           # frames anotados da análise postural
├── docs/
│   ├── relatorio_tecnico.md       # relatório técnico (fonte markdown)
│   ├── relatorio_tecnico.pdf      # relatório técnico (entrega final)
│   └── imagens/                   # imagens versionadas usadas no relatório
├── notebook/
├── src/
│   ├── vitals/                    # Etapa 1 — sinais vitais
│   │   ├── generator.py           #   geração sintética + injeção de anomalias
│   │   ├── detector.py            #   z-score + Isolation Forest
│   │   └── plotter.py
│   ├── video/                     # Etapa 2 — análise postural
│   │   ├── synthetic_video.py     #   geração do vídeo + extração analítica de pose
│   │   ├── video_processor.py     #   cálculo de ângulos articulares
│   │   ├── anomaly_detector.py
│   │   └── plotter.py
│   ├── audio/                     # Etapa 3 — fala
│   │   ├── synthetic_audio.py     #   TTS local (pyttsx3) — frases de triagem
│   │   ├── audio_processor.py     #   transcrição via Whisper local
│   │   ├── anomaly_detector.py
│   │   └── plotter.py
│   ├── llm/                       # Etapa 4 — Groq API
│   │   ├── groq_client.py         #   cliente robusto (st.secrets + .env)
│   │   ├── prompts.py
│   │   ├── text_analyzer.py       #   extração de termos/urgência
│   │   └── summarizer.py          #   resumo clínico executivo
│   ├── alerts/                    # Etapa 5 — fusão multimodal
│   │   ├── alert_rules.py         #   regra de decisão de alerta
│   │   ├── fusion_engine.py
│   │   ├── alert_dispatcher.py    #   log + histórico em JSON
│   │   ├── plotter.py             #   dashboard de alerta
│   │   └── run_pipeline.py        #   pipeline completo via CLI
│   └── dashboard/                 # Etapa 6 — páginas da interface Streamlit
│       ├── pages_vitals.py
│       ├── pages_video.py
│       ├── pages_audio.py
│       └── pages_alerts.py
├── tests/                         # suíte de testes automatizados (pytest)
├── .env.example                   # GROQ_API_KEY, GROQ_MODEL_FAST, GROQ_MODEL_SMART
├── .gitignore
├── main.py                        # ponto de entrada da aplicação Streamlit
├── packages.txt                   # dependências de sistema (deploy — ffmpeg)
├── pytest.ini
├── pyproject.toml
└── requirements.txt                # dependências Python (deploy)
```

---

## ⚙️ Como Executar Localmente

### Pré-requisitos

- Python 3.12.9
- [`uv`](https://docs.astral.sh/uv/) como gerenciador de pacotes
- Chave de API da Groq, gratuita em [console.groq.com](https://console.groq.com)
- Motor de TTS disponível para geração de áudio sintético (SAPI5 — nativo
  no Windows; em Linux requer `espeak`/`espeak-ng` via gerenciador de
  pacotes do sistema)

### Passo 1 — Clonar o repositório

```bash
git clone git@github.com:leojosants/fiap-pos-tech-ia-para-devs-tech-challenge-fase-4.git
cd fiap-pos-tech-ia-para-devs-tech-challenge-fase-4
```

### Passo 2 — Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite o .env e preencha GROQ_API_KEY com sua chave
```

### Passo 3 — Instalar dependências

```bash
uv sync
```

### Passo 4 — Executar a aplicação

```bash
uv run streamlit run main.py
```

Abre automaticamente em `http://localhost:8501`. Clique em
**"▶️ Executar Monitoramento Completo"** e aguarde a barra de progresso
(30-60 segundos na primeira execução — inclui download do modelo
Whisper e duas chamadas à Groq API).

### Rodar os testes automatizados

```bash
uv add --dev pytest   # se ainda não instalado
uv run pytest
```

Resultado esperado: 18 testes aprovados em poucos segundos, sem
nenhuma chamada real a Whisper ou à Groq API (todas mockadas).

---

## 🧪 Módulos e Tecnologias

| Módulo | Tecnologia | Função |
|--------|-----------|--------|
| `src/vitals/` | `scikit-learn` | Detecção de anomalias em sinais vitais (z-score + Isolation Forest) |
| `src/video/` | `OpenCV`, geometria vetorial | Geração de vídeo sintético + extração analítica de pose e ângulos articulares¹ |
| `src/audio/` | `pyttsx3`, `openai-whisper` | Síntese de fala (TTS local) e transcrição local |
| `src/llm/` | Groq API (`gpt-oss-20b`/`120b`) | Extração de termos críticos, urgência e sumarização clínica |
| `src/alerts/` | Python puro | Fusão multimodal e geração de alertas estruturados |
| `src/dashboard/` | `Streamlit` | Interface interativa com 4 abas (Alertas, Vitais, Vídeo, Áudio) |

¹ A extração de pose é feita analiticamente a partir das coordenadas já
calculadas na geração do vídeo sintético, e não via MediaPipe Pose —
decisão técnica detalhada na Seção 7.2 do relatório (a API do MediaPipe
0.10.x exige modelo externo baixado separadamente, o que conflitaria com
a filosofia local-first do projeto).

---

## 📊 Datasets

Todos os dados utilizados são **sintéticos**, com características
estatísticas baseadas em padrões reais e gabarito de anomalias conhecido
— permitindo avaliação quantitativa de cada detector (ver relatório
técnico, Seções 5, 7 e 8):

- **Sinais vitais:** características inspiradas no MIT-BIH Arrhythmia
  Database ([PhysioNet](https://physionet.org/)), adaptado para a escala
  de monitoramento clínico discreto (1 amostra/segundo).
- **Vídeo:** sessão de fisioterapia simulada matematicamente (exercício
  de elevação bilateral de braços), com anomalias posturais injetadas
  em janelas de frames conhecidas.
- **Áudio:** 12 frases de triagem médica sintetizadas via TTS local,
  com perfis de velocidade/pausa simulando fala normal, lentificada e
  acelerada.

---

## ☁️ Deploy

Aplicação publicada no **Streamlit Community Cloud**, a partir da
branch `main`: **<https://medwatch-fiap-fase4.streamlit.app/>**

O processo de deploy revelou e exigiu a correção de seis problemas
reais específicos do ambiente de servidor Linux (codec de vídeo,
permissões de arquivo, separador de caminho, leitura de secrets, entre
outros) — documentados em detalhe na Seção 12 do relatório técnico.

---

## 🎓 Instituição

**FIAP — Pós Tech em IA para Devs**
**Tech Challenge — Fase 4**
**Autor:** Leonardo José de Oliveira Santos (RM369985) — [github.com/leojosants](https://github.com/leojosants)

---

> Este projeto foi desenvolvido como prova de conceito acadêmica para
> monitoramento multimodal de pacientes com IA aplicada à saúde. Não é
> destinado a uso clínico real sem validação extensiva adicional — ver
> Seção 13 (Considerações Éticas) do relatório técnico.
