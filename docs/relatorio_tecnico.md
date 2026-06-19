# Relatório Técnico — Sistema de Monitoramento Multimodal de Pacientes (MedWatch)

**FIAP | Pós Tech em IA para Devs — Tech Challenge Fase 4**
**Autor:** Leonardo José de Oliveira Santos (RM369985)
**Repositório:** github.com/leojosants/fiap-pos-tech-ia-para-devs-tech-challenge-fase-4

---

## 1. Introdução e Contexto do Desafio

O desafio proposto pela Fase 4 solicita o desenvolvimento de um sistema de
monitoramento contínuo de pacientes hospitalares por meio de dados
multimodais — vídeo, áudio e sinais vitais — com o objetivo de identificar
sinais precoces de risco clínico. O sistema deve:

- Analisar vídeos de cirurgias ou sessões de fisioterapia para identificar
  padrões anômalos de movimento;
- Processar gravações de voz de pacientes em consultas, detectando sintomas
  relacionados à fala (fadiga, disartria);
- Detectar anomalias em sinais vitais, prescrições e evolução clínica,
  alertando a equipe médica em tempo real;
- Integrar os módulos com serviços de IA para ampliar a capacidade de
  processamento.

O enunciado original sugere o uso de serviços gerenciados em nuvem (Azure
Cognitive Services, AWS Comprehend) para parte dessas tarefas. Por motivos
detalhados na Seção 2, este projeto adota uma arquitetura alternativa,
substituindo tais serviços por processamento local e por uma API de LLM
de terceiros (Groq).

---

## 2. Adaptação Arquitetural: Azure/AWS → Soluções Locais + Groq API

Esta é uma decisão de engenharia deliberada, e não uma limitação. A
justificativa é tripla:

1. **Privacidade de dados clínicos.** Dados de pacientes (voz, vídeo,
   sinais vitais) são informações sensíveis sob a LGPD. Mantendo o
   processamento local, eliminamos o risco de trânsito e armazenamento
   desses dados em serviços de terceiros, exceto quando estritamente
   necessário (uso de LLM para tarefas de linguagem).
2. **Independência de custo e de conectividade.** Serviços em nuvem pagos
   (Azure Speech, Azure Text Analytics, AWS Comprehend) introduzem custo
   recorrente e dependência de internet estável. As alternativas locais
   eliminam ambos os fatores, tornando a solução viável para ambientes com
   recursos computacionais limitados — o caso deste autor.
3. **Equivalência funcional comprovada.** Cada substituição foi escolhida
   por oferecer capacidade equivalente à do serviço original, validada
   durante as aulas do curso (ver Tabela 1).

**Tabela 1 — Mapeamento de substituições**

| Requisito do desafio              | Serviço sugerido (Azure/AWS) | Solução adotada                  |
|------------------------------------|-------------------------------|-----------------------------------|
| Transcrição de áudio               | Azure Speech to Text          | Whisper (OpenAI, execução local)  |
| Análise de sentimento/termos       | Azure Text Analytics / AWS Comprehend | Groq API (LLM)             |
| Análise postural em vídeo          | OpenPose                      | MediaPipe Pose (Google, local)    |
| Sumarização de laudos              | —                              | Groq API (LLM)                    |
| Detecção de anomalias em séries    | —                              | Z-score + Isolation Forest (scikit-learn) |
| LLM para tarefas de linguagem      | OpenAI GPT                    | Groq API (compatível com SDK OpenAI) |

A Groq API foi escolhida em vez da OpenAI por oferecer inferência de alta
velocidade a custo reduzido, mantendo compatibilidade de interface com o
SDK da OpenAI — o que simplifica a integração sem acoplamento a um único
fornecedor.

---

## 3. Datasets Utilizados

Conforme sugerido no enunciado do desafio, avaliou-se o uso do PhysioNet
como fonte de dados reais de sinais vitais. Optou-se, no entanto, por
**gerar dados sintéticos com características estatísticas equivalentes ao
padrão MIT-BIH Arrhythmia Database** (PhysioNet), pelos seguintes motivos:

- Controle total sobre a posição e o tipo de anomalia, permitindo validar
  a eficácia dos algoritmos de detecção contra um gabarito conhecido;
- Eliminação de dependência de rede externa durante o desenvolvimento e a
  demonstração da solução;
- Anomalias raramente aparecem de forma didática em recortes curtos de
  dados reais, exigindo de qualquer forma a injeção artificial de eventos
  anômalos para fins de demonstração.

Características reais do MIT-BIH replicadas na simulação: amostragem
originalmente a 360 Hz, resolução de 11 bits, faixa de ±10 mV, dois canais
de ECG por registro. O dataset sintético adapta esses parâmetros para uma
escala de monitoramento clínico padrão (1 leitura/segundo), mais adequada
à interpretação de sinais vitais discretos (frequência cardíaca, SpO2,
pressão arterial) do que ao sinal de ECG bruto.

---

## 4. Arquitetura Geral do Sistema

O sistema é estruturado em módulos independentes, cada um responsável por
uma modalidade de dado, convergindo em um motor central de fusão e alertas:
Vídeo (MediaPipe) ──┐
Áudio (Whisper)  ───┼──► Análise via LLM (Groq) ──► Motor de Fusão e Alertas ──► Equipe médica
Sinais vitais ──────┘                                      │
(estatística/ML local)                                     └──► Relatório técnico

Todo o processamento de vídeo, áudio e sinais vitais ocorre localmente.
A Groq API é utilizada exclusivamente para tarefas de linguagem natural
(extração de termos críticos, sentimento, sumarização) sobre o texto já
transcrito localmente — nenhum áudio ou vídeo bruto é enviado a serviços
externos.

---

## 5. Etapa 1 — Módulo de Sinais Vitais

### 5.1 Geração de dados sintéticos (`src/vitals/generator.py`)

Gera um DataFrame com 600 amostras (10 minutos, 1 amostra/segundo) contendo
quatro sinais vitais:

| Sinal                  | Faixa normal      | Tipo de anomalia injetada              |
|-------------------------|-------------------|------------------------------------------|
| Frequência cardíaca (bpm) | 60–100           | Taquicardia (130–160) / Bradicardia (35–44) |
| SpO2 (%)                 | 95–100           | Hipóxia (82–89)                          |
| Pressão sistólica (mmHg) | 90–120            | Hipertensão (160–185) / Hipotensão (70–84) |
| Pressão diastólica (mmHg)| 60–80             | Hipertensão (100–115) / Hipotensão (45–59) |

As anomalias são injetadas em seis índices fixos e conhecidos
(`[60, 150, 250, 380, 480, 550]`), com `np.random.seed(42)` garantindo
reprodutibilidade total entre execuções. A coluna `is_injected_anomaly`
funciona como gabarito (ground truth) para avaliação quantitativa dos
detectores.

### 5.2 Detecção de anomalias (`src/vitals/detector.py`)

Foram implementados e comparados dois métodos:

**Z-score (estatístico).** Mede quantos desvios padrão cada ponto está da
média do próprio sinal. Pontos com `|z| > 3` são marcados como anômalos.
Método simples, interpretável e eficaz para anomalias estatisticamente
extremas — como as injetadas neste dataset.

**Isolation Forest (machine learning, `scikit-learn`).** Modelo não
supervisionado que isola pontos anômalos por meio de árvores de decisão
aleatórias; pontos que exigem menos divisões para serem isolados são mais
prováveis de serem anômalos. Analisa os quatro sinais vitais de forma
multivariada, capturando padrões que a análise univariada do z-score não
detectaria.

**Estratégia combinada.** Um ponto é classificado como anomalia final se
**qualquer um** dos dois métodos o identificar (operação lógica OR). A
escolha maximiza o recall (sensibilidade) — em contexto clínico, o custo
de um falso negativo (deixar de detectar uma anomalia real) é
substancialmente maior que o custo de um falso positivo.

**Resultados obtidos** (parâmetro `contamination=0.015` no Isolation
Forest, após ajuste experimental):

| Método              | Anomalias injetadas | Detectadas | Precisão | Recall | F1-Score |
|----------------------|----------------------|------------|----------|--------|----------|
| Z-score              | 6                    | 6          | 1.000    | 1.000  | 1.000    |
| Isolation Forest      | 6                    | 9          | 0.667    | 1.000  | 0.800    |
| Combinado (OR)        | 6                    | 9          | 0.667    | 1.000  | 0.800    |

**Discussão.** O z-score atingiu desempenho perfeito porque as anomalias
injetadas são estatisticamente extremas e univariadas — exatamente o
cenário em que esse método é mais eficaz. O Isolation Forest, embora não
tenha superado o z-score neste dataset sintético, mantém recall de 1.000
(nenhuma anomalia real foi perdida), validando sua utilidade como camada
de segurança redundante. Em sinais vitais reais, com padrões mais sutis e
correlações multivariadas, espera-se que o Isolation Forest assuma um
papel mais relevante do que neste cenário controlado.

O parâmetro `contamination` do Isolation Forest foi calibrado
experimentalmente: valores muito baixos (0.0015) resultaram em
subdetecção severa (recall 0.167); valores muito altos (0.05) geraram
excesso de falsos positivos (precisão 0.200). O valor `0.015` (1.5% do
dataset) produziu o melhor equilíbrio para este cenário.

### 5.3 Visualização (`src/vitals/plotter.py`)

Foram implementadas duas funções de plotagem com `matplotlib`:

- `plot_vital_signal()`: gráfico individual de um sinal vital, com faixa
  de normalidade sombreada, anomalias detectadas (marcador X vermelho) e
  anomalias injetadas/gabarito (círculo laranja vazado).
- `plot_all_vitals()`: painel com os quatro sinais vitais em subplots
  empilhados, compartilhando o mesmo eixo temporal — permitindo
  identificar visualmente quando múltiplas anomalias ocorrem de forma
  simultânea em diferentes sinais, o que indicaria um evento clínico de
  maior gravidade.

O painel completo é salvo em `data/processed/vitals_panel.png` e
demonstra visualmente a coincidência entre as anomalias detectadas e as
injetadas, validando a eficácia da estratégia combinada.

![Painel de sinais vitais](../data/processed/vitals_panel.png)

---

## 6. Estrutura do Projeto (estado atual)

├── data/
│   ├── raw/
│   └── processed/
│       └── vitals_panel.png
├── docs/
│   └── relatorio_tecnico.md
├── notebook/
├── src/
│   ├── video/
│   ├── audio/
│   ├── vitals/
│   │   ├── generator.py
│   │   ├── detector.py
│   │   └── plotter.py
│   ├── alerts/
│   └── llm/
├── tests/
├── .env.example
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
└── README.md
---

## 7. Próximas Etapas

- Etapa 2: Análise de vídeo com MediaPipe Pose (detecção de padrões
  anômalos de movimento em sessões de fisioterapia simuladas);
- Etapa 3: Transcrição e análise de áudio com Whisper local;
- Etapa 4: Integração com Groq API para extração de termos críticos,
  sentimento e sumarização;
- Etapa 5: Motor de fusão multimodal e geração de alertas;
- Etapa 6: Interface Streamlit;
- Etapa 7: Consolidação final do relatório técnico;
- Etapa 8: Deploy no Streamlit Cloud.

*Este relatório é um documento vivo, atualizado incrementalmente conforme
o desenvolvimento avança.*
