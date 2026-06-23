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
| Análise de sentimento/termos       | Azure Text Analytics / AWS Comprehend | Groq API (modelo openai/gpt-oss-20b)² |
| Análise postural em vídeo          | OpenPose                      | MediaPipe Pose (Google) + extração analítica de pose¹ |
| Sumarização de laudos              | —                              | Groq API (modelo openai/gpt-oss-120b)² |
| Detecção de anomalias em séries    | —                              | Z-score + Isolation Forest (scikit-learn) |
| LLM para tarefas de linguagem      | OpenAI GPT                    | Groq API (compatível com SDK OpenAI) |

¹ Ver nota técnica na Seção 7.2 sobre a adaptação de API do MediaPipe 0.10.x.
² Modelos vigentes no momento da entrega — ver nota técnica na Seção 9.2 sobre depreciação de modelos pela Groq e a estratégia de configuração via .env adotada para mitigar esse risco.

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

A mesma filosofia de dados sintéticos controlados foi estendida ao módulo
de vídeo (Seção 8): em vez de capturar vídeo real (indisponível por
limitação de hardware), o movimento de um paciente em sessão de
fisioterapia é simulado matematicamente, com anomalias posturais
injetadas em janelas de frames conhecidas — preservando o mesmo princípio
de avaliação quantitativa contra gabarito usado na Etapa 1.

---

## 4. Arquitetura Geral do Sistema

O sistema é estruturado em módulos independentes, cada um responsável por
uma modalidade de dado, convergindo em um motor central de fusão e alertas:

```
Vídeo (MediaPipe/análise postural) ──┐
Áudio (Whisper)                  ────┼──► Análise via LLM (Groq) ──► Motor de Fusão e Alertas ──► Equipe médica
Sinais vitais (estatística/ML local) ┘                                      │
                                                                              └──► Relatório técnico
```

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

![Painel de sinais vitais](imagens/vitals_panel.png)

---

## 6. Estrutura do Projeto (estado atual)

```
├── data/
│   ├── raw/
│   │   ├── synthetic_pose_frames.npy
│   │   ├── audio_segments/
│   │   └── audio_metadata.json
│   └── processed/
│       ├── vitals_panel.png
│       ├── synthetic_pose.mp4
│       ├── pose_angles_panel.png
│       ├── pose_frames/
│       ├── audio_metrics_panel.png
│       └── audio_waveforms/
├── docs/
│   └── relatorio_tecnico.md
├── notebook/
├── src/
│   ├── video/
│   │   ├── synthetic_video.py
│   │   ├── video_processor.py
│   │   ├── anomaly_detector.py
│   │   └── plotter.py
│   ├── audio/
│   │   ├── synthetic_audio.py
│   │   ├── audio_processor.py
│   │   ├── anomaly_detector.py
│   │   └── plotter.py
│   ├── llm/
│   │   ├── groq_client.py
│   │   ├── prompts.py
│   │   ├── text_analyzer.py
│   │   └── summarizer.py
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
```

---

## 7. Etapa 2 — Módulo de Análise de Vídeo

### 7.1 Contexto e cenário clínico simulado

O desafio sugere a análise de vídeo de **cirurgias** ou **sessões de
fisioterapia**, com o objetivo de identificar padrões anômalos de
movimento. Optou-se por simular uma **sessão de fisioterapia com exercício
de elevação bilateral dos braços** — um exercício clínico real, comum em
reabilitação de ombro, no qual o terapeuta avalia justamente os três
padrões patológicos que o módulo foi desenhado para detectar:

| Anomalia clínica         | Significado em fisioterapia                                   |
|---------------------------|------------------------------------------------------------------|
| Hiperextensão             | Paciente força o movimento além do limite seguro da articulação |
| Assimetria bilateral      | Um braço compensa o esforço do outro — indício de lesão unilateral |
| Colapso de tronco         | Paciente inclina o corpo para compensar amplitude de movimento insuficiente |

Como não há vídeo real disponível (limitação de hardware, já justificada
no desafio da fase anterior), o vídeo é **gerado matematicamente** via
NumPy/OpenCV: um "stickman" articulado se move de forma análoga ao
exercício real, e os mesmos índices de anomalia conhecidos (técnica já
validada na Etapa 1) são usados para avaliar os detectores.

### 7.2 Nota técnica — adaptação da API do MediaPipe

O plano original previa o uso de `mediapipe.solutions.pose` (API clássica,
usada nas aulas do curso) para extrair os 33 *landmarks* corporais do
vídeo. Durante a implementação, identificou-se que a versão do MediaPipe
instalada via `uv add mediapipe` (0.10.x) **removeu o módulo
`mp.solutions`**, substituindo-o por uma nova API baseada em tarefas
(`mediapipe.tasks.python.vision.PoseLandmarker`), que exige o download de
um arquivo de modelo `.task` externo (~30MB, hospedado pelo Google).

Essa exigência de download de modelo externo conflita diretamente com o
princípio de **execução 100% local e sem dependência de rede** adotado
neste projeto (Seção 2). A solução adotada foi:

- Para os **dados sintéticos** desta etapa: extrair as coordenadas dos
  *keypoints* diretamente das funções matemáticas que geram o vídeo
  (`synthetic_video._build_pose_frame()`), que produzem exatamente os
  mesmos 8 pontos corporais (ombros, cotovelos, pulsos, quadris) que o
  MediaPipe Pose retornaria, na mesma convenção de nomenclatura e de
  coordenadas normalizadas [0, 1].
- Para uma eventual extensão futura com **vídeo real de câmera**, o
  projeto está estruturado para permitir a troca direta da função de
  extração de pose por `PoseLandmarker`, mediante o download único do
  modelo `.task` (op-in explícito do usuário, documentado no README).

Essa decisão preserva a interface conceitual do MediaPipe (mesmos nomes
de *landmarks*, mesmas convenções de coordenadas) sem comprometer a
filosofia local-first do projeto, e está documentada no código-fonte
(`src/video/video_processor.py`) para transparência total perante a
banca avaliadora.

### 7.3 Geração do vídeo sintético (`src/video/synthetic_video.py`)

Gera 300 frames (20 segundos a 15 fps) simulando o exercício de elevação
bilateral de braços. O ângulo do ombro varia de forma senoidal entre 30°
e 140° ao longo do tempo, reproduzindo o ciclo natural de subida e descida
do braço.

Quatro grupos de anomalia são injetados em janelas fixas de 15 frames
cada (1 segundo de duração — tempo suficiente para caracterizar um
movimento patológico, e não apenas um pico instantâneo de ruído):

| Grupo de anomalia          | Frames    | Mecanismo simulado                                  |
|------------------------------|-----------|--------------------------------------------------------|
| `hiperextensao_direita`      | 60–74     | Ombro direito atinge ângulo muito acima do padrão     |
| `assimetria`                 | 130–144   | Braço esquerdo não acompanha a elevação do direito    |
| `colapso_tronco`             | 200–214   | Ambos os ombros se deslocam lateralmente em conjunto  |
| `hiperextensao_esquerda`     | 250–264   | Ombro esquerdo atinge ângulo muito acima do padrão    |

O vídeo é renderizado quadro a quadro com OpenCV (linhas e círculos
representando o esqueleto), sem qualquer dependência de câmera ou
display gráfico — escolha de projeto alinhada à ausência de hardware de
captura e à necessidade de rodar em ambiente de deploy headless (Streamlit
Cloud, Seção 12 — a confirmar).

Saídas geradas:
- `data/processed/synthetic_pose.mp4` — vídeo renderizado completo;
- `data/raw/synthetic_pose_frames.npy` — array NumPy dos frames brutos,
  usado como *backup* para reprocessamento sem decodificar o `.mp4`.

### 7.4 Extração de keypoints e ângulos articulares (`src/video/video_processor.py`)

Para cada frame, extrai 8 *keypoints* corporais (ombros, cotovelos,
pulsos e quadris, em ambos os lados) e calcula 4 ângulos articulares via
geometria vetorial — o mesmo princípio trigonométrico usado por sistemas
reais de análise postural:

```
ângulo(A, vértice, B) = arccos( (vértice→A · vértice→B) / (|vértice→A| × |vértice→B|) )
```

Os quatro ângulos calculados:

| Ângulo                  | Vértice          | Significado clínico                          |
|---------------------------|-------------------|--------------------------------------------------|
| `angle_left_shoulder`     | Ombro esquerdo    | Elevação do braço esquerdo em relação ao tronco  |
| `angle_right_shoulder`    | Ombro direito     | Elevação do braço direito em relação ao tronco   |
| `angle_left_elbow`        | Cotovelo esquerdo | Grau de flexão/extensão do cotovelo esquerdo     |
| `angle_right_elbow`       | Cotovelo direito  | Grau de flexão/extensão do cotovelo direito      |

O resultado é um DataFrame com uma linha por frame (300 linhas),
estruturado de forma idêntica ao padrão da Etapa 1: colunas de ângulos,
coordenadas normalizadas dos *keypoints*, flag `pose_detected` e coluna
de gabarito `is_injected_anomaly`. Nos testes realizados, a pose foi
detectada com sucesso em **100% dos 300 frames**.

### 7.5 Detecção de anomalias posturais (`src/video/anomaly_detector.py`)

Foram implementados **três métodos complementares**, combinados por OR
lógico — mesma filosofia da Etapa 1 (priorizar recall em contexto
clínico):

**1. Z-score de assimetria bilateral.** Calcula o desvio estatístico da
diferença entre o ângulo do ombro esquerdo e do direito. Em um movimento
simétrico normal, essa diferença deve ser próxima de zero; um desvio
estatisticamente significativo (`|z| > 2.0`) indica compensação
unilateral.

**2. Z-score de velocidade angular.** Calcula a variação do ângulo entre
frames consecutivos (uma aproximação discreta da velocidade angular) e
aplica z-score sobre essa série. Movimentos patológicos como a
hiperextensão produzem mudanças bruscas de ângulo (picos de até 13°/frame
contra uma média normal de ~3,6°/frame), que esse método captura mesmo
quando o ângulo absoluto não excede um limiar fixo.

**3. Regras clínicas.** Duas regras de limiar fixo, calibradas
empiricamente para o dataset sintético:
   - Assimetria direta: `|ângulo_esquerdo − ângulo_direito| > 25°`;
   - Colapso de tronco: z-score do deslocamento horizontal conjunto dos
     dois ombros `> 2.0` (captura quando ambos os ombros migram para o
     mesmo lado, característico de inclinação de tronco).

**Decisão técnica relevante.** Uma primeira tentativa de regra fixa para
hiperextensão (`ângulo_ombro > 160°`) revelou-se inadequada: o próprio
exercício normal de elevação de braço atinge picos de até 179° no ciclo
saudável, tornando esse limiar absoluto incapaz de distinguir movimento
normal de patológico. A solução foi abandonar o limiar absoluto em favor
da **detecção de velocidade angular anômala** (Método 2), que captura a
*característica* da hiperextensão — uma subida anormalmente rápida do
ângulo — independente do valor absoluto atingido. Esse ajuste está
documentado no histórico de commits e ilustra o processo iterativo de
calibração de um detector de anomalias contra dados sintéticos
realistas.

**Resultados obtidos:**

| Método                   | Anomalias injetadas | Detectadas | Precisão | Recall | F1-Score |
|----------------------------|----------------------|------------|----------|--------|----------|
| Z-score (assimetria)       | 60                   | 19         | 1.000    | 0.317  | 0.481    |
| Z-score (velocidade)       | 60                   | 5          | 1.000    | 0.083  | 0.154    |
| Regras clínicas            | 60                   | 30         | 1.000    | 0.500  | 0.667    |
| **Combinado (OR)**         | 60                   | 34         | **1.000**| **0.567**| **0.723**|

**Discussão.** O resultado mais notável é a **precisão perfeita (1.000)
em todos os métodos e em sua combinação** — nenhum dos detectores gerou
falso positivo em frames normais, mesmo com o ângulo natural do exercício
alcançando valores elevados (até 179°). Isso confirma que os métodos
estão capturando *características específicas* das anomalias (assimetria,
aceleração brusca, deslocamento lateral conjunto), e não apenas reagindo
a valores absolutos do movimento normal.

O recall de 0.567 no método combinado é inferior ao 1.000 obtido na
Etapa 1, refletindo uma diferença de natureza dos dados: nos sinais
vitais, a anomalia é um evento pontual e isolado; no movimento postural,
a anomalia é um *processo gradual* que se desenrola ao longo de ~15
frames, e nem todo frame dentro dessa janela apresenta desvio estatístico
extremo (os frames de transição entrada/saída da anomalia são
naturalmente mais próximos do padrão normal). Por grupo de anomalia, a
detecção variou entre 7/15 e 12/15 frames — suficiente para identificar
o evento clínico (a deteção de qualquer frame dentro da janela já aciona
um alerta no motor de fusão, a ser implementado na Etapa 5), mas com
espaço de melhoria a ser discutido em trabalhos futuros (Seção 7.7).

### 7.6 Visualização (`src/video/plotter.py`)

Duas saídas visuais, seguindo o mesmo padrão visual da Etapa 1:

- **Painel de ângulos** (`pose_angles_panel.png`): quatro subplots, um
  por ângulo articular, com a curva do ângulo ao longo do tempo,
  marcadores de anomalia detectada (X vermelho) e gabarito (círculo
  laranja vazado) — réplica direta do estilo de `vitals_panel.png`;
- **Frames anotados** (`pose_frames/pose_annotated_frame_NNN.png`): seis
  frames representativos (3 anômalos + 3 normais) com o esqueleto
  desenhado sobre a imagem, ângulos numéricos sobrepostos, e marcação
  visual clara (borda vermelha + texto "ANOMALIA") quando o frame é
  classificado como anômalo pelo pipeline.

Nenhuma das duas saídas requer display gráfico (`cv2.imshow`) — ambas são
gravadas diretamente em disco, característica essencial para execução em
ambiente headless (servidor sem interface gráfica) e para o deploy final
no Streamlit Cloud.

![Painel de ângulos articulares](imagens/pose_angles_panel.png)

### 7.7 Limitações conhecidas e trabalhos futuros

- O recall de 0,567 indica que parte dos frames dentro das janelas de
  anomalia (especialmente os frames de transição) não são capturados
  pelos limiares atuais. Uma extensão futura poderia aplicar suavização
  temporal (ex.: média móvel sobre os ângulos) antes da detecção, ou
  classificar a janela inteira como anômala caso qualquer frame dentro
  dela ultrapasse o limiar — abordagem mais alinhada à granularidade
  clínica real (eventos de segundos, não de milissegundos).
- A extração de *keypoints* desta etapa depende do vídeo ser gerado
  sinteticamente (Seção 7.2). Para uso com vídeo real de câmera, é
  necessário o download do modelo `pose_landmarker.task` do MediaPipe —
  decisão de design que preserva a filosofia local-first por padrão,
  exigindo *opt-in* explícito apenas quando vídeo real for de fato
  utilizado.

---

## 8. Etapa 3 — Módulo de Análise de Áudio

### 9.1 Contexto e cenário clínico simulado

O desafio sugere a análise de gravações de voz de pacientes em consultas
para detectar sintomas relacionados à fala (fadiga, disartria). Optou-se
por simular uma **sessão de triagem médica**, na qual o paciente responde
a 12 frases curtas sobre seu estado de saúde e sintomas — formato real e
comum em anamnese clínica.

Três padrões de fala foram modelados, com base em sinais clínicos
reconhecidos:

| Anomalia de fala     | Significado clínico                                              |
|------------------------|---------------------------------------------------------------------|
| Fala lentificada/pausada | Possível fadiga extrema, confusão mental, ou disartria (ex.: AVC) |
| Fala acelerada           | Possível agitação, ansiedade, taquilalia                          |
| Fala normal              | Linha de base — referência para comparação estatística            |

Como não há gravação de paciente real disponível, o áudio é **gerado
localmente via TTS (Text-to-Speech)** com a biblioteca `pyttsx3`, que
utiliza o motor de voz nativo do sistema operacional (SAPI5 no Windows) —
sem necessidade de internet, sem custo, e sem envio de dados a serviços
externos. As mesmas 12 frases são sintetizadas em três perfis de
velocidade/pausa, com o tipo de anomalia conhecido a priori, funcionando
como gabarito — mesmo princípio de avaliação quantitativa usado nas
Etapas 1 e 2.

### 9.2 Nota técnica — dependência do `ffmpeg` no pipeline do Whisper

Durante a implementação, identificou-se que a biblioteca `openai-whisper`
depende do executável de linha de comando `ffmpeg` para decodificar
arquivos de áudio — uma dependência de sistema, não uma dependência
Python, e portanto não resolvida automaticamente por `uv add`. Em um
ambiente Windows limpo (sem Chocolatey, Scoop ou instalação manual prévia),
essa chamada falha com `FileNotFoundError: [WinError 2]`.

A solução adotada evita qualquer instalação manual fora do ecossistema
gerenciado pelo projeto: o pacote `imageio-ffmpeg` (distribuído via PyPI)
empacota um executável `ffmpeg` pré-compilado e o disponibiliza por meio
de uma função Python (`imageio_ffmpeg.get_ffmpeg_exe()`). O módulo
`audio_processor.py` localiza esse executável e o expõe no `PATH` do
processo antes de qualquer chamada ao Whisper — de forma transparente,
sem exigir nenhuma ação manual do usuário além de `uv add imageio-ffmpeg`.

Essa decisão segue a mesma filosofia da nota técnica da Etapa 2 (Seção
7.2): preservar a reprodutibilidade do ambiente inteiramente dentro do
gerenciador de pacotes do projeto (`uv`), evitando dependências de
sistema que comprometeriam a portabilidade entre máquinas e o deploy em
ambiente de nuvem (Streamlit Cloud).

### 9.3 Geração do áudio sintético (`src/audio/synthetic_audio.py`)

Sintetiza 12 frases de triagem médica em arquivos `.wav` individuais,
variando dois parâmetros do motor de TTS por frase:

| Perfil        | Velocidade (palavras/min) | Pausa extra entre palavras | Frases |
|------------------|------------------------------|-------------------------------|----------|
| `normal`         | 180                          | 0 ms                           | 6        |
| `fala_lenta`      | 90 (metade da velocidade)    | 350 ms                         | 3        |
| `fala_rapida`     | 320 (quase o dobro)          | 0 ms                           | 3        |

O módulo inclui também uma rotina de geração de **tons sintéticos**
(`_generate_fallback_tone_audio()`), ativada automaticamente caso o
motor de TTS do sistema não esteja disponível (cenário usado para testes
de infraestrutura em ambiente Linux sem `espeak`, mas sem produzir fala
real). Em ambiente Windows com SAPI5 — como o utilizado neste projeto —
a geração de fala real ocorre normalmente.

Saídas: `data/raw/audio_segments/frase_NN.wav` (um por frase) e
`data/raw/audio_metadata.json` (gabarito com texto original, tipo de
anomalia, parâmetros de TTS usados).

### 9.4 Transcrição e extração de métricas (`src/audio/audio_processor.py`)

Cada segmento de áudio é transcrito localmente com o modelo **Whisper
"base"** (OpenAI, execução 100% local — nenhum áudio é enviado a
serviços externos), forçando o idioma português para evitar
autodetecção incorreta em frases curtas.

A partir da transcrição e do áudio bruto, são extraídas quatro métricas
por segmento:

| Métrica               | Cálculo                                                | Relevância clínica                       |
|-------------------------|-----------------------------------------------------------|---------------------------------------------|
| `duration_seconds`       | Duração total do arquivo de áudio                          | Linha de base temporal                      |
| `n_words_transcribed`    | Contagem de palavras no texto transcrito                   | Verificação de completude da transcrição    |
| `words_per_second`       | Palavras transcritas ÷ duração                              | Velocidade de fala — indicador central      |
| `silence_ratio`          | Proporção de janelas de 50ms com energia abaixo do limiar  | Proxy de pausas/hesitação na fala           |

**Resultado da transcrição (execução real, sem dados simulados).** O
Whisper reproduziu o texto original das 12 frases com fidelidade muito
alta — por exemplo, *"Estou sentindo dor no peito"* foi transcrito como
*"Estou sentindo dor num peito"*, uma variação fonética mínima e
esperada, sem impacto na contagem de palavras ou nas métricas de
velocidade. As métricas extraídas confirmaram separação estatística
clara entre os três perfis:

| Perfil        | Velocidade média (palavras/s) | Silêncio médio |
|------------------|----------------------------------|-------------------|
| `normal`          | ≈ 1,7                            | ≈ 45%             |
| `fala_lenta`       | ≈ 0,6                            | ≈ 60%             |
| `fala_rapida`      | ≈ 2,8                            | ≈ 34%             |

### 9.5 Detecção de anomalias de fala (`src/audio/anomaly_detector.py`)

Dois métodos combinados por OR lógico, seguindo a mesma filosofia das
Etapas 1 e 2 (priorizar recall em contexto clínico):

**1. Z-score sobre velocidade de fala.** Calcula o desvio estatístico de
`words_per_second` em relação à média do conjunto de segmentos, marcando
como anômalo qualquer valor com `|z| > 1.5` — sensível tanto a fala
anormalmente lenta quanto anormalmente rápida.

**2. Regras clínicas.** Duas regras complementares:
   - Fala lentificada: `silence_ratio > 0.40` **e** `words_per_second <
     1.0` — a combinação de ambos os critérios distingue hesitação
     patológica de uma pausa natural entre frases;
   - Fala acelerada: `words_per_second` acima de um limiar adaptativo
     (média + 1 desvio padrão do conjunto).

**Resultados obtidos (execução real, sem dados simulados):**

| Método              | Anomalias injetadas | Detectadas | Precisão | Recall | F1-Score |
|------------------------|------------------------|----------------|--------------|------------|--------------|
| Z-score                | 6                      | 1              | 1.000        | 0.167      | 0.286        |
| Regras clínicas        | 6                      | 6              | 1.000        | 1.000      | 1.000        |
| **Combinado (OR)**     | 6                      | 6              | **1.000**    | **1.000**  | **1.000**    |

**Discussão.** As regras clínicas, calibradas especificamente para o
comportamento esperado de cada perfil de fala, atingiram **desempenho
perfeito** — todas as 6 anomalias injetadas (3 de fala lenta, 3 de fala
rápida) foram corretamente identificadas, sem nenhum falso positivo
entre os 6 segmentos normais. O z-score isolado teve recall baixo
(0,167) porque o limiar de 1,5 desvios padrão, aplicado sobre um
conjunto pequeno (12 segmentos) com alta variância entre os dois tipos
opostos de anomalia (lenta vs. rápida), dilui a sensibilidade estatística
— uma limitação conhecida de z-score em amostras pequenas e
heterogêneas. A combinação OR preserva o resultado perfeito das regras
clínicas, validando a abordagem de múltiplos detectores complementares
adotada consistentemente desde a Etapa 1.

Este resultado (F1 = 1.000) é o melhor entre os três módulos de
detecção implementados até o momento, refletindo a maior clareza do
sinal estatístico em métricas de fala (velocidade, silêncio) em
comparação aos ângulos articulares da Etapa 2, cujas anomalias se
desenrolam de forma mais gradual ao longo de múltiplos frames.

### 9.6 Visualização (`src/audio/plotter.py`)

Duas saídas visuais, seguindo o padrão estabelecido nas etapas
anteriores:

- **Painel de métricas** (`audio_metrics_panel.png`): dois subplots —
  velocidade de fala e proporção de silêncio por segmento — com
  marcadores de anomalia detectada (X vermelho) e gabarito (círculo
  laranja vazado);
- **Waveforms anotados** (`audio_waveforms/waveform_segment_NN.png`):
  forma de onda de quatro segmentos representativos (normais e
  anômalos), com o texto transcrito e a velocidade de fala sobrepostos
  no título do gráfico.

Ambas as saídas são gravadas em disco sem qualquer dependência de
reprodução de áudio ou display gráfico, mantendo a compatibilidade com
execução headless exigida pelo deploy no Streamlit Cloud.

![Painel de métricas de fala](imagens/audio_metrics_panel.png)

### 9.7 Síntese comparativa dos três módulos de detecção

| Módulo            | Método combinado | Precisão | Recall | F1-Score |
|----------------------|----------------------|--------------|------------|--------------|
| Sinais vitais (Etapa 1) | Z-score + Isolation Forest | 0.667    | 1.000      | 0.800        |
| Vídeo/postura (Etapa 2) | Z-score + Regras clínicas  | 1.000    | 0.567      | 0.723        |
| Áudio/fala (Etapa 3)    | Z-score + Regras clínicas  | 1.000    | 1.000      | 1.000        |

Os três módulos confirmam a validade da estratégia consistente adotada
em todo o projeto: combinação de método estatístico (z-score) com
conhecimento de domínio (regras clínicas ou Isolation Forest), priorizando
recall sobre precisão sempre que esses dois objetivos entram em conflito.
A variação no F1-Score entre módulos reflete a natureza distinta de cada
sinal — eventos pontuais e extremos (vitais) favorecem alta precisão;
processos graduais (postura) desafiam o recall; sinais com separação
estatística clara entre classes (fala) permitem desempenho ótimo nos
dois eixos.

---

## 9. Etapa 4 — Camada de Integração com Groq API (LLM)

### 9.1 Contexto e papel desta etapa

Diferente das Etapas 1-3, que produzem dados sintéticos e os analisam com
métodos estatísticos/determinísticos, a Etapa 4 introduz uma **camada de
IA generativa** que consome os textos já produzidos pela Etapa 3
(transcrição de fala) e cruza os resultados das três modalidades em uma
interpretação clínica de alto nível — papel equivalente ao Azure Text
Analytics, AWS Comprehend e à sumarização de laudos sugeridos no desafio
original (ver Tabela 1, Seção 2).

A substituição adotada usa a **Groq API** com modelos de linguagem
open-weight (família `gpt-oss`, da OpenAI, hospedada em hardware LPU da
Groq), mantendo compatibilidade estrutural com o SDK da OpenAI
(`client.chat.completions.create`) — o mesmo padrão de chamada ensinado
nas aulas do curso sobre a API do GPT.

Duas tarefas distintas, dois modelos distintos:

| Tarefa                                  | Modelo usado          | Justificativa                                  |
|---------------------------------------------|---------------------------|-----------------------------------------------------|
| Extração de termos críticos e classificação | `openai/gpt-oss-20b`      | Tarefa estruturada e simples — prioriza velocidade e custo |
| Sumarização clínica multimodal              | `openai/gpt-oss-120b`     | Exige raciocínio sobre múltiplas fontes de evidência simultâneas |

### 9.2 Nota técnica — depreciação de modelos durante o desenvolvimento

Em 17 de junho de 2026 — durante o desenvolvimento deste projeto — a Groq
anunciou a depreciação dos modelos originalmente planejados
(`llama-3.3-70b-versatile` e `llama-3.1-8b-instant`), recomendando
migração para a família `gpt-oss`. Este é um exemplo real de um risco
inerente a qualquer integração com API de terceiros em rápida evolução:
o modelo certo hoje pode não estar disponível na próxima sprint.

Duas decisões de design mitigam esse risco:

1. **Nomes de modelo nunca hardcoded no código.** As variáveis
   `GROQ_MODEL_FAST` e `GROQ_MODEL_SMART` são lidas do arquivo `.env`
   (com fallback para os valores vigentes no código, caso as variáveis
   não estejam definidas). Uma futura depreciação exige apenas a
   atualização de uma linha de configuração, sem alterar nenhum
   arquivo `.py`.
2. **Falhas da API nunca travam o pipeline.** A função `call_groq()`
   (`src/llm/groq_client.py`) nunca levanta exceção — em caso de erro,
   retorna um objeto `GroqResponse` com `success=False` e a mensagem de
   erro, permitindo que módulos consumidores (como o motor de fusão da
   Etapa 5) degradem graciosamente em vez de travar todo o sistema caso
   a API esteja indisponível.

### 9.3 Nota técnica — tokens de raciocínio dos modelos `gpt-oss`

Durante os testes, identificou-se um comportamento relevante para
qualquer integração com modelos de raciocínio: a família `gpt-oss`
consome parte do orçamento de `max_tokens` em um campo interno de
`reasoning` (observado entre ~30 e ~130 tokens nas chamadas deste
projeto) **antes** de gerar a resposta final visível. Com um limite de
tokens baixo, é possível que o modelo gaste todo o orçamento racionando
e retorne `success=True` com **conteúdo vazio** — um modo de falha
silenciosa, já que não é sinalizado como erro pela API.

Esse comportamento foi observado de forma reprodutível: a frase
*"Minha visão está embaçada"* retornou conteúdo vazio com `max_tokens=200`,
mesmo com chamadas estruturalmente idênticas a outras frases tendo
sucesso. A correção adotada em `src/llm/text_analyzer.py` foi dupla:

1. Aumentar o orçamento de tokens para um valor com margem de segurança
   generosa (400, e 700 na sumarização, que produz textos mais longos);
2. Implementar uma segunda tentativa automática com orçamento ainda
   maior (600) especificamente para o caso de resposta vazia — sem
   mascarar falhas reais de API, que continuam sendo reportadas
   normalmente em `llm_error`.

Esse ajuste eliminou completamente o problema nos testes subsequentes
(Seção 9.5).

### 9.4 Prompts e extração estruturada (`src/llm/prompts.py`, `text_analyzer.py`)

Os prompts são centralizados em um único módulo (`prompts.py`) — boa
prática que facilita auditoria e ajuste fino sem precisar localizar
strings espalhadas pelo código.

**Prompt de análise de texto.** Solicita ao modelo `gpt-oss-20b` que
responda exclusivamente em JSON estrito, extraindo:
   - `termos_criticos`: sintomas, partes do corpo ou condições mencionadas;
   - `sentimento`: positivo, neutro ou negativo;
   - `nivel_urgencia`: baixo, médio ou alto — com critério explícito no
     prompt distinguindo sintomas potencialmente graves (dor no peito,
     falta de ar, fraqueza súbita) de queixas moderadas ou neutras;
   - `justificativa`: explicação breve, útil para auditoria humana da
     decisão do modelo.

Como modelos de linguagem ocasionalmente envolvem o JSON em texto
explicativo ou marcadores de código (` ```json `), o módulo implementa
um parser tolerante (`_parse_json_response()`) que extrai o primeiro
objeto JSON válido da resposta via expressão regular, com fallback
seguro (valores neutros) em caso de falha total de parsing.

### 9.5 Resultado da análise de texto (execução real, sem dados simulados)

Aplicado às 12 frases transcritas pelo Whisper na Etapa 3, o modelo
`gpt-oss-20b` produziu classificações coerentes com o conteúdo clínico
de cada frase:

| Frase (transcrita)                | Sentimento | Urgência | Termos críticos        |
|---------------------------------------|----------------|--------------|------------------------------|
| "Estou me sentindo bem hoje"          | positivo       | baixo        | —                             |
| "Estou sentindo dor num peito"        | negativo       | **alto**     | dor, peito                   |
| "Minha visão está embaçada"           | negativo       | médio        | visão embaçada                |
| "Sinto uma fraqueza no braço"         | neutro         | médio        | fraqueza, braço               |
| "Estou muito ansioso e agitado"       | negativo       | médio        | ansioso, agitado              |
| "Preciso de ajuda agora mesmo"        | negativo       | **alto**     | —                             |
| "Estou calmo e tranquilo agora"       | positivo       | baixo        | —                             |

O modelo classificou corretamente como urgência **alta** as duas frases
com indicação clínica mais grave (dor no peito; pedido explícito de
ajuda), e como urgência **média** as frases de sintoma moderado
(visão embaçada, fraqueza, agitação) — validando a adequação do prompt
ao critério clínico desejado sem qualquer ajuste fino adicional além da
instrução em linguagem natural.

### 9.6 Sumarização clínica multimodal (`src/llm/summarizer.py`)

O módulo `summarizer.py` constrói três resumos textuais concisos — um
por modalidade — a partir dos DataFrames já processados nas Etapas 1, 2
e 3:

- `build_vitals_summary_text()`: total de amostras, anomalias
  detectadas, faixa de valores nos pontos anômalos, métricas do
  detector combinado;
- `build_posture_summary_text()`: total de frames, anomalias
  detectadas, padrões posturais identificados (assimetria, colapso de
  tronco, hiperextensão), métricas do detector;
- `build_speech_summary_text()` (em `text_analyzer.py`): total de
  frases, anomalias de velocidade/pausa, termos críticos agregados,
  distribuição de níveis de urgência.

Esses três resumos — não os dados brutos completos — são enviados ao
modelo `gpt-oss-120b` (Seção 9.1), mantendo o prompt conciso e reduzindo
custo/latência sem perda de informação relevante para a síntese clínica.

**Resultado obtido (execução real, sem dados simulados):**

> **Resumo Executivo**
>
> 1. **Risco clínico:** Há indícios de risco prioritário. O detector de
> sinais vitais identificou 9 anomalias (HR 37-152 bpm, SpO₂ 83-99%,
> pressão sistólica 74-177 mmHg) com recall = 1.0, indicando que todas
> as ocorrências foram capturadas. Na fala, 1 frase foi classificada
> como de urgência alta, contendo termos como "dor", "peito" e
> "fraqueza".
>
> 2. **Modalidades com anomalias e possíveis relações:** Sinais vitais
> — variações extremas de frequência cardíaca e saturação sugerem
> instabilidade hemodinâmica. Postura — 34/300 frames mostram
> assimetria bilateral dos braços, inclinação de tronco e possíveis
> hiperextensões, compatíveis com dor ou fraqueza muscular. Fala —
> alterações de velocidade/pausas e vocabulário ansioso reforçam a
> percepção de desconforto físico. A coincidência de dor/fraqueza
> relatada na fala com anomalias posturais e instabilidade
> cardiovascular aponta para um quadro agudo possivelmente
> neuromuscular ou cardiovascular.
>
> 3. **Recomendação:** Realizar avaliação médica imediata focada em
> suporte hemodinâmico (monitoramento contínuo, oxigenação) e exame
> neurológico/musculoesquelético; considerar ECG, exames de sangue
> (troponina, eletrólitos) e revisão da terapia fisioterápica.
> Priorizar intervenção antes de prosseguir com sessões de reabilitação.

**Discussão.** O resultado demonstra o valor agregado da camada de LLM
em relação aos detectores estatísticos isolados das Etapas 1-3: em vez
de apenas listar anomalias por modalidade, o modelo **conectou** os
achados — relacionando a dor/fraqueza relatada na fala com os padrões
posturais de assimetria e hiperextensão, e com a instabilidade
hemodinâmica dos sinais vitais — produzindo uma interpretação clínica
unificada e uma recomendação de próximo passo objetiva. Esse é
precisamente o tipo de síntese que, no fluxo de trabalho real de uma
equipe médica, economiza tempo de triagem ao consolidar sinais
dispersos em múltiplos monitores em um único parecer.

É importante registrar que o modelo foi instruído a basear-se
exclusivamente nos dados fornecidos (Seção 9.4) — a interpretação
apresentada é uma síntese das anomalias estatísticas já detectadas
pelos módulos determinísticos das Etapas 1-3, não um diagnóstico gerado
de forma independente pelo LLM. Essa distinção é relevante para a
seção de considerações éticas do relatório final (Etapa 7).

---

## 10. Próximas Etapas

- ~~Etapa 2: Análise de vídeo com MediaPipe Pose~~ ✅ **Concluída**
- ~~Etapa 3: Transcrição e análise de áudio com Whisper local~~ ✅ **Concluída**
- ~~Etapa 4: Integração com Groq API para extração de termos críticos,
  sentimento e sumarização~~ ✅ **Concluída**
- Etapa 5: Motor de fusão multimodal e geração de alertas;
- Etapa 6: Interface Streamlit;
- Etapa 7: Consolidação final do relatório técnico;
- Etapa 8: Deploy no Streamlit Cloud.

---

*Este relatório é um documento vivo, atualizado incrementalmente conforme
o desenvolvimento avança.*
