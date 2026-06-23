"""
plotter.py  (src/alerts/)
==========================
Gera o dashboard consolidado do motor de fusão multimodal: uma única
visualização que resume o status das três modalidades monitoradas e o
nível de alerta final — a "tela única" que a equipe médica veria.

Sem display gráfico — salvo em arquivo, mesmo padrão das Etapas 1-3.

Nota de implementação — quebra de linha manual:
    O matplotlib não quebra linha automaticamente em texto posicionado
    por coordenadas de eixo (`ax.text`), mesmo com `wrap=True`, quando o
    texto não tem uma largura de bounding box previamente definida. Como
    os textos reais deste projeto variam em tamanho (de uma frase curta
    a um resumo de ~130 palavras gerado pelo LLM), a quebra de linha é
    feita manualmente via `textwrap.fill()`, e a altura dos elementos é
    estimada a partir do número de linhas resultante — evitando texto
    cortado ou extrapolando a caixa, problema observado e corrigido
    durante o desenvolvimento (ver relatório técnico).

Saída:
    data/processed/fusion_dashboard.png
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.alerts.alert_rules import AlertDecision, AlertLevel


# ── Paleta de cores por nível de alerta ───────────────────────────────────────
ALERT_COLORS = {
    AlertLevel.NORMAL: "#4CAF50",
    AlertLevel.ATENCAO: "#FF9800",
    AlertLevel.CRITICO: "#F44336",
}

ALERT_LABELS = {
    AlertLevel.NORMAL: "NORMAL",
    AlertLevel.ATENCAO: "ATENÇÃO",
    AlertLevel.CRITICO: "CRÍTICO",
}

BACKGROUND = "#F8F9FA"
MODALITY_OK_COLOR = "#2196F3"
MODALITY_ANOMALY_COLOR = "#F44336"

# Larguras de quebra de linha (em caracteres), calibradas para os tamanhos
# de fonte usados em cada elemento do dashboard.
WRAP_WIDTH_CARD_DETAIL = 28
WRAP_WIDTH_BODY_TEXT = 130


def plot_fusion_dashboard(
    decision: AlertDecision,
    clinical_summary: str = "",
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Gera o dashboard consolidado: painel de nível de alerta + status de
    cada modalidade + (se disponível) o resumo clínico do LLM.

    A altura da figura e do painel de texto inferior é calculada
    dinamicamente a partir do volume de texto a exibir, evitando corte
    de conteúdo quando o resumo clínico do LLM é extenso.

    Parâmetros:
        decision        : AlertDecision retornada pelo motor de fusão
        clinical_summary: texto do resumo clínico (Etapa 4), opcional
        save_path        : caminho para salvar a imagem
    """
    alert_color = ALERT_COLORS[decision.level]
    alert_label = ALERT_LABELS[decision.level]

    # ── Pré-calcula o texto do painel inferior, já quebrado em linhas ────────
    reasoning_wrapped = textwrap.fill(decision.reasoning, width=WRAP_WIDTH_BODY_TEXT)
    body_lines = ["Justificativa da decisão:"] + reasoning_wrapped.split("\n")

    if clinical_summary:
        body_lines.append("")
        body_lines.append("Resumo clínico (gerado por LLM):")
        # Preserva quebras de parágrafo originais do LLM, quebrando cada
        # uma à largura definida, em vez de tratar o resumo como um bloco
        # único de texto (que perderia a estrutura de itens numerados).
        for paragraph in clinical_summary.split("\n"):
            if not paragraph.strip():
                body_lines.append("")
                continue
            wrapped_paragraph = textwrap.fill(paragraph, width=WRAP_WIDTH_BODY_TEXT)
            body_lines.extend(wrapped_paragraph.split("\n"))

    n_body_lines = len(body_lines)

    # ── Calcula a altura total da figura dinamicamente ────────────────────────
    # Altura fixa para banner + cartões (~5.5"), mais altura variável para o
    # texto (cada linha ocupa ~0.22" a 9pt, com margem de segurança).
    fixed_height = 5.5
    text_height = max(2.5, n_body_lines * 0.235 + 0.6)
    total_height = fixed_height + text_height

    fig = plt.figure(figsize=(13, total_height))
    fig.patch.set_facecolor(BACKGROUND)

    # Frações de altura recalculadas para a figura total.
    # Posições calculadas de cima para baixo, em polegadas absolutas,
    # depois convertidas para fração — evita o espaço residual que
    # sobrava entre os cartões e o painel de texto quando o texto era
    # curto e a margem inferior não absorvia corretamente.
    margin_top = 0.15
    gap_banner_cards = 0.35
    gap_cards_text = 0.25
    margin_bottom = 0.15

    banner_h = 0.9
    cards_h = 2.4

    banner_h_frac = banner_h / total_height
    cards_h_frac = cards_h / total_height
    text_h_frac = text_height / total_height

    banner_y = 1 - margin_top / total_height - banner_h_frac
    cards_y = banner_y - gap_banner_cards / total_height - cards_h_frac
    text_y = margin_bottom / total_height

    # Se sobrar espaço entre os cards e o texto (texto mais curto que o
    # espaço disponível), o texto é ancorado IMEDIATAMENTE após os cards
    # em vez de ficar fixo na base da figura — elimina o vão em branco.
    text_top_y = cards_y - gap_cards_text / total_height
    text_y = max(margin_bottom / total_height, text_top_y - text_h_frac)
    text_h_frac = text_top_y - text_y

    # ── Painel superior: nível de alerta (banner) ─────────────────────────────
    ax_banner = fig.add_axes([0.03, banner_y, 0.94, banner_h_frac])
    ax_banner.set_facecolor(alert_color)
    ax_banner.set_xticks([])
    ax_banner.set_yticks([])
    for spine in ax_banner.spines.values():
        spine.set_visible(False)
    ax_banner.text(
        0.5, 0.5, f"NÍVEL DE ALERTA: {alert_label}",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color="white", transform=ax_banner.transAxes,
    )

    # ── Painel de modalidades (cartões lado a lado) ───────────────────────────
    n_mod = len(decision.modalities)
    gap = 0.02
    card_width = (0.94 - gap * (n_mod - 1)) / n_mod

    for i, modality in enumerate(decision.modalities):
        x_pos = 0.03 + i * (card_width + gap)
        ax_card = fig.add_axes([x_pos, cards_y, card_width, cards_h_frac])

        color = MODALITY_ANOMALY_COLOR if modality.has_anomaly else MODALITY_OK_COLOR
        ax_card.set_facecolor(BACKGROUND)
        ax_card.set_xlim(0, 1)
        ax_card.set_ylim(0, 1)
        ax_card.set_xticks([])
        ax_card.set_yticks([])

        rect = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            linewidth=2.5, edgecolor=color, facecolor="white",
            transform=ax_card.transAxes,
        )
        ax_card.add_patch(rect)

        icon = "⚠" if modality.has_anomaly else "✓"
        ax_card.text(
            0.5, 0.72, icon, ha="center", va="center", fontsize=26,
            color=color, transform=ax_card.transAxes,
        )
        ax_card.text(
            0.5, 0.48, modality.name.upper(), ha="center", va="center",
            fontsize=13, fontweight="bold", color="#333333",
            transform=ax_card.transAxes,
        )

        detail_wrapped = textwrap.fill(modality.detail, width=WRAP_WIDTH_CARD_DETAIL)
        ax_card.text(
            0.5, 0.22, detail_wrapped, ha="center", va="center",
            fontsize=8.5, color="#555555",
            transform=ax_card.transAxes,
        )

    # ── Painel inferior: justificativa + resumo clínico ───────────────────────
    ax_text = fig.add_axes([0.03, text_y, 0.94, text_h_frac])
    ax_text.set_facecolor("white")
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    for spine in ax_text.spines.values():
        spine.set_edgecolor("#CCCCCC")

    full_text = "\n".join(body_lines)
    ax_text.text(
        0.02, 0.97, full_text, ha="left", va="top", fontsize=9,
        color="#333333", transform=ax_text.transAxes,
        linespacing=1.5, family="sans-serif",
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
        print(f"Dashboard salvo em: {save_path}")

    return fig


if __name__ == "__main__":
    from src.alerts.alert_rules import ModalityStatus, decide_alert_level

    decision = decide_alert_level(
        modalities=[
            ModalityStatus("vitais", True, "9 amostra(s) com anomalia em sinais vitais."),
            ModalityStatus("postura", True, "34 frame(s) com anomalia postural detectada."),
            ModalityStatus("fala", True, "6 segmento(s) de fala com anomalia de velocidade/pausa."),
        ],
        llm_urgency="alto",
    )

    long_summary_demo = (
        "**Resumo Executivo (≈130 palavras)**\n"
        "1. **Risco clínico prioritário:** Há indícios de instabilidade "
        "hemodinâmica (HR 37-152 bpm, SpO2 83-99%, pressão sistólica "
        "74-177 mmHg) associados a queixas de dor, fraqueza, visão "
        "embaçada, peito e tremor.\n"
        "2. **Modalidades com anomalias:** Sinais vitais, postura e fala "
        "apresentaram anomalias simultâneas, sugerindo um processo "
        "sistêmico agudo.\n"
        "3. **Recomendação objetiva:** Realizar avaliação clínica "
        "imediata, com ECG, exames laboratoriais e exame neurológico "
        "focal. Considerar transferência para unidade de cuidados "
        "intensivos."
    )

    plot_fusion_dashboard(
        decision,
        clinical_summary=long_summary_demo,
        save_path=Path("data/processed/fusion_dashboard.png"),
    )