"""
Lab 1 - Financial Analytics

Série temporal
    Sequência de números indexada pelo tempo: Y_1, Y_2, ..., Y_t. No nosso
    caso, o IBC-Br mensal (uma observação por mês).

Estacionariedade
    Uma série é estacionária quando sua média, variância e estrutura de
    correlação NÃO mudam ao longo do tempo. Modelos clássicos (ARIMA/SARIMA)
    só funcionam sobre séries estacionárias. Por isso a maior parte do
    trabalho é transformar a série até ela ficar estacionária.

ACF e PACF
    ACF (autocorrelation function) no lag k = correlação entre Y_t e Y_{t-k}.
    Mede quanto a série "lembra" do passado, considerando todo o caminho.
    PACF (partial autocorrelation function) no lag k = correlação entre Y_t e
    Y_{t-k} DEPOIS de descontar o efeito dos lags intermediários. Mede a
    correlação direta. Olhando esses dois gráficos a gente "lê" a estrutura
    da série e propõe um modelo (metodologia Box-Jenkins).

SARIMA(p,d,q)(P,D,Q)_s
    Sete números que definem o modelo:
        p = quantos lags da própria série entram (parte AR não-sazonal)
        d = quantas diferenças regulares foram aplicadas (Y_t - Y_{t-1})
        q = quantos lags do erro entram (parte MA não-sazonal)
        P, D, Q = mesma ideia, mas no nível sazonal
        s = período da sazonalidade (12 para dados mensais com ciclo anual)

Tarefas
--------------------
    1. Carrega uma série do BCB via bcb_serie() (wrapper sobre bcb.sgs.get).
    2. Plota a série, ACF e PACF; discute tendência e sazonalidade.
    3. Decompõe a série (STL) em tendência, sazonalidade e resíduo.
    4. Aplica diferenças (regular e/ou sazonal) para estacionarizar.
    5. Verifica com o teste ADF (varredura de lags) e KPSS de apoio.
    6. Usa ACF/PACF para propor manualmente um SARIMA(p,d,q)(P,D,Q)_s,
       compara candidatos por AIC/BIC e confirma com AutoARIMA.
    7. Estima e avalia o diagnóstico dos resíduos.

Série usada: IBC-Br SEM ajuste sazonal (SGS 24363), proxy mensal do PIB,
com tendência clara e sazonalidade anual forte (s = 12).

Como rodar:
    cd /Users/akatsurada/Documents/INSPER/financial_analytics/Lab1
    uv sync                              # se for a 1a vez (instala seaborn etc.)
    uv run python lab1_timeseries_stuff.py

O script:
    - NÃO abre janelas (não bloqueia o terminal).
    - Salva todos os PNGs em ./figs/ na ordem do pipeline (01_, 02_, ...).
    - Imprime no terminal o "como ler" antes de cada figura ser salva.

No fim, abra a pasta figs/ e folheie os PNGs em ordem alfabética:
    open ./figs           # macOS
    xdg-open ./figs       # linux

Os links da documentação de cada lib/API estão INLINE no código, ao lado de
cada import e dentro do docstring de cada função (procure por "docs:" ou
"Documentação").
"""

from __future__ import annotations

import warnings
from pathlib import Path

# Backend "Agg" = non-interactive. Não abre janela, só renderiza em arquivo.
# Tem que ser SETADO antes de importar pyplot, senão não pega.
# docs: https://matplotlib.org/stable/users/explain/figure/backends.html
import matplotlib
matplotlib.use("Agg")

# docs: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
import matplotlib.pyplot as plt
# docs: https://numpy.org/doc/stable/
import numpy as np
# docs: https://pandas.pydata.org/docs/
import pandas as pd
# docs: https://wilsonfreitas.github.io/python-bcb/sgs.html
from bcb import sgs
# docs: https://docs.scipy.org/doc/scipy/reference/stats.html
from scipy import stats
# docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
from scipy.stats import norm

# qqplot:   https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html
from statsmodels.graphics.gofplots import qqplot
# plot_acf: https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
# plot_pacf: https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_pacf.html
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Dois testes estatísticos sobre os resíduos do modelo:
#   acorr_ljungbox -> "ainda sobrou autocorrelação nos resíduos?"
#       docs: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
#   het_arch       -> "a variância do resíduo varia no tempo? (clusters de
#                      volatilidade tipo crise)"
#       docs: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# STL = Seasonal-Trend decomposition using Loess. Algoritmo que separa uma
# série em três componentes (tendência + sazonalidade + resíduo). Bom para
# VISUALIZAR o que está dentro dos dados antes de modelar.
# docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html
# teoria: https://otexts.com/fpp3/stl.html
from statsmodels.tsa.seasonal import STL

# A classe que estima modelos SARIMA. O "X" no final indica que ela aceita
# variáveis exógenas (regressores extras), mas aqui não vamos usar.
# docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
# teoria: https://otexts.com/fpp3/seasonal-arima.html
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Dois testes de estacionariedade com hipóteses INVERTIDAS:
#   adfuller (ADF):  H0 = tem raiz unitária (não-estacionária)
#                    rejeitar H0 (p < 0.05) é BOM (estacionária).
#       docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
#   kpss:            H0 = é estacionária
#                    NÃO rejeitar H0 é BOM. Usar os dois juntos dá
#                    confirmação cruzada.
#       docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
# teoria do par ADF+KPSS: https://otexts.com/fpp3/stationarity.html#unit-root-tests
from statsmodels.tsa.stattools import adfuller, kpss

# AutoARIMA é uma busca automática da melhor ordem (p,d,q,P,D,Q) por AICc.
# É opcional: se o usuário não tiver statsforecast instalado, o script segue
# sem quebrar, apenas pula a confirmação automática.
# docs StatsForecast: https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html
# docs AutoARIMA:     https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoarima
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    HAS_STATSFORECAST = True
except ImportError:
    HAS_STATSFORECAST = False


warnings.filterwarnings("ignore")


# Seaborn = wrapper de estética em cima do matplotlib. Não muda o que a
# gente desenha, só como fica bonito (paleta, grid, tipografia).
# É opcional: se não estiver instalado, o script segue com o look padrão.
# docs: https://seaborn.pydata.org/
try:
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",      # fundo branco com grid leve
        context="notebook",     # tamanhos médios (bom pra script local)
        palette="muted",        # paleta com tons sóbrios
        font_scale=1.0,
        rc={
            "figure.figsize": (12, 4.5),
            "axes.spines.top": False,    # tira borda de cima
            "axes.spines.right": False,  # tira borda da direita
            "axes.titleweight": "bold",
        },
    )
    HAS_SEABORN = True
except ImportError:
    # Fallback: aplica só os rcParams essenciais sem seaborn.
    plt.rcParams.update(
        {
            "figure.figsize": (12, 4.5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )
    HAS_SEABORN = False

INSPER_RED = "#E50505"
INSPER_TURQUESA = "#3ACC9F"
INSPER_AMARELO = "#FFCC00"
INSPER_GRAY = "#5B5B5B"


# Pasta onde TODAS as figuras vão ser salvas. Fica do lado do script.
# Por que salvar em vez de plt.show()?
#   - script roda do começo ao fim sem travar pedindo "feche a janela".
#   - você abre a pasta no Finder e folheia tudo no seu ritmo.
#   - dá pra anexar em relatório / colar no Slack sem screenshot.
FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def salvar_figura(fig: plt.Figure, nome_arquivo: str) -> Path:
    """Salva a figura em FIGS_DIR, fecha e imprime o caminho.

    Por que salvar e não plt.show()?
        plt.show() bloqueia o script até vc fechar a janela. Em pipeline com
        20 gráficos isso é um inferno. Salvando, o script roda end-to-end e
        vc abre a pasta figs/ no Finder no fim.

    Por que fechar (plt.close)?
        Se a gente só salva sem fechar, o matplotlib mantém todas as figuras
        em memória. Em scripts longos isso vaza memória e gera warning.

    Espaçamento:
        Aplicamos margens generosas (top=0.85, bottom=0.18) pra dar AR no
        título e nas datas do eixo X. Sem isso o título cola no topo e as
        datas ficam apertadas embaixo.
        bbox_inches='tight' + pad_inches=0.4 ainda adiciona uma BORDA externa
        no PNG, então quando vc encostar dois charts no Finder ou colar num
        relatório, eles não vão se tocar.
    """
    caminho = FIGS_DIR / nome_arquivo
    # Margens internas generosas. Top alto = espaço pro título não colar.
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.10, right=0.96)
    fig.savefig(
        caminho,
        dpi=130,
        bbox_inches="tight",
        pad_inches=0.4,        # borda externa em volta da figura
        facecolor="white",     # fundo branco mesmo se o tema mudar
    )
    plt.close(fig)
    print(f"        => salvo em figs/{nome_arquivo}")
    return caminho


def explicar_grafico(titulo: str, *paragrafos: str) -> None:
    """Imprime um banner explicativo antes do gráfico ser gerado.

    Por que existe?
        Cada gráfico do pipeline diz uma coisa específica. Se vc só abre o
        PNG sem contexto, perde metade do valor. Aqui a gente imprime ANTES
        de salvar: o que o gráfico mostra e como ler.
    """
    largura = 72
    print()
    print(f"[CHART] {titulo}")
    print("-" * largura)
    for p in paragrafos:
        print(p)
    print("-" * largura)


def bcb_serie(
    codigo: int,
    nome: str,
    start: str = "2003-01-01",
    end: str | None = None,
) -> pd.Series:
    """Baixa uma série temporal do SGS/BCB e devolve uma pd.Series mensal.

    Por que essa função existe?
        Toda série do BCB tem um código numérico (ex.: 433 = IPCA, 24363 =
        IBC-Br). Esse wrapper recebe o código, baixa via API e monta uma
        Series com índice de datas regular (frequência mensal "MS") e sem
        valores faltantes - pronta para ir direto para o modelo.

    Parâmetros
    ----------
    codigo : int
        Código SGS da série (ex.: 24363 para IBC-Br sem ajuste sazonal).
    nome : str
        Label amigável para a série (vira o .name do objeto retornado).
    start, end : str
        Datas inicial e final, formato "YYYY-MM-DD". end=None significa
        "até o último valor disponível".

    Retorna
    -------
    pd.Series com índice DatetimeIndex em frequência mensal "MS"
    (Month Start, primeiro dia de cada mês), valores float, sem NaN.

    Documentação
    ------------
    bcb.sgs.get  : https://wilsonfreitas.github.io/python-bcb/sgs.html
    Catálogo SGS : https://www3.bcb.gov.br/sgspub/
    asfreq       : https://pandas.pydata.org/docs/reference/api/pandas.Series.asfreq.html
    interpolate  : https://pandas.pydata.org/docs/reference/api/pandas.Series.interpolate.html
    """
    # Chamada HTTP à API do BCB. O dicionário {nome: codigo} diz "baixe o
    # código X e nomeie a coluna como Y". Volta um DataFrame.
    df = sgs.get({nome: codigo}, start=start, end=end)

    # Encadeamento de três operações:
    #   df[nome]        -> extrai a coluna como Series
    #   .astype(float)  -> garante tipo numérico (a API às vezes manda string)
    #   .sort_index()   -> ordena por data (defensivo)
    s = df[nome].astype(float).sort_index()

    # Garante DatetimeIndex de verdade (não string). Operações de séries
    # temporais (diff, asfreq, etc.) só funcionam em datetime.
    s.index = pd.to_datetime(s.index)

    # .asfreq("MS")  -> força frequência mensal "Month Start"; cria índice
    #                   REGULAR; meses ausentes viram NaN.
    # .interpolate() -> preenche NaN por interpolação linear entre vizinhos.
    # Modelos SARIMA assumem espaçamento uniforme; se houver buracos, o
    # s=12 deixa de significar "1 ano".
    s = s.asfreq("MS").interpolate()

    s.name = nome
    return s


def chart_serie(
    serie: pd.Series,
    titulo: str,
    nome_arquivo: str,
    cor: str = INSPER_RED,
    ylabel: str = "Valor",
) -> None:
    """Plota UMA linha temporal e salva como PNG.

    Linha tracejada cinza marca a média da série como referência visual.
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(serie, color=cor, lw=1.2)
    # Linha horiz na média = referência visual rápida (acima/abaixo da média).
    ax.axhline(
        serie.mean(),
        color=INSPER_GRAY,
        ls="--",
        alpha=0.6,
        label=f"média = {serie.mean():.2f}",
    )
    ax.set_title(titulo, fontweight="bold", pad=14)
    ax.set_xlabel("Tempo", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.legend(loc="best", frameon=False)
    fig.autofmt_xdate()  # rotaciona/encolhe datas pra não sobrepor
    salvar_figura(fig, nome_arquivo)


def chart_acf(
    serie: pd.Series, titulo: str, nome_arquivo: str, lags: int = 48
) -> None:
    """Plota a ACF e salva como PNG.

    Como ler:
        Cada barra vertical = correlação entre Y_t e Y_{t-k} pra cada k.
        A área azul sombreada é a banda de confiança 95%.
          - Barra DENTRO da banda  -> não significativa (basicamente ruído).
          - Barra FORA   da banda  -> autocorrelação real, estatisticamente
                                      diferente de zero.

    Documentação
    ------------
    plot_acf : https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html
    teoria   : https://otexts.com/fpp3/acf.html
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))
    plot_acf(
        serie.dropna(),  # ACF não tolera NaN
        lags=lags,
        zero=False,  # esconde lag 0 (vale sempre 1, é inútil)
        ax=ax,
        color=INSPER_GRAY,
        vlines_kwargs={"colors": INSPER_GRAY},
    )
    # plot_acf seta um título genérico ("Autocorrelation"). Sobrescrevemos.
    ax.set_title(f"ACF — {titulo}", fontweight="bold", pad=14)
    ax.set_xlabel("Lag (defasagem em meses)", labelpad=10)
    ax.set_ylabel("Autocorrelação", labelpad=10)
    salvar_figura(fig, nome_arquivo)


def chart_pacf(
    serie: pd.Series, titulo: str, nome_arquivo: str, lags: int = 48
) -> None:
    """Plota a PACF e salva como PNG.

    PACF mede a correlação ENTRE Y_t e Y_{t-k} controlando por todos os
    lags intermediários. Útil pra identificar a ordem AR de um modelo.

    Documentação
    ------------
    plot_pacf : https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_pacf.html
    teoria    : https://otexts.com/fpp3/non-seasonal-arima.html#acf-and-pacf-plots
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))
    plot_pacf(
        serie.dropna(),
        lags=lags,
        zero=False,
        # method="ywm" = Yule-Walker modified, padrão recomendado.
        # Evita warning chato em séries pequenas.
        method="ywm",
        ax=ax,
        color=INSPER_GRAY,
        vlines_kwargs={"colors": INSPER_GRAY},
    )
    ax.set_title(f"PACF — {titulo}", fontweight="bold", pad=14)
    ax.set_xlabel("Lag (defasagem em meses)", labelpad=10)
    ax.set_ylabel("Autocorrelação parcial", labelpad=10)
    salvar_figura(fig, nome_arquivo)


def chart_componente_stl(
    componente: pd.Series,
    nome: str,
    cor: str,
    nome_arquivo: str,
    centrar_em_zero: bool = False,
) -> None:
    """Plota UM componente da decomposição STL e salva como PNG.

    Parâmetros
    ----------
    componente : pd.Series
        Série de algum componente (trend, seasonal, ou resid).
    nome : str
        Como chamar (vira título e label do eixo Y).
    cor : str
        Cor da linha.
    nome_arquivo : str
        Nome do PNG gerado em figs/.
    centrar_em_zero : bool
        Se True, desenha linha horizontal em y=0 (útil pra sazonalidade
        e resíduo, que oscilam em torno de zero).
    """
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(componente, color=cor, lw=1.1)
    if centrar_em_zero:
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title(nome, fontweight="bold", pad=14)
    ax.set_xlabel("Tempo", labelpad=10)
    ax.set_ylabel(nome, labelpad=10)
    fig.autofmt_xdate()
    salvar_figura(fig, nome_arquivo)


def decompor_stl(serie: pd.Series, period: int = 12, titulo: str = "") -> None:
    """Decomposição STL em tendência + sazonalidade + resíduo.

    Por que decompor?
        Para VISUALIZAR cada componente separadamente. Se vê tendência
        clara, isso justifica aplicar diferença regular (d=1). Se vê padrão
        sazonal estável, justifica diferença sazonal (D=1). A decomposição
        é a "motivação visual" para os parâmetros que vamos escolher depois.

    Por que STL e não a decomposição clássica?
        STL usa LOESS (regressão local) e é ROBUSTA a outliers - importante
        para séries macro brasileiras que têm crises violentas (2008, 2015,
        2020). A decomposição clássica seria muito distorcida por esses
        choques.

    Os componentes somam: trend + seasonal + resid ~ serie observada.

    Esta função gera 4 GRÁFICOS SEPARADOS, um por componente, cada um
    precedido de uma explicação no terminal sobre o que olhar.

    Documentação
    ------------
    STL classe : https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html
    Paper STL  : https://www.wessa.net/download/stl.pdf  (Cleveland et al., 1990)
    teoria     : https://otexts.com/fpp3/stl.html
    """
    res = STL(serie, period=period, robust=True).fit()
    sufixo = f" - {titulo}" if titulo else ""

    explicar_grafico(
        f"STL: Observado{sufixo}",
        "É a série original, repetida aqui só pra vc comparar visualmente",
        "com os 3 componentes que vêm a seguir. Pergunta-chave: olhando",
        "esse gráfico, dá pra enxergar tendência? E sazonalidade? Os",
        "próximos 3 PNGs vão ISOLAR cada coisa.",
    )
    chart_componente_stl(
        serie, f"Observado{sufixo}", INSPER_RED, "03a_stl_observado.png"
    )

    explicar_grafico(
        f"STL: Tendência{sufixo}",
        "É o componente de LONGO PRAZO da série, depois de remover ruído",
        "e sazonalidade. Se essa linha SOBE ou DESCE consistentemente, há",
        "tendência -> isso justifica usar d>=1 no SARIMA (diferenciação",
        "regular para remover a tendência).",
        "Quebras visíveis aqui (queda em 2009, 2015, 2020) refletem crises.",
    )
    chart_componente_stl(
        res.trend, f"Tendência{sufixo}", INSPER_TURQUESA, "03b_stl_tendencia.png"
    )

    explicar_grafico(
        f"STL: Sazonalidade{sufixo}",
        "É o padrão que se REPETE a cada 12 meses (period=12). Oscila em",
        "torno de zero por construção. A AMPLITUDE (distância pico-vale)",
        "diz quão forte é a sazonalidade. Se vc vê um padrão limpo de altos",
        "e baixos repetindo ano após ano -> justifica D>=1 (dif. sazonal).",
    )
    chart_componente_stl(
        res.seasonal,
        f"Sazonalidade (período = {period}){sufixo}",
        INSPER_AMARELO,
        "03c_stl_sazonalidade.png",
        centrar_em_zero=True,
    )

    explicar_grafico(
        f"STL: Resíduo{sufixo}",
        "É o que sobra depois de remover tendência e sazonalidade. Se o",
        "STL tá fazendo um bom trabalho, o resíduo deve PARECER RUÍDO, sem",
        "padrão visível. Se vc ainda enxerga tendência ou ciclos aqui, a",
        "decomposição não capturou tudo (ou a série é complicada demais",
        "pra um modelo aditivo simples).",
    )
    chart_componente_stl(
        res.resid,
        f"Resíduo{sufixo}",
        INSPER_GRAY,
        "03d_stl_residuo.png",
        centrar_em_zero=True,
    )


def chart_serie_diferenciada(
    serie_d: pd.Series, label_diff: str, descricao: str, nome_arquivo: str
) -> None:
    """Plota uma série diferenciada e salva como PNG.

    Série estacionária deve oscilar em torno de zero, sem deriva, com
    amplitude relativamente constante.

    `label_diff` deve vir em formato mathtext do matplotlib (entre $...$),
    ex: r"$\\Delta Y_t$". Sem isso o "_t" aparece literal e fica feio.
    """
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.plot(serie_d, color=INSPER_TURQUESA, lw=1.1)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title(f"{label_diff}   —   {descricao}", fontweight="bold", pad=14)
    ax.set_xlabel("Tempo", labelpad=10)
    ax.set_ylabel("Variação", labelpad=10)
    fig.autofmt_xdate()
    salvar_figura(fig, nome_arquivo)


def teste_estac(serie: pd.Series, nome: str) -> None:
    """Roda ADF + KPSS em uma série e imprime a interpretação humana.

    Os dois testes têm hipóteses INVERTIDAS, o que é útil:
        ADF:  H0 = tem raiz unitária (NÃO estacionária)
              p < 0.05  =>  REJEITA H0  =>  é estacionária (BOM)
        KPSS: H0 = é estacionária
              p < 0.05  =>  REJEITA H0  =>  NÃO é estacionária (RUIM)

    Se ambos concordarem (ADF rejeita E KPSS não rejeita) você está
    confiante na estacionariedade. Se discordarem, é zona cinza e vale
    investigar mais (talvez precise de mais diferenças, ou de uma
    transformação como log).

    Intuição do ADF:
        Estima Y_t = alpha + beta * Y_{t-1} + ... + epsilon_t e testa se
        beta = 1 (random walk, raiz unitária) ou beta < 1 (puxa para a
        média, estacionária).

    Documentação
    ------------
    adfuller : https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
    kpss     : https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
    teoria   : https://otexts.com/fpp3/stationarity.html#unit-root-tests
    """
    print(f"[{nome}]")

    # autolag="AIC" deixa o teste decidir sozinho qtos lags incluir na
    # regressão auxiliar via AIC.
    # Volta tupla (stat, pvalor, lags_usados, n_obs, valores_críticos, ic).
    adf = adfuller(serie.dropna(), autolag="AIC")
    adf_stat, adf_p, adf_lags = adf[0], adf[1], adf[2]
    adf_critico_5pct = adf[4]["5%"]
    print(f"  ADF  : stat = {adf_stat:+.3f} | p-valor = {adf_p:.4f} | lags = {adf_lags}")
    # Tradução em português do que cada número quer dizer.
    veredito_adf = (
        "REJEITA H0 -> estacionária (BOM)"
        if adf_p < 0.05
        else "Não rejeita H0 -> raiz unitária (ainda precisa diferenciar)"
    )
    print(f"          veredito : {veredito_adf}")
    print(
        f"          intuição : stat {adf_stat:+.3f} vs. crítico 5% "
        f"{adf_critico_5pct:+.3f}; p={adf_p:.4f} é a prob. de ver esse stat"
    )
    print("                     se H0 fosse verdade. Quanto MENOR, mais forte a evidência.")

    # KPSS com regression="c" testa estacionariedade em torno de uma
    # CONSTANTE (média não-zero). Pra testar em torno de tendência linear,
    # usaria regression="ct".
    kp = kpss(serie.dropna(), regression="c", nlags="auto")
    kp_stat, kp_p = kp[0], kp[1]
    print(f"  KPSS : stat = {kp_stat:+.3f} | p-valor = {kp_p:.4f}")
    veredito_kpss = (
        "REJEITA H0 -> NÃO estacionária (RUIM)"
        if kp_p < 0.05
        else "Não rejeita H0 -> estacionária (BOM)"
    )
    print(f"          veredito : {veredito_kpss}")

    # Conclusão combinada dos dois testes (a parte mais útil).
    if adf_p < 0.05 and kp_p >= 0.05:
        combinado = "ADF e KPSS CONCORDAM: série é estacionária."
    elif adf_p >= 0.05 and kp_p < 0.05:
        combinado = "ADF e KPSS CONCORDAM: série NÃO é estacionária. Diferenciar."
    else:
        combinado = (
            "Testes DISCORDAM (zona cinza). Olhar mais lags ou tentar log-transform."
        )
    print(f"  >>> {combinado}")
    print()


def varredura_adf(
    serie: pd.Series, nome: str, lags_grid: tuple[int, ...] = (0, 4, 8, 12)
) -> None:
    """Roda o ADF várias vezes com maxlag fixo e exibe todos os resultados.

    Por que existe?
        O p-valor do ADF É SENSÍVEL ao número de lags incluídos na regressão
        auxiliar. Se com 4 lags ele rejeita H0 mas com 12 ele não rejeita,
        sua conclusão é frágil. Mostrar a varredura é mais honesto que
        confiar cegamente no autolag.

        Se a coluna p-valor ficar TODA abaixo de 0.05, a conclusão é
        ROBUSTA. Se variar muito entre os k, atenção.
    """
    print(f"[{nome}] varredura ADF (regression='c', autolag=None)")
    print(f"  {'maxlag':>7} | {'stat':>9} | {'p-valor':>8} | veredito (5%)")
    pvalores = []
    for k in lags_grid:
        # autolag=None + maxlag=k força exatamente k lags. Se passasse
        # autolag, ele ignoraria o maxlag.
        adf = adfuller(serie.dropna(), maxlag=k, autolag=None, regression="c")
        pvalores.append(adf[1])
        veredito = "estac." if adf[1] < 0.05 else "n.estac."
        print(
            f"  {k:>7d} | {adf[0]:>+9.3f} | {adf[1]:>8.4f} | {veredito}"
        )
    # Resumo: a conclusão muda dependendo do maxlag? Se sim, é frágil.
    todos_estac = all(p < 0.05 for p in pvalores)
    todos_nao = all(p >= 0.05 for p in pvalores)
    if todos_estac:
        print("  >>> Robusto: TODOS os maxlag rejeitam H0 -> estacionária.")
    elif todos_nao:
        print("  >>> Robusto: NENHUM maxlag rejeita H0 -> precisa diferenciar.")
    else:
        print("  >>> FRÁGIL: conclusão depende do maxlag. Tomar cuidado.")
    print()


def chart_residuos_no_tempo(resid: pd.Series, nome_arquivo: str) -> None:
    """Resíduos plotados em sequência temporal."""
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.plot(resid, color=INSPER_RED, lw=0.9)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title("Resíduos no tempo", fontweight="bold", pad=14)
    ax.set_xlabel("Tempo", labelpad=10)
    ax.set_ylabel("Resíduo", labelpad=10)
    fig.autofmt_xdate()
    salvar_figura(fig, nome_arquivo)


def chart_residuos_histograma(resid: pd.Series, nome_arquivo: str) -> None:
    """Histograma dos resíduos com a curva da normal teórica sobreposta."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # density=True normaliza o histograma pra área 1 -> dá pra comparar
    # diretamente com a densidade normal sobreposta.
    ax.hist(
        resid,
        bins=30,
        density=True,
        color=INSPER_TURQUESA,
        edgecolor=INSPER_GRAY,
        alpha=0.7,
        label="Resíduos",
    )
    # Constrói a curva normal usando média e desvio dos próprios resíduos.
    x = np.linspace(resid.min(), resid.max(), 200)
    ax.plot(
        x,
        norm.pdf(x, resid.mean(), resid.std()),
        color=INSPER_RED,
        lw=2,
        label="Normal teórica",
    )
    ax.set_title("Histograma dos resíduos vs. normal", fontweight="bold", pad=14)
    ax.set_xlabel("Valor do resíduo", labelpad=10)
    ax.set_ylabel("Densidade", labelpad=10)
    ax.legend(frameon=False)
    salvar_figura(fig, nome_arquivo)


def chart_residuos_qq(resid: pd.Series, nome_arquivo: str) -> None:
    """Q-Q plot: quantis empíricos vs. quantis teóricos da normal."""
    fig, ax = plt.subplots(figsize=(7, 7))
    # line='45' desenha a diagonal y=x; fit=True padroniza pelos quantis
    # da normal estimada com média e variância dos resíduos.
    qqplot(
        resid,
        line="45",
        fit=True,
        ax=ax,
        markerfacecolor=INSPER_RED,
        markeredgecolor=INSPER_RED,
        alpha=0.7,
    )
    ax.set_title("Q-Q plot dos resíduos vs. normal", fontweight="bold", pad=14)
    ax.set_xlabel("Quantis teóricos (normal padrão)", labelpad=10)
    ax.set_ylabel("Quantis empíricos (resíduos)", labelpad=10)
    salvar_figura(fig, nome_arquivo)


def diagnostico_residuos(modelo) -> None:
    """Avalia se os resíduos do modelo são "ruído branco" (bem-comportados).

    Intuição
    ---------
    Se um SARIMA capturou TODA a estrutura dinâmica dos dados, os resíduos
    e_t = Y_t - Y_t_chapeu devem ser:
        - sem autocorrelação (Ljung-Box não rejeita)
        - com variância constante (ARCH-LM não rejeita)
        - aproximadamente normais (Jarque-Bera não rejeita; em séries
          financeiras quase sempre rejeita por causa de outliers, ok)

    Se algum desses falhar, o modelo está MAL ESPECIFICADO em alguma
    dimensão e ainda tem padrão para ser explicado.

    Esta função gera 4 GRÁFICOS SEPARADOS de diagnóstico, cada um com sua
    explicação, e roda 3 testes formais.

    Documentação
    ------------
    qqplot         : https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html
    acorr_ljungbox : https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html
    het_arch       : https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html
    jarque_bera    : https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html
    teoria        : https://otexts.com/fpp3/diagnostics.html
    """
    # Pega os resíduos descartando os primeiros loglikelihood_burn valores.
    # Os primeiros resíduos do filtro de Kalman são instáveis (aquecimento).
    # O próprio modelo informa quantos descartar.
    resid = modelo.resid.iloc[modelo.loglikelihood_burn :]

    explicar_grafico(
        "Diagnóstico 1/4: Resíduos no tempo",
        "Cada ponto é o ERRO do modelo num período: e_t = Y_t - previsão.",
        "O que esperar de um bom modelo:",
        "  - oscilação aleatória em torno de zero, sem deriva",
        "  - amplitude (variância) constante ao longo do tempo",
        "  - sem 'clusters' visíveis de variância (períodos calmos seguidos",
        "    de períodos agitados sugerem heterocedasticidade -> GARCH)",
        "Outliers isolados são aceitáveis (covid, crises). Padrões NÃO.",
    )
    chart_residuos_no_tempo(resid, "07a_residuos_tempo.png")

    explicar_grafico(
        "Diagnóstico 2/4: Histograma vs. normal",
        "Compara a DISTRIBUIÇÃO empírica dos resíduos (barras) com a curva",
        "de uma normal teórica (linha vermelha) com mesma média e desvio.",
        "O que esperar:",
        "  - barras encostando na curva -> resíduos quase normais",
        "  - caudas mais 'gordas' (barras acima da curva nos extremos) ->",
        "    distribuição leptocúrtica, comum em séries financeiras",
        "  - assimetria visível (mais peso de um lado) -> modelo viesado",
    )
    chart_residuos_histograma(resid, "07b_residuos_histograma.png")

    explicar_grafico(
        "Diagnóstico 3/4: Q-Q plot",
        "Quantis empíricos dos resíduos (eixo Y) contra quantis teóricos",
        "de uma normal padronizada (eixo X). A linha diagonal vermelha é",
        "y=x: pontos sobre ela = perfeitamente normal.",
        "Como ler:",
        "  - pontos no MEIO encostam na linha -> centro da distribuição é OK",
        "  - pontos nos EXTREMOS afastando da linha (curva em S) -> caudas",
        "    pesadas; previsões com IC 95% serão otimistas demais",
        "  - desvio sistemático para cima ou para baixo -> assimetria",
    )
    chart_residuos_qq(resid, "07c_residuos_qq.png")

    explicar_grafico(
        "Diagnóstico 4/4: ACF dos resíduos",
        "Esse é o teste mais importante: se o SARIMA capturou tudo, os",
        "resíduos não devem ter autocorrelação NENHUMA - todas as barras",
        "devem ficar DENTRO da banda de confiança 95%.",
        "Se sobrar pico em algum lag (especialmente em 12, 24 sazonais),",
        "o modelo está mal especificado e sobrou estrutura para capturar.",
    )
    chart_acf(resid, "Resíduos do modelo", "07d_residuos_acf.png", lags=36)

    # Agora os testes FORMAIS (numéricos) que complementam os gráficos:
    print()
    print("=" * 72)
    print("TESTES FORMAIS DOS RESÍDUOS")
    print("=" * 72)

    # LJUNG-BOX em 3 horizontes (1, 2, 3 anos = lags 12, 24, 36).
    # Intuição: em vez de testar se a ACF do resíduo no lag k é zero
    # individualmente, testa CONJUNTAMENTE se as primeiras k autocorrelações
    # são TODAS zero. Mais poderoso que olhar lag a lag.
    # H0: resíduos = ruído branco. NÃO rejeitar (p > 0.05) é BOM.
    lb = acorr_ljungbox(resid, lags=[12, 24, 36], return_df=True)
    print("\nLjung-Box (H0: sem autocorrelação) -- rejeitar é RUIM")
    print(lb.round(4))
    # Veredito linha por linha (mais fácil de ler que olhar a tabela).
    for lag, p in zip(lb.index, lb["lb_pvalue"]):
        veredito = "OK (sem autocorrelação)" if p >= 0.05 else "RUIM (sobrou padrão)"
        print(f"   lag {int(lag):>2d}: p = {p:.4f}  ->  {veredito}")

    # ARCH-LM(12).
    # Intuição: regride e_t^2 contra e_{t-1}^2, ..., e_{t-12}^2. Se os
    # QUADRADOS dos resíduos são autocorrelacionados, há cluster de
    # volatilidade (períodos agitados seguem agitados). H0: sem ARCH.
    # Se rejeitar, o SARIMA tá OK na média, mas vc precisa de um GARCH na
    # variância pra previsão honesta.
    # A função devolve 4 valores; usamos só os 2 primeiros. "_" = "não me
    # importo".
    arch_stat, arch_p, _, _ = het_arch(resid, nlags=12)
    print(f"\nARCH-LM(12) : stat = {arch_stat:.3f} | p-valor = {arch_p:.4f}")
    if arch_p < 0.05:
        print(
            "          >>> heterocedasticidade detectada (clusters de volatilidade)."
        )
        print("              Ação: considerar modelar a variância com GARCH.")
    else:
        print("          >>> sem evidência de ARCH. Variância tá estável, OK.")

    # JARQUE-BERA: combina assimetria + curtose dos resíduos numa só
    # estatística (segue chi-quadrado_2 sob H0).
    # H0: resíduos são normais. Em séries macro brasileiras quase sempre é
    # rejeitado por outliers (covid, crises). Não invalida o modelo na
    # média, mas sugere que os IC da previsão são otimistas demais.
    jb_stat, jb_p = stats.jarque_bera(resid)
    skew = stats.skew(resid)        # assimetria
    kurt = stats.kurtosis(resid)    # curtose em excesso (normal = 0)
    print(f"\nJarque-Bera : stat = {jb_stat:.3f} | p-valor = {jb_p:.4f}")
    print(f"              assimetria = {skew:+.3f}  (normal: 0)")
    print(f"              curtose excesso = {kurt:+.3f}  (normal: 0; >0 = caudas pesadas)")
    if jb_p < 0.05:
        print("          >>> resíduos NÃO normais (cauda pesada ou assimetria).")
        print("              Implicação: os IC 95% das previsões são otimistas demais.")
    else:
        print("          >>> compatível com normalidade.")
    print()


def confirmar_com_autoarima(serie: pd.Series, nome: str = "y") -> None:
    """Roda AutoARIMA como validação cruzada da escolha manual.

    Ideia
    -----
    Você escolheu (p,d,q,P,D,Q) "à mão" lendo a ACF/PACF. Será que sua
    leitura está correta? Para checar, deixamos um algoritmo automático
    (AutoARIMA) buscar a melhor ordem por conta própria. Se ele cair na
    MESMA ordem que você, ótimo: identificação visual confirmada. Se cair
    em algo bem diferente, vale revisitar a leitura.

    Como o AutoARIMA funciona (algoritmo Hyndman-Khandakar):
        1. Decide d e D via testes de raiz unitária.
        2. Faz busca em grade restrita sobre p, q, P, Q.
        3. Escolhe a combinação com menor AICc (AIC corrigido para
           amostras pequenas).

    Documentação
    ------------
    AutoARIMA      : https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#autoarima
    StatsForecast  : https://nixtlaverse.nixtla.io/statsforecast/src/core/core.html
    Algoritmo HK   : https://www.jstatsoft.org/article/view/v027i03
    teoria         : https://otexts.com/fpp3/arima-r.html
    """
    if not HAS_STATSFORECAST:
        print("statsforecast não instalado - pulando confirmação automática.")
        print("Para habilitar: pip install statsforecast\n")
        return

    # statsforecast espera DataFrame com colunas EXATAS:
    #   unique_id (id da série), ds (date stamp), y (valor).
    # Aqui convertemos a Series para esse formato:
    df = (
        serie.rename(nome)
        .reset_index()
        .rename(columns={serie.index.name or "index": "ds"})
    )
    df["unique_id"] = "serie"
    df = df[["unique_id", "ds", nome]].rename(columns={nome: "y"})

    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS")
    sf.fit(df=df)

    # sf.fitted_ é array bidimensional (séries x modelos). Como temos
    # 1 série e 1 modelo, indexamos [0, 0]. .model_ é dict com resultados.
    modelo = sf.fitted_[0, 0].model_
    print("\nAutoARIMA (statsforecast) - ordem selecionada:")
    # A tupla 'arma' segue convenção herdada do R: (p, q, P, Q, s, d, D).
    print(f"  arma   = {modelo.get('arma')}   # (p, q, P, Q, s, d, D)")
    print(f"  AICc   = {modelo.get('aicc'):.2f}")
    print(f"  loglik = {modelo.get('loglik'):.2f}")


def chart_previsao(
    serie_observada: pd.Series,
    media: pd.Series,
    ic: pd.DataFrame,
    titulo: str,
    nome_arquivo: str,
    janela_observada: int = 60,
) -> None:
    """Plota observado recente + previsão pontual + faixa do IC 95%.

    Parâmetros
    ----------
    janela_observada : int
        Quantos meses de série observada mostrar à esquerda da previsão.
        Mostrar a série inteira (20+ anos) faria a previsão sumir.

    Documentação
    ------------
    get_forecast : https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast.html
    conf_int     : https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.PredictionResults.conf_int.html
    fill_between : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill_between.html
    """
    fig, ax = plt.subplots(figsize=(13, 5.5))
    serie_observada.iloc[-janela_observada:].plot(
        ax=ax, color=INSPER_GRAY, lw=1.2, label="Observado"
    )
    media.plot(ax=ax, color=INSPER_RED, lw=2.2, label="Previsão (média)")
    # fill_between pinta uma faixa entre dois limites; aqui, limite
    # inferior (coluna 0 do DF de IC) e superior (coluna 1).
    ax.fill_between(
        ic.index,
        ic.iloc[:, 0],
        ic.iloc[:, 1],
        color=INSPER_RED,
        alpha=0.15,
        label="IC 95%",
    )
    ax.set_title(titulo, fontweight="bold", pad=14)
    ax.set_xlabel("Tempo", labelpad=10)
    ax.set_ylabel("Valor", labelpad=10)
    ax.legend(frameon=False, loc="best")
    fig.autofmt_xdate()
    salvar_figura(fig, nome_arquivo)


def main() -> None:
    """Pipeline completo: download -> exploração -> identificação -> diagnóstico.

    Esse main() roda do começo ao fim sem travar. Todas as figuras vão pra
    pasta figs/ ao lado do script (já criada lá em cima). Quando terminar,
    abra a pasta no Finder e folheie os PNGs na ordem (01_, 02_, 03a_, ...)
    junto com o log do terminal.
    """

    # PASSO 1: download.
    # IBC-Br SEM ajuste sazonal (SGS 24363) é proxy mensal do PIB brasileiro,
    # com tendência crescente clara e sazonalidade anual forte (cenário ideal
    # pra um SARIMA didático).
    print("=" * 72)
    print("PASSO 1: download da série")
    print("=" * 72)
    ibc = bcb_serie(24363, "ibc_br_sa_nao", start="2003-01-01")
    print(f"Série   : IBC-Br sem ajuste sazonal (SGS 24363)")
    print(f"Período : {ibc.index.min().date()} -> {ibc.index.max().date()}")
    print(f"N obs   : {len(ibc)}")
    print(f"Figuras : todas vão pra {FIGS_DIR}")

    # PASSO 2: Visualização do nível.
    print()
    print("=" * 72)
    print("PASSO 2: olhar a série e sua estrutura de autocorrelação")
    print("=" * 72)

    explicar_grafico(
        "Série bruta (nível)",
        "É o IBC-Br original. O que olhar:",
        "  - há TENDÊNCIA? (linha sobe ou desce no longo prazo)",
        "  - há SAZONALIDADE? (padrão repetindo a cada 12 meses)",
        "  - há QUEBRAS estruturais? (mudanças bruscas em 2008, 2015, 2020)",
        "  - a VARIÂNCIA parece constante? (oscilações de tamanho similar)",
        "Esperamos ver tendência crescente clara, com vales nas crises.",
    )
    chart_serie(ibc, "IBC-Br - nível", "02a_serie_nivel.png", ylabel="Índice")

    explicar_grafico(
        "ACF do nível",
        "Mostra quanto a série de hoje se correlaciona com seus passados.",
        "Em série com tendência (não-estacionária), as barras decaem MUITO",
        "LENTAMENTE e ficam fora da banda por dezenas de lags. Essa é a",
        "'assinatura' clássica de RAIZ UNITÁRIA -> precisa diferenciar.",
    )
    chart_acf(ibc, "IBC-Br nível", "02b_acf_nivel.png", lags=48)

    explicar_grafico(
        "PACF do nível",
        "Mede correlação direta com cada lag, controlando os intermediários.",
        "Em série com raiz unitária: spike GIGANTE no lag 1 (próximo de 1)",
        "e quase nada nos demais. Reforça o diagnóstico de não-estacionária.",
    )
    chart_pacf(ibc, "IBC-Br nível", "02c_pacf_nivel.png", lags=48)

    # PASSO 3: Decomposição STL (4 gráficos separados, cada um explicado).
    print()
    print("=" * 72)
    print("PASSO 3: decomposição STL em tendência + sazonalidade + resíduo")
    print("=" * 72)
    decompor_stl(ibc, period=12, titulo="IBC-Br")

    # PASSO 4: Diferenciação.
    print()
    print("=" * 72)
    print("PASSO 4: diferenciação para remover tendência e sazonalidade")
    print("=" * 72)

    # Construímos 3 versões transformadas para comparar:
    #   ibc.diff()           = Y_t - Y_{t-1}    (remove tendência linear)
    #   ibc.diff(12)         = Y_t - Y_{t-12}   (remove sazonalidade estável)
    #   ibc.diff().diff(12)  = aplica AS DUAS   (remove tendência + sazon.)
    # .dropna() porque a primeira (ou primeiras 12) observações viram NaN.
    ibc_d1 = ibc.diff().dropna()
    ibc_d12 = ibc.diff(12).dropna()
    ibc_d1_12 = ibc.diff().diff(12).dropna()

    explicar_grafico(
        "Diferença regular:  ΔYₜ = Yₜ - Yₜ₋₁",
        "Remove TENDÊNCIA linear. Cada ponto é a variação de um mês pro",
        "seguinte. Se a tendência fosse perfeitamente linear, esse gráfico",
        "já oscilaria estável em torno de uma constante.",
        "Aqui ainda deve sobrar sazonalidade visível (ondas de 12 meses).",
    )
    chart_serie_diferenciada(
        ibc_d1,
        r"$\Delta Y_t$",
        "1ª diferença regular",
        "04a_diff_regular.png",
    )

    explicar_grafico(
        "Diferença sazonal:  Δ₁₂Yₜ = Yₜ - Yₜ₋₁₂",
        "Remove SAZONALIDADE estável. Cada ponto é a variação em relação",
        "ao MESMO mês do ano anterior. Aqui ainda deve sobrar a tendência",
        "de longo prazo (a série deve ter média != 0 ou deriva visível).",
    )
    chart_serie_diferenciada(
        ibc_d12,
        r"$\Delta_{12} Y_t$",
        "1ª diferença sazonal",
        "04b_diff_sazonal.png",
    )

    explicar_grafico(
        "Δ Δ₁₂ Yₜ = (Yₜ - Yₜ₋₁) - (Yₜ₋₁₂ - Yₜ₋₁₃)",
        "Aplica AS DUAS diferenças. Remove tendência E sazonalidade.",
        "Essa é a versão que esperamos ser ESTACIONÁRIA: oscila em torno",
        "de zero, sem deriva, com amplitude relativamente constante.",
        "Outliers nas crises (2009, 2020) são esperados; o resto deve",
        "parecer ruído.",
    )
    chart_serie_diferenciada(
        ibc_d1_12,
        r"$\Delta\,\Delta_{12} Y_t$",
        "dif. regular + sazonal (estacionária?)",
        "04c_diff_regular_sazonal.png",
    )

    # PASSO 5: Testes formais de estacionariedade.
    print()
    print("=" * 72)
    print("PASSO 5: testes formais (ADF + KPSS)")
    print("=" * 72)
    print()
    print("Esperado:")
    print("  - Nível            : ADF NÃO rejeita | KPSS rejeita -> não-estac.")
    print("  - Δ Δ_12 Y_t       : ADF rejeita     | KPSS NÃO rej. -> ESTACIONÁRIA")
    print()
    teste_estac(ibc, "Y_t (nível)")
    teste_estac(ibc_d1, "Delta Y_t (1a diferença)")
    teste_estac(ibc_d12, "Delta12 Y_t (diferença sazonal)")
    teste_estac(ibc_d1_12, "Delta Delta12 Y_t (regular + sazonal)")

    print("Robustez do ADF: o p-valor depende do número de lags na regressão")
    print("auxiliar. Se ele NÃO MUDA quando variamos 'maxlag', a conclusão é")
    print("sólida. Vamos varrer maxlag em (0, 4, 8, 12):\n")
    varredura_adf(ibc, "Y_t (nível)")
    varredura_adf(ibc_d1_12, "Delta Delta12 Y_t")

    # Conclusão dos passos 3-5:
    # O nível tem raiz unitária; após Delta Delta12 a série é estacionária.
    # FIXAMOS d=1, D=1, s=12 para o SARIMA.

    # PASSO 6: Identificação manual via ACF/PACF da série já estacionária.
    print()
    print("=" * 72)
    print("PASSO 6: identificação Box-Jenkins (ACF/PACF da série diferenciada)")
    print("=" * 72)

    explicar_grafico(
        "ACF da série diferenciada",
        "Agora que a série é estacionária, a ACF/PACF tem significado",
        "diferente: serve para escolher q (parte MA não-sazonal) e Q",
        "(parte MA sazonal).",
        "REGRA PRÁTICA:",
        "  - pico isolado no lag 1 e nada depois -> q = 1",
        "  - pico isolado no lag 12 e nada depois -> Q = 1 (sazonal)",
        "  - pico em 1 e 12 simultâneos -> SARIMA(0,1,1)(0,1,1)_12",
        "    (o famoso 'airline model' de Box & Jenkins)",
    )
    chart_acf(
        ibc_d1_12,
        r"$\Delta\,\Delta_{12}$ IBC-Br",
        "06a_acf_diff.png",
        lags=48,
    )

    explicar_grafico(
        "PACF da série diferenciada",
        "Serve para escolher p (AR não-sazonal) e P (AR sazonal).",
        "REGRA PRÁTICA:",
        "  - pico isolado no lag 1 e nada depois -> p = 1",
        "  - pico isolado no lag 12 -> P = 1",
        "  - decaimento gradual (sem spikes isolados) -> não precisa de AR",
        "    (o modelo é só MA, regular e sazonal)",
    )
    chart_pacf(
        ibc_d1_12,
        r"$\Delta\,\Delta_{12}$ IBC-Br",
        "06b_pacf_diff.png",
        lags=48,
    )

    # Lista de candidatos (p, d, q, P, D, Q).
    # Mantemos d=1, D=1 fixos (já decididos pelos testes).
    # Variamos p, q, P, Q para ver se vale a pena adicionar termos.
    candidatos = [
        (0, 1, 1, 0, 1, 1),  # airline model puro: nossa hipótese
        (1, 1, 1, 0, 1, 1),  # adiciona um AR(1) regular
        (0, 1, 1, 1, 1, 1),  # adiciona um SAR(1) sazonal
        (1, 1, 0, 0, 1, 1),  # troca MA por AR na parte regular
    ]

    # Loop: estima cada candidato, coleta AIC e BIC.
    # AIC e BIC medem "ajuste penalizado por complexidade":
    #   AIC = -2 log L + 2 k
    #   BIC = -2 log L + k log n
    # MENOR é melhor nos dois. AIC é mais permissivo, BIC mais conservador.
    # docs SARIMAX: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    # docs .fit():  https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.fit.html
    # AIC/BIC docs: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html
    # teoria:       https://otexts.com/fpp3/seasonal-arima.html
    resultados = []
    for p, d, q, P, D, Q in candidatos:
        fit = SARIMAX(
            ibc,
            order=(p, d, q),
            seasonal_order=(P, D, Q, 12),
            # As duas opções abaixo afrouxam restrições do otimizador. Em
            # modelos parcimoniosos (como o airline) ajuda a convergência.
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)  # disp=False esconde output do otimizador
        resultados.append(
            {
                "order": f"({p},{d},{q})({P},{D},{Q})_12",
                "AIC": fit.aic,
                "BIC": fit.bic,
            }
        )

    tabela = pd.DataFrame(resultados).sort_values("AIC").reset_index(drop=True)
    print("\nComparação de candidatos (ordenado por AIC, menor = melhor):")
    print(tabela.to_string(index=False), "\n")

    # Interpretação dos números: quem ganhou e por quanto?
    melhor_aic = tabela.iloc[0]
    pior_aic = tabela.iloc[-1]
    delta_aic = pior_aic["AIC"] - melhor_aic["AIC"]
    print("Interpretação:")
    print(f"  - melhor por AIC : {melhor_aic['order']}  (AIC = {melhor_aic['AIC']:.2f})")
    print(f"  - pior   por AIC : {pior_aic['order']}  (AIC = {pior_aic['AIC']:.2f})")
    print(f"  - gap (pior-melhor) = {delta_aic:.2f}")
    # Regra de bolso (Burnham & Anderson): ΔAIC<2 = equivalentes; 4-7 =
    # diferença "considerável"; >10 = um modelo é claramente superior.
    if delta_aic < 2:
        leitura = "todos os candidatos são praticamente equivalentes (ΔAIC<2)."
    elif delta_aic < 7:
        leitura = "diferença considerável: vale ficar com o melhor."
    else:
        leitura = "diferença GRANDE: o melhor modelo é claramente superior."
    print(f"  >>> {leitura}")
    print("  (Regra de Burnham & Anderson: ΔAIC<2 ~ equivalentes; >10 = decisivo)\n")

    # Confirmação cruzada: o que o AutoARIMA escolhe SOZINHO?
    confirmar_com_autoarima(ibc, nome="ibc")

    # Estima o modelo final escolhido (airline) e imprime sumário completo.
    print()
    print("=" * 72)
    print("MODELO FINAL: SARIMA(0,1,1)(0,1,1)_12 - airline model")
    print("=" * 72)
    modelo = SARIMAX(
        ibc,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    print(modelo.summary())

    # Interpretação direta dos parâmetros estimados (em vez de só "olha aí").
    # Para o airline model: ma.L1 (theta_1) e ma.S.L12 (Theta_12).
    # Ambos devem ser NEGATIVOS e SIGNIFICATIVOS (|t| > ~2). Quanto mais
    # próximos de -1, mais "memória" o modelo extrai do erro passado.
    print()
    print("Interpretação dos coeficientes:")
    params = modelo.params
    pvalues = modelo.pvalues
    for nome_param in ["ma.L1", "ma.S.L12"]:
        if nome_param in params.index:
            coef = params[nome_param]
            pval = pvalues[nome_param]
            sig = "SIGNIFICATIVO" if pval < 0.05 else "NÃO significativo"
            tipo = "regular (curto prazo)" if "S" not in nome_param else "sazonal (12 meses)"
            print(
                f"  {nome_param:>10s} = {coef:+.4f}  (p={pval:.4f})  -> {sig}, MA {tipo}"
            )
    print(f"  log-likelihood = {modelo.llf:.2f}   AIC = {modelo.aic:.2f}   BIC = {modelo.bic:.2f}")
    print("  Sigma^2 (variância do erro) =", f"{modelo.params.get('sigma2', float('nan')):.3f}")
    print("  >>> coefs MA NEGATIVOS = padrão saudável do airline. Se algum")
    print("      vier com p>0.05, o termo é dispensável (modelo simplifica).")

    # PASSO 7: Diagnóstico dos resíduos (4 gráficos + 3 testes formais).
    print()
    print("=" * 72)
    print("PASSO 7: diagnóstico dos resíduos do modelo")
    print("=" * 72)
    diagnostico_residuos(modelo)

    # BÔNUS: Previsão 12 meses à frente com IC 95%.
    print()
    print("=" * 72)
    print("BÔNUS: previsão 12 meses à frente")
    print("=" * 72)
    fcst = modelo.get_forecast(steps=12)
    media = fcst.predicted_mean       # ponto médio (a "previsão" propriamente)
    ic = fcst.conf_int(alpha=0.05)    # IC 95% (deixa 2.5% em cada cauda)

    # Tabela legível com previsão + IC mês a mês (números, não só gráfico).
    tabela_fcst = pd.DataFrame(
        {
            "previsão": media.round(2),
            "IC inf 95%": ic.iloc[:, 0].round(2),
            "IC sup 95%": ic.iloc[:, 1].round(2),
            "amplitude IC": (ic.iloc[:, 1] - ic.iloc[:, 0]).round(2),
        }
    )
    tabela_fcst.index = tabela_fcst.index.strftime("%Y-%m")
    print("\nPrevisão 12 meses à frente:")
    print(tabela_fcst.to_string())

    # Interpretação numérica da previsão e da incerteza.
    ultimo_obs = ibc.iloc[-1]
    primeira_prev = media.iloc[0]
    ultima_prev = media.iloc[-1]
    ic_h1 = ic.iloc[0, 1] - ic.iloc[0, 0]
    ic_h12 = ic.iloc[-1, 1] - ic.iloc[-1, 0]
    print()
    print("Interpretação:")
    print(f"  - último observado ({ibc.index[-1].strftime('%Y-%m')}) : {ultimo_obs:.2f}")
    print(f"  - previsão h+1                         : {primeira_prev:.2f}")
    print(f"  - previsão h+12                        : {ultima_prev:.2f}")
    var_pct = (ultima_prev / ultimo_obs - 1) * 100
    print(f"  - variação prevista em 12 meses        : {var_pct:+.2f}%")
    print(f"  - amplitude do IC95% em h+1            : {ic_h1:.2f}")
    print(f"  - amplitude do IC95% em h+12           : {ic_h12:.2f}")
    razao = ic_h12 / ic_h1 if ic_h1 else float("nan")
    print(f"  - razão IC(h+12)/IC(h+1)               : {razao:.2f}x")
    print(f"  >>> a incerteza cresce ~{razao:.1f}x do mês 1 ao mês 12.")
    print("      Isso é esperado: erros de previsão acumulam no horizonte.\n")

    explicar_grafico(
        "Previsão 12 meses à frente",
        "Linha cinza: últimos 60 meses observados (pra dar contexto).",
        "Linha vermelha: previsão pontual (média condicional do modelo).",
        "Faixa rosa: IC 95% - 95% de prob. do valor real cair na faixa,",
        "ASSUMINDO que o modelo tá certo e que os resíduos são normais.",
        "A faixa ALARGA com o horizonte: incerteza acumula no tempo.",
    )
    chart_previsao(
        ibc,
        media,
        ic,
        "IBC-Br - Previsão SARIMA(0,1,1)(0,1,1)_12",
        "08_previsao_12m.png",
        janela_observada=60,
    )

    # Mensagem final = "tudo terminou, vai pra pasta de figuras".
    print()
    print("=" * 72)
    print("PIPELINE COMPLETO. Abra a pasta abaixo pra ver todas as figuras:")
    print(f"  {FIGS_DIR}")
    print("=" * 72)


# Convenção Python: o bloco abaixo só roda quando o arquivo é EXECUTADO
# diretamente (uv run python lab1_timeseries_stuff.py). Se alguém fizer
# `import lab1_timeseries_stuff` para reusar as funções, main() não é
# chamada automaticamente. É boa prática separar definição de execução.
if __name__ == "__main__":
    main()
