"""
Checkpoint
Autor: André Ichiro Katsurada
Data: 29/08/2025
Curso: Programa Avançado em Data Science e Decisão, Computação para a Ciência de Dados, INSPER
"""

from pathlib import Path
from bisect import bisect_right

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# Parâmetros
UF = "RJ"                        
DATE_TOP10 = "2021-02-25"        # data de ref
POP_MIN = 10_000                 # Q4: filtro 
FIG_DIR = Path("figures")        # path p/ figuras
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://raw.githubusercontent.com/wcota/covid19br/master/"
URL_STATES = BASE + "cases-brazil-states.csv"
URL_CITIES = BASE + "cases-brazil-cities.csv"
URL_CITIES_INFO = BASE + "cities_info.csv"
URL_GPS = BASE + "gps_cities.csv"

# Tema padrão dos gráficos 
sns.set_theme(style="whitegrid", context="talk")

# Utils
def salvar_fig(nome: str) -> None:
    """Salva a figura atual em FIG_DIR/<nome>.png """
    plt.tight_layout()
    out = FIG_DIR / f"{nome}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[Figura salva] {out}")

def formata_num_pt(valor: float, _pos=None) -> str:
    """
    Formata números em PT com sufixos 'mil', 'mi', 'bi'
    Ex.: 1_200 -> '1,2 mil', 5_000_000 -> '5,0 mi'
    """
    # ordem de grandeza e sufixos
    limites = [1.0, 1e3, 1e6, 1e9]     # base, mil, milhão, bilhão
    sufixos = ["", " mil", " mi", " bi"]

    negativo = valor < 0
    x = abs(float(valor))

    # N. pequenos, exibir com até 2 casas
    if x < 1:
        s = f"{x:.2f}"
    else:
        i = bisect_right(limites, x) - 1 
        escala = limites[i]
        if i == 0:
            s = f"{int(round(x))}"
        else:
            s = f"{x/escala:.1f}{sufixos[i]}"
    return f"-{s}" if negativo else s

PT_FMT = FuncFormatter(formata_num_pt)

def anotar_barras_h(ax, formatador=lambda v: f"{v:.1f}") -> None:
    """Escreve o valor ao final de cada barra horizontal."""
    for patch in ax.patches:
        v = patch.get_width()
        ax.text(v, patch.get_y() + patch.get_height()/2,
                " " + formatador(v), va="center", ha="left", fontsize=9)

def eixo_datas_conciso(ax) -> None:
    """Aplica formatação de datas concisa no eixo X."""
    loc = AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))

def exige_colunas(df: pd.DataFrame, cols: list[str], contexto: str = "") -> None:
    """Falha explicitamente se colunas obrigatórias não existirem."""
    faltantes = set(cols) - set(df.columns)
    assert not faltantes, f"[{contexto}] colunas ausentes: {sorted(faltantes)}"

def novos_casos_robustos(df: pd.DataFrame, total_col: str = "totalCases",
                         flow_col: str = "newCases") -> pd.Series:
    """
    Constrói série de 'novos casos' por estado:
    - Prioriza coluna de newCases quando positiva.
    - Na ausência/zero, usa diff(totalCases), tornando negativos == 0
    Retorna uma Series 'newCases_final'
    """
    s = df[flow_col].copy()
    if total_col in df.columns:
        diff = df.groupby("state")[total_col].diff()
        s = s.where(s.notna() & (s > 0), diff)
    return s.clip(lower=0)

# Tratamentos simples
states = pd.read_csv(URL_STATES)
cities = pd.read_csv(URL_CITIES)
cities_info = pd.read_csv(URL_CITIES_INFO)
gps = pd.read_csv(URL_GPS)

# Cast no tipo p/ datas
states["date"] = pd.to_datetime(states["date"], errors="coerce")
for c in ("date", "last_info_date"):
    if c in cities.columns:
        cities[c] = pd.to_datetime(cities[c], errors="coerce")

# Q1, cases-brazil-states.csv

# 1a) Linhas e colunas
q1_rows, q1_cols = states.shape
print("1a) Linhas e colunas:", q1_rows, q1_cols)

# Gráfico 1a — barras com escala log
plt.figure(figsize=(8, 5))
tmp = pd.DataFrame({"dimensao": ["linhas", "colunas"], "valor": [q1_rows, q1_cols]})
ax = sns.barplot(data=tmp, x="dimensao", y="valor")
ax.set_yscale("log")
ax.set_title("Q1a — Dimensões do arquivo (linhas vs colunas)")
ax.set_ylabel("Contagem (escala log)")
ax.grid(True, axis="y", alpha=0.3)
for container in ax.containers:
    ax.bar_label(container, fmt=lambda v: f"{int(v):,}".replace(",", "."), padding=2, fontsize=9)
salvar_fig("q1a_shape")

# 1b) dtypes
q1_dtypes = states.dtypes
print("\n1b) dtypes:")
print(q1_dtypes)

# Gráfico 1b, contagem por dtype
plt.figure(figsize=(8, 5))
dtype_counts = q1_dtypes.astype(str).value_counts().sort_values()
ax = sns.barplot(x=dtype_counts.values, y=dtype_counts.index, orient="h")
ax.set_title("Q1b — Quantidade de colunas por dtype")
ax.set_xlabel("Número de colunas")
ax.set_ylabel("dtype")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, axis="x", alpha=0.3)
salvar_fig("q1b_dtypes")

# 1c) Estatísticas básicas 
q1_stats = states.select_dtypes("number").describe().T
print("\n1c) Estatísticas básicas (numéricas):")
print(q1_stats)

# Gráfico 1c — boxplot de fluxos em escala symlog
cols_fluxos = [c for c in ("newCases", "newDeaths") if c in states.columns]
if cols_fluxos:
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=states[cols_fluxos])
    ax.set_yscale("symlog")
    ax.set_title("Q1c — Distribuição (newCases e newDeaths)")
    ax.set_ylabel("Valor (escala symlog)")
    ax.grid(True, axis="y", alpha=0.3)
    salvar_fig("q1c_flows_boxplot")

# 1d) Per day / linhas por dia
per_day = (states.dropna(subset=["date"])
                 .groupby("date", as_index=True)
                 .size()
                 .rename("linhas_no_dia"))
print("\n1d) Amostra de cobertura temporal (5 primeiras linhas):")
print(per_day.head())

plt.figure(figsize=(9, 5))
ax = sns.lineplot(x=per_day.index, y=per_day.values)
ax.set_title("Q1d — Cobertura temporal (registros por data)")
ax.set_xlabel("Data")
ax.set_ylabel("Nº de registros")
eixo_datas_conciso(ax)
ax.grid(True, axis="y", alpha=0.3)
salvar_fig("q1d_records_per_day")

# Sinais de revisão usando newCases e newDeaths
if {"newCases", "newDeaths"}.issubset(states.columns):
    neg_cases = int((states["newCases"] < 0).sum())
    neg_deaths = int((states["newDeaths"] < 0).sum())
    print(f"NewCases negativos={neg_cases} e newDeaths negativos={neg_deaths}")
    print("Valores negativos decorrem de revisões administrativas nas séries históricas.")

# Q2, cases-brazil-cities.csv
# Limpar linhas não municipais (total e importados/indefinidos)
city_series = cities["city"].fillna("")
mask_valid_city = (~city_series.eq("TOTAL")
                   & ~city_series.str.startswith("Importados/Indefinidos"))
cities_valid = cities.loc[mask_valid_city].copy()

# 2a) Distinct p/ cidades (usa ibgeID)
if "ibgeID" in cities_valid.columns:
    q2a_n_cidades = int(cities_valid["ibgeID"].nunique())
else:
    q2a_n_cidades = int(cities_valid[["state", "city"]].drop_duplicates().shape[0])

print(f"\n2a) Número de cidades distintas: {q2a_n_cidades}")

# Gráfico 2a, n. de cidades por UF
plt.figure(figsize=(8, 10))
if "ibgeID" in cities_valid.columns:
    cidades_por_uf = cities_valid.groupby("state")["ibgeID"].nunique()
else:
    cidades_por_uf = cities_valid.groupby("state")["city"].nunique()

cidades_por_uf = cidades_por_uf.sort_values(ascending=False)
ax = sns.barplot(x=cidades_por_uf.values, y=cidades_por_uf.index, orient="h")
ax.set_title("Q2a — Nº de cidades distintas por UF")
ax.set_xlabel("Nº de cidades")
ax.set_ylabel("UF")
ax.grid(True, axis="x", alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
salvar_fig("q2a_distinct_cities_by_state")

# 2b) Casos e mortes por estado 
exige_colunas(cities, ["state", "city", "totalCases", "deaths"], "Q2b")
mask_agg = ~cities["city"].fillna("").eq("TOTAL")
q2b_state_agg = (cities.loc[mask_agg]
                 .groupby("state", as_index=False)[["totalCases", "deaths"]]
                 .sum()
                 .sort_values(["totalCases", "deaths"], ascending=[False, False]))

print("\n2b) Casos e mortes por estado (top 10 linhas):")
print(q2b_state_agg.head(10))

# Gráfico 2b, top 10 UFs por casos (óbitos)
plt.figure(figsize=(9, 6))
top10 = q2b_state_agg.nlargest(10, "totalCases").set_index("state").sort_values("totalCases")
ax = sns.barplot(x=top10["totalCases"].values, y=top10.index, orient="h")
ax.set_title("Q2b — Top 10 UFs por casos (anotado: mortes)")
ax.set_xlabel("Total de casos")
ax.set_ylabel("UF")
ax.xaxis.set_major_formatter(PT_FMT)
ax.grid(True, axis="x", alpha=0.3)
for i, (uf, row) in enumerate(top10.iterrows()):
    ax.text(row["totalCases"], i, f"  óbitos={int(row['deaths']):,}".replace(",", "."),
            va="center", fontsize=9)
salvar_fig("q2b_cases_by_state_top10")

# Q3 — Top 10 UFs por 2a dose em 25/02/2021
exige_colunas(states, ["state", "city", "date", "vaccinated_second"], "Q3")
mask_states = states["city"].eq("TOTAL") & states["state"].ne("TOTAL")
mask_date = states["date"].eq(pd.to_datetime(DATE_TOP10))

q3_base = states.loc[mask_states & mask_date, ["state", "vaccinated_second"]].copy()
q3_base["vaccinated_second"] = q3_base["vaccinated_second"].fillna(0)
q3_top10 = (q3_base.sort_values("vaccinated_second", ascending=False)
            .head(10)
            .reset_index(drop=True))

print(f"\n3) TOP10 estados com mais imunizados (2ª dose) em {DATE_TOP10}:")
print(q3_top10)

plt.figure(figsize=(9, 6))
if not q3_top10.empty:
    ax = sns.barplot(data=q3_top10.sort_values("vaccinated_second"),
                     x="vaccinated_second", y="state", orient="h")
    ax.set_title(f"Q3 — Top 10 UFs por 2ª dose em {DATE_TOP10}")
    ax.set_xlabel("Pessoas com 2ª dose")
    ax.set_ylabel("UF")
    ax.xaxis.set_major_formatter(PT_FMT)
    ax.grid(True, axis="x", alpha=0.3)
    salvar_fig("q3_top10_second_dose")
else:
    plt.close()
    print("Aviso Q3: sem dados de 2ª dose na data informada.")

# Q4 — TOP5 cidades do UF com menor mortalidade (óbitos/100k)
# Usar filtro na população 2020
if {"ibge", "pop2020"}.issubset(cities_info.columns):
    cities_info["ibge"] = pd.to_numeric(cities_info["ibge"], errors="coerce")
    pop_por_ibge = cities_info.set_index("ibge")["pop2020"]
    cities_valid["pop2020"] = cities_valid["ibgeID"].map(pop_por_ibge)
else:
    cities_valid["pop2020"] = pd.NA

cols_q4 = ["state", "city", "ibgeID", "totalCases", "deaths_per_100k_inhabitants"]
exige_colunas(cities_valid, cols_q4 + ["deaths", "pop2020"], "Q4")
q4_base = (cities_valid.loc[cities_valid["state"].eq(UF), cols_q4 + ["deaths", "pop2020"]]
           .dropna(subset=["deaths_per_100k_inhabitants", "pop2020"])
           .query("totalCases > 0 and pop2020 >= @POP_MIN"))

q4_top5 = (q4_base.sort_values("deaths_per_100k_inhabitants", ascending=True)
           .head(5)
           .reset_index(drop=True))

print(f"\n4) TOP5 cidades de {UF} com menor mortalidade (óbitos por 100k) "
      f"(população ≥ {POP_MIN} e totalCases>0):")
print(q4_top5[["state", "city", "ibgeID", "totalCases", "deaths", "deaths_per_100k_inhabitants"]])

# Gráfico 4, menores no topo 
plt.figure(figsize=(9, 6))
ordem = q4_top5.sort_values("deaths_per_100k_inhabitants")["city"]
ax = sns.barplot(data=q4_top5, x="deaths_per_100k_inhabitants", y="city",
                 order=ordem, orient="h")
ax.set_title(f"Q4 — Menor mortalidade por município ({UF})")
ax.set_xlabel("Óbitos por 100 mil habitantes")
ax.set_ylabel("Município")
ax.grid(True, axis="x", alpha=0.3)
anotar_barras_h(ax, formatador=lambda v: f"{v:.1f}")
ax.invert_yaxis()  # menor no topo visualmente
salvar_fig(f"q4_top5_lowest_mortality_{UF}")

# Q5, enriquecer TOP5 com IBGE, nome completo, lat/lon, pop2020 e indicador metropolitano
exige_colunas(cities_info, ["ibge"], "Q5")
ci_slim = cities_info.copy()
ci_slim["ibge"] = pd.to_numeric(ci_slim["ibge"], errors="coerce")

# Detectar coluna p/ "metropolitana"
flag = None
for cand in ["isMetropolitan", "metropolitan", "inMetropolitanRegion", "isCountryside"]:
    if cand in ci_slim.columns:
        flag = cand
        break

# Construir 'is_metropolitan' com base no melhor flag disponível
if flag == "isCountryside":
    # 0 = não interior (~metropolitana/capital), 1 = interior
    ci_slim["is_metropolitan"] = pd.to_numeric(ci_slim["isCountryside"], errors="coerce").map({0.0: True, 1.0: False})
elif flag is not None:
    # Ao invés de flags 1/0 casting p/ boolean
    ci_slim["is_metropolitan"] = pd.to_numeric(ci_slim[flag], errors="coerce").round().map({1.0: True, 0.0: False})
else:
    # Sem info, então deixar como NA
    ci_slim["is_metropolitan"] = pd.NA

gps_slim = gps[["ibgeID", "lat", "lon", "longName"]].copy()

q5_enriched = (q4_top5.merge(ci_slim[["ibge", "is_metropolitan"]], left_on="ibgeID", right_on="ibge", how="left")
                        .merge(gps_slim, on="ibgeID", how="left"))


q5_out = (q5_enriched.rename(columns={
                "ibgeID": "codigo_ibge",
                "city": "nome_municipio",
                "state": "uf",
                "longName": "nome_completo",
                "lat": "latitude",
                "lon": "longitude",
                "pop2020": "populacao_2020"})
          [["codigo_ibge", "nome_municipio", "uf", "nome_completo",
            "latitude", "longitude", "populacao_2020", "is_metropolitan"]])

exige_colunas(q5_out, ["codigo_ibge", "nome_municipio", "uf", "nome_completo",
                       "latitude", "longitude", "populacao_2020", "is_metropolitan"], "Q5_out")

print(f"\n5) Informações integradas (TOP5 de {UF}):")
print(q5_out)

# Gráfico 5, dispersão c/ rótulos
plt.figure(figsize=(9, 6))
ax = sns.scatterplot(data=q5_out, x="longitude", y="latitude")
for _, r in q5_out.iterrows():
    ax.annotate(r["nome_municipio"], (r["longitude"], r["latitude"]),
                xytext=(3, 3), textcoords="offset points", fontsize=8)
ax.set_title(f"Q5 — Localização dos TOP5 ({UF})")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, alpha=0.3)
salvar_fig(f"q5_locations_{UF}")

# Q6 — Novos casos (últimos 7 dias) — Brasil total vs UF total 
exige_colunas(states, ["state", "city", "date", "newCases", "totalCases"], "Q6")
sel = (states.loc[(states["city"].eq("TOTAL")) & (states["state"].isin(["TOTAL", UF]))]
             .sort_values(["state", "date"])
             .copy())

sel["newCases_final"] = novos_casos_robustos(sel, total_col="totalCases", flow_col="newCases")
pivot = (sel.pivot(index="date", columns="state", values="newCases_final")
           .dropna(how="any"))

# definir janela de 7 dias; se não houver atividade, warning
aviso_q6 = False
if len(pivot) >= 7:
    roll = pivot.rolling(7, min_periods=7).sum()
    validas = roll.notna().all(axis=1) & (roll.sum(axis=1) > 0)
    last_date = validas[validas].index.max() if validas.any() else pivot.index.max()
    aviso_q6 = not validas.any()
else:
    last_date = pivot.index.max()
    aviso_q6 = True

window = last_date - pd.Timedelta(days=6)
plot_df = (pivot.loc[window:last_date]
                .rename(columns={"TOTAL": "Brasil (TOTAL)", UF: UF})[["Brasil (TOTAL)", UF]]
                .reset_index()
                .melt(id_vars="date", var_name="Série", value_name="Novos casos"))

plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=plot_df, x="date", y="Novos casos", hue="Série", marker="o")
ax.set_title(f"Q6 — Novos casos (últimos 7 dias) — Brasil vs {UF}")
ax.set_xlabel("Data")
ax.set_ylabel("Novos casos")
eixo_datas_conciso(ax)
ax.yaxis.set_major_formatter(PT_FMT)
ax.grid(True, axis="y", alpha=0.3)
salvar_fig(f"q6_7day_new_cases_brazil_vs_{UF}")

if aviso_q6:
    print("Aviso Q6: período final tem zeros/ausências; 'newCases' foi complementado via diff(totalCases).")