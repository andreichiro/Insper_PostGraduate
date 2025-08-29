# =============================================
# Script: Análise Exploratória de Dados (EDA)
# Objetivo: Ler uma planilha Excel de mortalidade, responder às
#           perguntas principais da EDA e salvar um relatório com
#           Perguntas e Respostas (Q&A) e poucos gráficos úteis.
# Público: Explicações didáticas em português, voltadas a usuários não técnicos.
# =============================================

import os
import io
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Bibliotecas numéricas e de análise de dados
import numpy as np
import pandas as pd

# Bibliotecas de visualização de gráficos
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Usa backend "sem tela" para salvar gráficos em arquivos
import matplotlib.pyplot as plt

# Biblioteca para visualizar faltantes (se disponível)
try:
    import missingno as msno
    HAS_MISSINGNO = True
except Exception:
    HAS_MISSINGNO = False

# Métodos estatísticos adicionais (z-score)
try:
    from scipy.stats import zscore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# Algoritmos de modelagem para valores mais prováveis
try:
    # IterativeImputer (BayesianRidge) e RandomForest
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer
    from sklearn.linear_model import BayesianRidge
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
    HAS_SKLEARN = True
except Exception:
    try:
        from sklearn.ensemble import IsolationForest  # pode existir mesmo sem os demais
        HAS_SKLEARN = True
    except Exception:
        HAS_SKLEARN = False

# Desativa relatórios de profiling pesados (para manter saídas mínimas)
HAS_PROFILING = False

# Configurações gerais de visual
warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="notebook")


@dataclass(frozen=True)
class EDAConfig:
    """Configuração da EDA (parâmetros de execução e limites).

    - data_path: caminho do arquivo Excel de entrada
    - output_dir: pasta onde os resultados serão salvos
    - corr_threshold: limite para considerar correlação alta
    - max_hist_cols: quantos histogramas (variáveis) salvar
    - max_outlier_cols: quantos boxplots (top outliers) salvar
    - max_scatter_pairs: quantos gráficos de dispersão salvar
    - max_heatmap_vars: máximo de variáveis no mapa de calor
    - clean_output_dir: limpar a pasta de saída antes de salvar
    - report_filename: nome do arquivo de relatório (Q&A)
    """
    data_path: str
    output_dir: str
    corr_threshold: float = 0.8
    max_hist_cols: int = 2
    max_outlier_cols: int = 2
    max_scatter_pairs: int = 2
    max_heatmap_vars: int = 8
    clean_output_dir: bool = True
    report_filename: str = 'Relatorio_EDA.txt'


class FileWriter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_dir(self) -> None:
        try:
            for name in os.listdir(self.output_dir):
                path = os.path.join(self.output_dir, name)
                if os.path.isfile(path) and (name.endswith(('.png', '.csv', '.txt', '.html'))):
                    os.remove(path)
        except FileNotFoundError:
            os.makedirs(self.output_dir, exist_ok=True)

    def savefig(self, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        # >>> NOVO: revalida a cada salvamento
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def write_report(self, filename: str, lines: List[str]) -> str:
        # Escreve uma lista de linhas em um arquivo .txt
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        return path


class EDAReport:
    """Orquestra a EDA respondendo a cada pergunta com Q&A e gráficos mínimos.

    Esta classe concentra as etapas da EDA em métodos separados que
    respondem a perguntas específicas. Assim, o fluxo fica claro e
    fácil de adaptar.
    """
    def __init__(self, df: pd.DataFrame, config: EDAConfig) -> None:
        # Cria uma cópia do DataFrame para evitar alterar o original
        self.df = df.copy()
        # Guarda a configuração
        self.config = config
        # Cria um utilitário para salvar arquivos/figuras
        self.writer = FileWriter(config.output_dir)
        # Limpa a pasta de saída se estiver habilitado
        if self.config.clean_output_dir:
            self.writer.clean_dir()
        # Identifica colunas numéricas e categóricas
        self.numeric_cols: List[str] = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols: List[str] = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Filtra colunas numéricas que parecem IDs (não servem para análise estatística)
        self.numeric_cols_model: List[str] = self._filter_id_like_numeric(self.numeric_cols)
        # Acumula linhas do relatório de perguntas e respostas
        self.report_lines: List[str] = []

    def _filter_id_like_numeric(self, cols: List[str]) -> List[str]:
        """Heurística: remove colunas numéricas provavelmente identificadores (ex.: ID, COD...).

        Ideia: se o nome parece conter "ID", "COD", "MUN" etc. ou se a coluna
        inteira é quase só valores únicos inteiros, tratamos como identificador.
        """
        keep: List[str] = []
        for col in cols:
            s = self.df[col]
            name_u = str(col).upper()
            # Sinalização por nome (ex.: ID, COD, MUN etc.)
            if any(tok in name_u for tok in ['ID', 'COD', 'CODE', 'MUN', 'CEP', 'CPF', 'CNPJ', 'NUM', 'NR']):
                continue
            # Remove valores ausentes para analisar a distribuição
            s_nonnull = s.dropna()
            if s_nonnull.empty:
                keep.append(col)
                continue
            # Taxa de valores únicos: muito alta pode ser "identificador"
            unique_ratio = s_nonnull.nunique() / len(s_nonnull)
            if pd.api.types.is_integer_dtype(s.dtype) and unique_ratio > 0.9:
                continue
            keep.append(col)
        # Se tudo foi filtrado, devolve a lista original para não perder análise
        return keep if keep else cols

    def _print_qa(self, pergunta: str, respostas: List[str]) -> None:
        """Imprime e acumula Q&A para o relatório final.

        Também garante que o relatório tenha a mesma estrutura mostrada no console.
        """
        print(f"Pergunta: {pergunta}")
        for linha in respostas:
            print(f"Resposta: {linha}")
        self.report_lines.append(f"Pergunta: {pergunta}")
        self.report_lines.extend([f"Resposta: {linha}" for linha in respostas])
        self.report_lines.append("")

    # PERGUNTA 1: Quantidade de linhas/colunas e tipos de dados
    def summarize_shape_and_types(self) -> None:
        # Obtém número de linhas e colunas
        n_rows, n_cols = self.df.shape
        # Conta quantas colunas de cada tipo existem (int, float, object...)
        dtypes_summary = self.df.dtypes.value_counts().to_string()
        # Complementa com uso de info(memory_usage) da própria biblioteca
        info_buf = io.StringIO()
        self.df.info(buf=info_buf, memory_usage='deep')  # uso automático da lib
        mem_line = [ln for ln in info_buf.getvalue().splitlines() if 'memory usage' in ln.lower()]
        mem_text = mem_line[0] if mem_line else ""
        # Monta respostas claras
        answer = [
            f"linhas={n_rows}, colunas={n_cols}",
            f"variáveis numéricas modeláveis={len(self.numeric_cols_model)}, categóricas={len(self.categorical_cols)}",
            "Tipos de variáveis (contagem):",
            dtypes_summary,
        ]
        if mem_text:
            answer.append(f"Uso de memória (aprox.): {mem_text}")
        # Registra pergunta e respostas
        self._print_qa("Quantas linhas e colunas existem? E quais os tipos?", answer)

    # PERGUNTA 2: Linhas duplicadas
    def analyze_duplicates(self) -> None:
        # Conta quantas linhas são cópias de outras
        num_dup = int(self.df.duplicated().sum())
        answer = [f"linhas duplicadas={num_dup}"]
        if num_dup > 0:
            answer.append("Sugestão: avaliar remoção com df.drop_duplicates().")
        self._print_qa("Há dados duplicados?", answer)

    # PERGUNTA 3: Valores faltantes (ausentes)
    def analyze_missingness(self) -> None:
        # Calcula percentual de faltantes por coluna e ordena do maior para o menor
        missing_pct = (self.df.isna().mean() * 100.0).sort_values(ascending=False)
        # Seleciona as 10 colunas com maiores faltantes para exibição
        top_missing = missing_pct.head(10)
        # Verifica se existe ao menos uma coluna com faltantes
        has_missing = float(missing_pct.max()) > 0.0
        # Prepara respostas em linguagem simples
        answer = [
            f"% faltantes (máximo)={missing_pct.max():.1f}%",
            "Top colunas com faltantes:" if has_missing else "Sem valores faltantes detectados."
        ]
        if has_missing:
            # Lista as colunas com maior percentual de faltantes
            answer.extend([f"- {c}: {v:.1f}%" for c, v in top_missing.items()])
        # Registra pergunta e respostas
        self._print_qa("Qual a porcentagem de valores ausentes?", answer)

        # Visualizações usando missingno (automático) e um barplot simples
        if has_missing:
            # missingno matrix: visão geral dos NA por linha/coluna
            if HAS_MISSINGNO:
                plt.figure(figsize=(10, 6))
                msno.matrix(self.df, fontsize=8)
                self.writer.savefig('faltantes_missingno_matrix.png')

                plt.figure(figsize=(10, 6))
                msno.bar(self.df, fontsize=8)
                self.writer.savefig('faltantes_missingno_bar.png')

            # Barplot manual focando nas top colunas com NA
            top_plot = missing_pct.head(20)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_plot.values, y=top_plot.index, orient='h')
            plt.xlabel('% faltantes')
            plt.title('Top colunas por faltantes')
            self.writer.savefig('faltantes_top_colunas.png')

    # PERGUNTA 4: Distribuições das variáveis numéricas
    def analyze_distributions(self) -> None:
        # Se não há numéricas úteis, informa e retorna
        if not self.numeric_cols_model:
            self._print_qa("Qual o tipo de distribuição das variáveis?", ["Sem variáveis numéricas modeláveis."])
            return

        # Cria um resumo estatístico e calcula assimetria e curtose
        desc = self.df[self.numeric_cols_model].describe().T
        desc['skew'] = self.df[self.numeric_cols_model].skew(numeric_only=True)
        desc['kurtosis'] = self.df[self.numeric_cols_model].kurtosis(numeric_only=True)
        # Ordena colunas por maior assimetria para identificar distribuições "tortas"
        high_skew = desc['skew'].abs().sort_values(ascending=False)
        # Escolhe poucas variáveis para salvar gráficos (para manter a pasta limpa)
        top_skew_cols = [c for c in high_skew.index if self.df[c].dropna().nunique() > 1][: self.config.max_hist_cols]

        # Escreve de forma simples as variáveis mais assimétricas
        answer = [
            f"variáveis numéricas modeláveis={len(self.numeric_cols_model)}",
            "Top variáveis por |assimetria (skew)|:" if len(top_skew_cols) else "Distribuições parecem pouco assimétricas.",
        ]
        answer.extend([f"- {c}: skew={desc.loc[c, 'skew']:.2f}, curtose={desc.loc[c, 'kurtosis']:.2f}" for c in top_skew_cols])
        self._print_qa("Qual o tipo de distribuição das variáveis?", answer)

        # Para cada variável escolhida, salva 1 ou 2 gráficos (normal e log1p se fizer sentido)
        for col in top_skew_cols:
            data = self.df[col].dropna()
            # Exige quantidade mínima de dados e mais de 1 valor distinto
            if data.size < 20 or data.nunique() < 2:
                continue
            # Histograma padrão com curva de densidade
            plt.figure(figsize=(8, 5))
            sns.histplot(data=data, kde=True, bins=30)
            plt.title(f'Distribuição: {col} (skew={float(data.skew()):.2f})')
            self.writer.savefig(f'distribuicao_{col}.png')

            # Se todos os valores são positivos e a assimetria é alta, tenta log1p
            if (data > 0).all() and abs(float(data.skew())) > 1.0:
                plt.figure(figsize=(8, 5))
                sns.histplot(data=np.log1p(data), kde=True, bins=30)
                plt.title(f'Distribuição log1p: {col}')
                self.writer.savefig(f'distribuicao_log1p_{col}.png')

        # Para dados categóricos, mostra um gráfico simples das categorias mais comuns
        if self.categorical_cols:
            # Escolhe a coluna categórica com mais valores preenchidos
            cat = max(self.categorical_cols, key=lambda c: self.df[c].notna().sum())
            vc = self.df[cat].astype(str).value_counts(dropna=False).head(10)
            self._print_qa("Resumo de variáveis categóricas?", [f"Exibindo top categorias para '{cat}'."])
            if vc.size > 1:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=vc.values, y=vc.index, orient='h')
                plt.title(f'Top categorias: {cat}')
                self.writer.savefig(f'top_categorias_{cat}.png')

    # PERGUNTA 5: Outliers (valores muito fora do padrão)
    def analyze_outliers(self) -> None:
        # Se não há numéricas úteis, informa e retorna
        if not self.numeric_cols_model:
            self._print_qa("Há outliers presentes?", ["Sem variáveis numéricas modeláveis."])
            return
        # Método manual (IQR) já implementado
        outlier_counts: List[Tuple[str, int]] = []
        for col in self.numeric_cols_model:
            series = self.df[col].dropna()
            # Se a variável não tem variedade de valores, não há outliers definíveis
            if series.nunique() < 2:
                outlier_counts.append((col, 0))
                continue
            # Quartil 1 e Quartil 3 + Intervalo Interquartil (IQR)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            # Limites "esperados" (IQR*1.5)
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            # Conta quantos pontos estão abaixo do limite inferior ou acima do superior
            num_outliers = int(((series < lower) | (series > upper)).sum())
            outlier_counts.append((col, num_outliers))
        outlier_counts.sort(key=lambda x: x[1], reverse=True)

        # Adiciona alternativa automática baseada em z-score (se scipy disponível)
        zscore_summary: List[str] = []
        if HAS_SCIPY and self.numeric_cols_model:
            # Calcula z-score coluna a coluna e conta valores com |z|>3
            z_out = {}
            for col in self.numeric_cols_model:
                s = self.df[col]
                if s.dropna().nunique() < 2:
                    z_out[col] = 0
                    continue
                z = zscore(s.dropna())
                # Alinhar de volta aos índices originais com NaN para ausentes
                z_full = pd.Series(index=s.index, dtype=float)
                z_full.loc[s.dropna().index] = z
                z_out[col] = int((z_full.abs() > 3).sum())
            # Pega top variáveis por z-outliers
            z_top = sorted(z_out.items(), key=lambda kv: kv[1], reverse=True)[:5]
            for col, cnt in z_top:
                if cnt > 0:
                    zscore_summary.append(f"- {col}: {cnt} (|z|>3)")

        # Monta respostas em linguagem simples
        top_outliers = [f"- {c}: {n}" for c, n in outlier_counts[:5] if n > 0]
        answer = ["Sem outliers relevantes detectados." if not top_outliers else "Top variáveis com mais outliers (IQR):"]
        answer.extend(top_outliers)
        if zscore_summary:
            answer.append("Alternativa automática (z-score |z|>3):")
            answer.extend(zscore_summary)
        self._print_qa("Há outliers presentes?", answer)

        # Salva um número pequeno de boxplots das variáveis com mais outliers
        for col, n in outlier_counts[: self.config.max_outlier_cols]:
            if n <= 0:
                continue
            series = self.df[col].dropna()
            if series.nunique() < 2:
                continue
            plt.figure(figsize=(8, 3))
            sns.boxplot(x=series, orient='h')
            plt.title(f'Boxplot: {col} (outliers={n})')
            self.writer.savefig(f'boxplot_{col}.png')

    # ETAPA OPCIONAL: Imputação em cópia (mediana/moda) e comparação de impacto
    def impute_copy_and_compare(self) -> None:
        # Cria uma cópia para imputar sem alterar o DataFrame original
        df_imp = self.df.copy()
        # Imputa numéricas com mediana
        num_imputed, cat_imputed = 0, 0
        for col in self.numeric_cols_model:
            median_val = df_imp[col].median()
            before_na = int(df_imp[col].isna().sum())
            if before_na > 0:
                df_imp[col] = df_imp[col].fillna(median_val)
                num_imputed += before_na
        # Imputa categóricas com moda (mais frequente) ou marcador "MISSING"
        for col in self.categorical_cols:
            mode_vals = df_imp[col].mode(dropna=True)
            fill_val = (mode_vals.iloc[0] if len(mode_vals) > 0 else 'MISSING')
            before_na = int(df_imp[col].isna().sum())
            if before_na > 0:
                df_imp[col] = df_imp[col].fillna(fill_val)
                cat_imputed += before_na
        # Compara faltantes totais antes/depois
        before_total_na = int(self.df.isna().sum().sum())
        after_total_na = int(df_imp.isna().sum().sum())
        # Compara impacto em médias de numéricas (variação percentual)
        impact_lines: List[str] = []
        for col in self.numeric_cols_model[:10]:  # limita a 10 para relatório enxuto
            orig_mean = self.df[col].mean()
            imp_mean = df_imp[col].mean()
            if pd.isna(orig_mean) or pd.isna(imp_mean):
                continue
            if orig_mean == 0:
                delta = abs(imp_mean - orig_mean)
                if delta > 0:
                    impact_lines.append(f"- {col}: mudança absoluta média={delta:.4f}")
            else:
                rel = abs(imp_mean - orig_mean) / abs(orig_mean)
                if rel >= 0.01:  # >= 1%
                    impact_lines.append(f"- {col}: mudança média≈{rel*100:.1f}%")
        # Q&A da imputação
        ans = [
            f"Imputação em cópia aplicada (numéricas→mediana, categóricas→moda).",
            f"Faltantes antes={before_total_na}, depois={after_total_na} (no DataFrame imputado).",
            f"Células imputadas: numéricas={num_imputed}, categóricas={cat_imputed}.",
        ]
        if impact_lines:
            ans.append("Impacto relevante nas médias (amostra):")
            ans.extend(impact_lines)
        else:
            ans.append("Impacto nas médias: baixo/negligenciável na amostra avaliada.")
        self._print_qa("Imputação (cópia): o que foi feito e qual o impacto?", ans)

    # ETAPA OPCIONAL: Modelos para valores mais prováveis (Bayes + Florestas)
    def model_based_most_probable_values(self) -> None:
        if not HAS_SKLEARN:
            self._print_qa("Valores mais prováveis por modelos?", ["Biblioteca de modelagem indisponível (sklearn)."])
            return
        # Seleciona colunas com mais faltantes (numéricas e categóricas)
        missing_ratio = self.df.isna().mean().sort_values(ascending=False)
        numeric_targets = [c for c in missing_ratio.index if c in self.numeric_cols_model][:3]
        categorical_targets = [c for c in missing_ratio.index if c in self.categorical_cols][:2]
        lines: List[str] = []
        suggestions_rows: List[Dict[str, object]] = []

        # Base numérica de features
        X_num_full = self.df[self.numeric_cols_model].copy()
        for c in self.numeric_cols_model:
            X_num_full[c] = X_num_full[c].fillna(X_num_full[c].median())

        # Predição para numéricas com IterativeImputer (BayesianRidge) e KNNImputer baseline; RFRegressor com validação
        if numeric_targets:
            # Iterative/BayesianRidge para imputação por cópia (sugestão)
            try:
                imp = IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=20, sample_posterior=False)
                X_imp = pd.DataFrame(imp.fit_transform(self.df[self.numeric_cols_model]), columns=self.numeric_cols_model, index=self.df.index)
            except Exception:
                X_imp = None
            # KNNImputer baseline
            try:
                knn = KNNImputer(n_neighbors=5, weights='distance')
                X_knn = pd.DataFrame(knn.fit_transform(self.df[self.numeric_cols_model]), columns=self.numeric_cols_model, index=self.df.index)
            except Exception:
                X_knn = None

            for col in numeric_targets:
                mask_missing = self.df[col].isna()
                nmiss = int(mask_missing.sum())
                if nmiss == 0:
                    continue
                # Sugestões Bayes/KNN 
                if X_imp is not None:
                    preds_bayes = X_imp.loc[mask_missing, col]
                    lines.append(f"- {col} (Bayes): sugeridos {nmiss} valores; média prev≈{preds_bayes.mean():.2f}, mediana≈{preds_bayes.median():.2f}")
                    for idx, val in preds_bayes.head(10).items():
                        suggestions_rows.append({"coluna": col, "linha": int(idx), "metodo": "Bayes", "sugerido": float(val)})
                if X_knn is not None:
                    preds_knn = X_knn.loc[mask_missing, col]
                    lines.append(f"- {col} (KNN): média prev≈{preds_knn.mean():.2f}, mediana≈{preds_knn.median():.2f}")
                    for idx, val in preds_knn.head(10).items():
                        suggestions_rows.append({"coluna": col, "linha": int(idx), "metodo": "KNN", "sugerido": float(val)})
                # RandomForestRegressor com validação (R2/MAE em dados não faltantes)
                try:
                    df_nonnull = self.df[self.numeric_cols_model + [col]].dropna()
                    if df_nonnull.shape[0] >= 200:
                        X = df_nonnull[self.numeric_cols_model].copy()
                        for c in self.numeric_cols_model:
                            X[c] = X[c].fillna(X[c].median())
                        y = df_nonnull[col]
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                        rfr = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
                        rfr.fit(Xtr, ytr)
                        yhat = rfr.predict(Xte)
                        r2 = r2_score(yte, yhat)
                        mae = mean_absolute_error(yte, yhat)
                        lines.append(f"- {col} (RF-Reg): R2≈{r2:.2f}, MAE≈{mae:.2f}")
                        if nmiss > 0:
                            Xmiss = X_num_full.loc[mask_missing, :]
                            preds_rf = rfr.predict(Xmiss)
                            for idx, val in zip(Xmiss.index[:10], preds_rf[:10]):
                                suggestions_rows.append({"coluna": col, "linha": int(idx), "metodo": "RF-Reg", "sugerido": float(val)})
                    else:
                        lines.append(f"- {col} (RF-Reg): dados insuficientes para validação (n<200).")
                except Exception:
                    lines.append(f"- {col} (RF-Reg): não foi possível treinar/validar.")

        # Predição para categóricas com RandomForestClassifier
        if categorical_targets and len(self.numeric_cols_model) >= 1:
            for col in categorical_targets:
                try:
                    df_nonnull = self.df[self.numeric_cols_model + [col]].dropna()
                    if df_nonnull.shape[0] < 100:
                        lines.append(f"- {col} (RF-Cla): dados insuficientes para treinar (n<{100}).")
                        continue
                    X = df_nonnull[self.numeric_cols_model].copy()
                    for c in self.numeric_cols_model:
                        X[c] = X[c].fillna(X[c].median())
                    y = df_nonnull[col].astype(str)
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
                    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                    rf.fit(Xtr, ytr)
                    acc = accuracy_score(yte, rf.predict(Xte)) if yte.size>0 else np.nan
                    mask_missing = self.df[col].isna()
                    nmiss = int(mask_missing.sum())
                    if nmiss > 0:
                        Xmiss = X_num_full.loc[mask_missing, :]
                        preds = pd.Series(rf.predict(Xmiss))
                        top_pred = preds.value_counts().head(1)
                        if not top_pred.empty:
                            cls, cnt = top_pred.index[0], int(top_pred.iloc[0])
                            lines.append(f"- {col} (RF-Cla): acc≈{acc:.2f}; mais provável='{cls}' (~{cnt}/{nmiss}).")
                            for idx, val in preds.head(10).items():
                                suggestions_rows.append({"coluna": col, "linha": int(idx), "metodo": "RF-Cla", "sugerido": str(val)})
                        else:
                            lines.append(f"- {col} (RF-Cla): acc≈{acc:.2f}; sem predições significativas.")
                    else:
                        lines.append(f"- {col} (RF-Cla): sem faltantes para prever.")
                except Exception:
                    lines.append(f"- {col} (RF-Cla): não foi possível treinar/prever.")

        # Persistência de sugestões (CSV pequeno)
        if suggestions_rows:
            sugg_df = pd.DataFrame(suggestions_rows)
            out_path = os.path.join(self.config.output_dir, 'sugestoes_valores_provaveis.csv')
            try:
                sugg_df.to_csv(out_path, index=False)
                lines.append(f"Sugestões salvas em: {out_path}")
            except Exception:
                lines.append("Não foi possível salvar o CSV de sugestões.")

        if not lines:
            lines = ["Sem colunas com faltantes suficientes para modelagem ou pré-requisitos ausentes."]
        self._print_qa("Valores mais prováveis (modelos Bayes/Florestas/KNN/RF-Reg):", lines)

    # ETAPA OPCIONAL: Relato de suavização (winsorização) e anomalias (IsolationForest)
    def outlier_smoothing_and_iforest_report(self) -> None:
        # Winsorização (clipping em quantis) apenas para medir impacto (não persiste)
        wins_impacts: List[str] = []
        for col in self.numeric_cols_model[:10]:  # limita para não poluir
            s = self.df[col].dropna()
            if s.size < 20:
                continue
            q01, q99 = s.quantile(0.01), s.quantile(0.99)
            s_w = s.clip(lower=q01, upper=q99)
            std_orig = s.std()
            std_w = s_w.std()
            if std_orig and std_w:
                red = (std_orig - std_w) / std_orig
                if red >= 0.05:  # redução >=5%
                    wins_impacts.append(f"- {col}: redução do desvio-padrão≈{red*100:.1f}% (winsor 1%-99%)")
        # IsolationForest: conta anomalias, sem alterar dados
        iforest_line = ""
        if HAS_SKLEARN and len(self.numeric_cols_model) >= 2 and self.df.shape[0] >= 200:
            X = self.df[self.numeric_cols_model].copy()
            # Imputa temporariamente NA só para o modelo (mediana)
            for c in self.numeric_cols_model:
                X[c] = X[c].fillna(X[c].median())
            try:
                clf = IsolationForest(random_state=42, contamination='auto')
                y_pred = clf.fit_predict(X.values)
                anomalies = int((y_pred == -1).sum())
                iforest_line = f"Anomalias detectadas (IsolationForest): {anomalies}."
            except Exception:
                iforest_line = "IsolationForest indisponível/não convergiu para estes dados."
        # Q&A
        ans = []
        if wins_impacts:
            ans.append("Winsorização (relato de impacto, 1%-99%):")
            ans.extend(wins_impacts)
        else:
            ans.append("Winsorização: sem impacto relevante nas amostras avaliadas.")
        if iforest_line:
            ans.append(iforest_line)
        self._print_qa("Suavização/Anomalias: quais resultados?", ans)

    # PERGUNTA 6: Correlações entre variáveis numéricas
    def analyze_correlations(self) -> None:
        # Se não há numéricas suficientes, informa e retorna
        if len(self.numeric_cols_model) < 2:
            self._print_qa("Qual a correlação existente entre as variáveis?", ["Menos de duas variáveis numéricas modeláveis."])
            return
        # Considera apenas as colunas numéricas úteis
        df_num = self.df[self.numeric_cols_model].copy()
        # Calcula correlação de Pearson (relacionamento linear)
        pearson = df_num.corr(method='pearson')
        # Alternativa automática: matriz empilhada (longa) para inspeção programática
        corr_long = pearson.where(~np.triu(np.ones(pearson.shape), k=0).astype(bool))\
                           .stack()\
                           .abs()\
                           .sort_values(ascending=False)
        # Seleciona variáveis com maior variância para um mapa de calor mais legível
        var_rank = df_num.var().sort_values(ascending=False)
        heat_cols = var_rank.head(self.config.max_heatmap_vars).index.tolist()
        heat = df_num[heat_cols].corr(method='pearson')

        # Procura os pares com maior correlação absoluta (com limiar)
        pairs: List[Tuple[str, str, float]] = []
        cols = pearson.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = float(abs(pearson.iloc[i, j]))
                if val >= self.config.corr_threshold:
                    pairs.append((cols[i], cols[j], val))
        # Ordena do mais correlacionado para o menos e limita
        pairs.sort(key=lambda t: t[2], reverse=True)
        high_pairs = pairs[:5]
        # Escreve respostas simples e diretas
        answer = [
            f"variáveis avaliadas (numéricas modeláveis)={len(self.numeric_cols_model)}",
            "Pares com alta correlação (|r|>=%.2f):" % self.config.corr_threshold if high_pairs else "Sem pares com alta correlação acima do limiar.",
        ]
        answer.extend([f"- {a} ~ {b}: r={v:.2f}" for a, b, v in high_pairs])
        self._print_qa("Qual a correlação existente entre as variáveis?", answer)

        # Salva um mapa de calor e poucos gráficos de dispersão para ilustrar
        if len(heat_cols) >= 2:
            plt.figure(figsize=(min(12, 1.1 * len(heat_cols)), min(10, 1.0 * len(heat_cols))))
            sns.heatmap(heat, cmap='vlag', center=0, square=True)
            plt.title('Mapa de Calor de Correlação (maior variância)')
            self.writer.savefig('mapa_calor_correlacao.png')
        for a, b, v in high_pairs[: self.config.max_scatter_pairs]:
            sub = df_num[[a, b]].dropna()
            # Evita gráfico com muito poucos pontos
            if sub.shape[0] < 30:
                continue
            plt.figure(figsize=(6, 5))
            sns.regplot(x=sub[a], y=sub[b], scatter_kws={'s': 12, 'alpha': 0.6}, line_kws={'color': 'red'})
            plt.title(f'{a} vs {b} (r={v:.2f})')
            self.writer.savefig(f'dispersao_{a}_vs_{b}.png')

    # PERGUNTA 7: Resumo numérico global (métricas chave)
    def summarize_numeric_brief(self) -> None:
        # Se não há numéricas úteis, informa e retorna
        if not self.numeric_cols_model:
            self._print_qa("Qual o resumo estatístico das variáveis numéricas?", ["Sem variáveis numéricas modeláveis."])
            return
        # Calcula medidas básicas: média, desvio, quartis
        desc = self.df[self.numeric_cols_model].describe().T
        metrics = []
        # Limita a listagem a 10 variáveis para o relatório ficar curto
        for col in desc.index[:10]:
            mean = desc.loc[col, 'mean']
            std = desc.loc[col, 'std']
            q1 = desc.loc[col, '25%']
            q3 = desc.loc[col, '75%']
            metrics.append(f"- {col}: média={mean:.2f}, desvio={std:.2f}, Q1={q1:.2f}, Q3={q3:.2f}")
        answer = ["Principais métricas (amostra de até 10 variáveis):"] + metrics
        self._print_qa("Qual o resumo estatístico das variáveis numéricas?", answer)

    # BLOCO: Diagnóstico de qualidade de dados (inconsistências)
    def data_quality_diagnostics(self) -> None:
        const_cols = []
        high_card = []
        negative_cols = []
        date_issues = []
        # Colunas constantes e alta cardinalidade categórica
        for col in self.df.columns:
            s = self.df[col]
            nunq = s.nunique(dropna=False)
            if nunq <= 1:
                const_cols.append(col)
        for col in self.categorical_cols:
            s = self.df[col]
            nonnull = s.dropna()
            if len(nonnull) == 0:
                continue
            ratio = nonnull.nunique() / len(nonnull)
            if ratio > 0.5 and nonnull.nunique() >= 50:
                high_card.append(col)
        # Negativos em numéricas (apenas sinaliza; não sabemos a semântica)
        for col in self.numeric_cols_model:
            s = self.df[col]
            neg_count = int((s < 0).sum())
            if neg_count > 0:
                negative_cols.append(f"- {col}: {neg_count} valores negativos")
        # Datas inválidas: tenta analisar objetos como datas
        for col in self.df.select_dtypes(include=['object']).columns:
            s = self.df[col]
            try:
                parsed = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
                nonnull = s.notna().sum()
                invalid = parsed.isna().sum()
                if nonnull > 0 and invalid / max(nonnull, 1) > 0.3:
                    date_issues.append(f"- {col}: alta taxa de datas inválidas/parse falho")
            except Exception:
                continue
        # Monta Q&A
        ans = []
        if const_cols:
            ans.append("Colunas constantes:")
            ans.append(", ".join(const_cols[:15]) + (" ..." if len(const_cols) > 15 else ""))
        else:
            ans.append("Colunas constantes: nenhuma encontrada.")
        if high_card:
            ans.append("Colunas categóricas com alta cardinalidade:")
            ans.append(", ".join(high_card[:15]) + (" ..." if len(high_card) > 15 else ""))
        else:
            ans.append("Alta cardinalidade categórica: não relevante nas top colunas.")
        if negative_cols:
            ans.append("Valores negativos em colunas numéricas:")
            ans.extend(negative_cols[:10])
        else:
            ans.append("Negativos: não encontrados nas colunas analisadas.")
        if date_issues:
            ans.append("Possíveis problemas de datas:")
            ans.extend(date_issues[:10])
        else:
            ans.append("Datas inválidas: não foram detectados problemas relevantes (parse).")
        self._print_qa("Diagnóstico de qualidade de dados: inconsistências detectadas?", ans)

    # PERGUNTA 8: Esta EDA está completa?
    def summarize_eda_completeness(self) -> None:
        # Resumo final com checagem de itens cobertos e próximos passos sugeridos
        covered = [
            "Forma e tipos (shape/dtypes)",
            "Duplicados",
            "Faltantes (com missingno)",
            "Distribuições (skew/curtose + gráficos focados)",
            "Outliers (IQR e alternativa z-score)",
            "Correlação (mapa de calor e dispersões)",
            "Resumo estatístico (métricas-chave)",
            "Imputação em cópia (mediana/moda) e impacto",
            "Modelos (Bayes/Florestas) para valores prováveis",
            "Winsorização & IsolationForest (relato)",
            "Diagnóstico de qualidade de dados",
        ]
        next_steps = [
            "Normalização fina de tipos (datas, booleanos, categorias)",
            "Associações categóricas (Cramér's V) e numérico-categóricas",
            "Testes de normalidade/robustez (quando aplicável)",
            "Análise de importância/seleção de variáveis (dependendo do objetivo)",
        ]
        answer = ["Coberto: " + "; ".join(covered), "Próximos passos sugeridos: " + "; ".join(next_steps)]
        self._print_qa("A EDA está completa? O que falta?", answer)

    def save_report(self) -> None:
        # Escreve o relatório final de Q&A na pasta de saída
        self.writer.write_report(self.config.report_filename, self.report_lines)

    # (Opcional) Gera um arquivo separado com explicações linha a linha do código
    def _explain_line_portuguese(self, line: str) -> str:
        s = line.strip()
        if s == '':
            return 'Linha em branco para organização visual.'
        if s.startswith('#'):
            return 'Comentário descritivo; não executa.'
        if s.startswith('import '):
            return 'Importa um módulo para uso no script.'
        if s.startswith('from ') and ' import ' in s:
            return 'Importa símbolos específicos de um módulo.'
        if s.startswith('@dataclass'):
            return 'Decorator que transforma a classe em uma dataclass (gera métodos úteis automaticamente).'
        if s.startswith('class '):
            return 'Declaração de classe que agrupa dados e/ou comportamentos.'
        if s.startswith('def '):
            return 'Declaração de função/método com uma responsabilidade específica.'
        if s.startswith('return '):
            return 'Retorna um valor do escopo da função/método.'
        if s.startswith('if __name__'):
            return 'Ponto de entrada do script quando executado diretamente.'
        if 'read_excel' in s:
            return 'Carrega dados de uma planilha Excel em um DataFrame pandas.'
        if 'duplicated()' in s:
            return 'Calcula máscara/quantidade de linhas duplicadas.'
        if '.isna()' in s or '.isnull()' in s:
            return 'Avalia valores ausentes (NaN) no DataFrame.'
        if 'describe()' in s:
            return 'Resumo estatístico das variáveis (contagem, média, quartis, etc.).'
        if 'skew' in s and 'sns.' not in s:
            return 'Calcula a assimetria (skewness) de distribuições numéricas.'
        if 'kurtosis' in s:
            return 'Calcula a curtose (achatamento) de distribuições numéricas.'
        if 'corr(' in s:
            return 'Calcula a matriz de correlação entre variáveis numéricas.'
        if 'sns.histplot' in s:
            return 'Plota histograma com densidade (KDE) para visualizar distribuição.'
        if 'sns.boxplot' in s:
            return 'Plota boxplot para inspecionar dispersão e outliers.'
        if 'sns.heatmap' in s:
            return 'Plota um mapa de calor, útil para visualizar correlações.'
        if 'sns.regplot' in s:
            return 'Plota diagrama de dispersão com linha de regressão (tendência).'
        if 'print(' in s:
            return 'Envia mensagem ao console (saída padrão).'
        if 'plt.title' in s or 'plt.xlabel' in s or 'plt.ylabel' in s:
            return 'Configura elementos textuais do gráfico (título/legendas).'
        if 'plt.savefig' in s or 'writer.savefig' in s:
            return 'Exporta o gráfico para arquivo de imagem (.png).'
        return 'Linha de lógica/controle de fluxo ou manipulação de dados.'

    def write_code_explanation_file(self) -> None:
        # Cria um arquivo separado com explicação linha a linha do código
        lines_out: List[str] = ["==== Explicação didática do código (linha a linha) ===="]
        try:
            source_path = os.path.abspath(__file__)
            with open(source_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for idx, raw in enumerate(lines, start=1):
                line = raw.rstrip('\n')
                explain = self._explain_line_portuguese(line)
                lines_out.append(f"L{idx:03d}: {line}")
                lines_out.append(f"Explicação: {explain}")
        except Exception as e:
            lines_out.append(f"Não foi possível carregar o código-fonte: {e}")
        self.writer.write_report('Explicacao_Codigo.txt', lines_out)


def main() -> None:
    # Monta a configuração com caminhos e limites de gráficos
    config = EDAConfig(
        data_path='/Users/akatsurada/Documents/INSPER/CS/Aula4/lab4/Mortalidade_Geral_2021_comUF_CID10.xlsx',
        output_dir='/Users/akatsurada/Documents/INSPER/CS/Aula4/eda_outputs',
        corr_threshold=0.8,
        max_hist_cols=2,
        max_outlier_cols=2,
        max_scatter_pairs=2,
        max_heatmap_vars=8,
        clean_output_dir=True,
        report_filename='Relatorio_EDA.txt',
    )

    # Lê a planilha Excel em um DataFrame (tabela de dados em memória)
    df = pd.read_excel(config.data_path)

    # Cria o orquestrador da EDA com os dados e a configuração
    report = EDAReport(df=df, config=config)

    # Chama, em ordem, as etapas que respondem cada pergunta da EDA
    report.summarize_shape_and_types()           # P1: Linhas/colunas e tipos
    report.analyze_duplicates()                  # P2: Duplicados
    report.analyze_missingness()                 # P3: Faltantes (com missingno)
    report.analyze_distributions()               # P4: Distribuições
    report.analyze_outliers()                    # P5: Outliers (IQR + z-score)
    report.impute_copy_and_compare()             # Extra: Imputação em cópia e impacto
    report.model_based_most_probable_values()    # Extra: Modelos (Bayes/Floresta) para valores prováveis
    report.outlier_smoothing_and_iforest_report()# Extra: Winsor & IsolationForest (relato)
    report.analyze_correlations()                # P6: Correlações
    report.summarize_numeric_brief()             # P7: Resumo estatístico
    report.data_quality_diagnostics()            # Extra: Diagnóstico de qualidade de dados
    report.summarize_eda_completeness()          # P8: Avaliação de completude

    # Salva o relatório final de Q&A
    report.save_report()

    # (Opcional) Gera um arquivo à parte com explicação didática do código
    report.write_code_explanation_file()

    # Mensagem final orientando onde estão os resultados
    print(f"Relatório consolidado e gráficos mínimos salvos em: {config.output_dir}")


if __name__ == '__main__':
    # Ponto de entrada do script: executa a função principal
    main()
