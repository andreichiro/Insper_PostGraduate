# Guia de Apresentação — Pipeline de Limpeza & Sobrevivência

> **Objetivo:** explicar termos, justificar decisões e **fundamentar com dados** (arquivos/figuras geradas).  
> **Formato:** Markdown puro, pronto para virar deck.

---

## Sumário

1) [Constantes e políticas — “as réguas do jogo”](#1-constantes-e-políticas--as-réguas-do-jogo)  
&emsp;1.1 [Degraus de receita e outliers](#11-degraus-de-receita-e-outliers)  
&emsp;1.2 [Janela de anos esperada](#12-janela-de-anos-esperada)  
&emsp;1.3 [Flags que “derrubam na hora” (sobrevivência)](#13-flags-que-derrubam-na-hora-sobrevivência)  
&emsp;1.4 [Tokens de NA e tipos preferidos](#14-tokens-de-na-e-tipos-preferidos)  
&emsp;1.5 [Não-negativas e negativas permitidas](#15-não-negativas-e-negativas-permitidas)  
&emsp;1.6 [Policy (política central)](#16-policy-política-central)

2) [Regras de qualidade que derrubam (tempo/realidade)](#2-regras-de-qualidade-que-derrubam-temporealidade)  
3) [Regras que tratam (consertos locais e seguros)](#3-regras-que-tratam-consertos-locais-e-seguros)  
4) [Inventário de flags (mapa de risco)](#4-inventário-de-flags-mapa-de-risco)  
5) [Limpeza focada em sobrevivência (motor)](#5-limpeza-focada-em-sobrevivência-motor)  
6) [Artefatos de QA e diagnósticos](#6-artefatos-de-qa-e-diagnósticos)  
7) [Dataset de sobrevivência (uma linha por firma, pronto p/ R)](#7-dataset-de-sobrevivência-uma-linha-por-firma-pronto-p-r)  
8) [Imputação avançada (MICE+RF) — opcional](#8-imputação-avançada-micerf--opcional)  
9) [Duplicatas, multicolinearidade, infinito, leakage](#9-duplicatas-multicolinearidade-infinito-leakage)  
10) [Fluxo executivo (resumo)](#10-fluxo-executivo-resumo)  
11) [Figuras obrigatórias (para slides)](#11-figuras-obrigatórias-para-slides)  
12) [Decisões sensíveis (com evidência)](#12-decisões-sensíveis-com-evidência)  
13) [Riscos (areias movediças) e mitigação](#13-riscos-areias-movediças-e-mitigação)  
14) [Roteiro de 6 minutos — versão completa](#14-roteiro-de-6-minutos--versão-completa)  
[Checklist de evidências (preencha com os números reais)](#checklist-de-evidências-preencha-com-os-números-reais)  
[Glossário essencial](#glossário-essencial)  
[Observações de engenharia](#observações-de-engenharia)

---

## 1) Hipóteses

### 1.1 Degraus de receita e outliers

**Termos (tradução simples)**  
- **Degraus de receita (`REVENUE_THRESHOLDS`)**: €1k, €10k, €100k, €1M, €10M — marcos para contar abaixo/igual/acima.  
- **Outlier por z-score (`Z_OUTLIER=3`)**: |z|≥3 indica extremo (≈ 3 desvios-padrão da média).  
- **IQR e cercas de Tukey (`IQR_FACTOR=1.5`)**: IQR=Q3−Q1 (miolo). Cercas **[Q1−1,5×IQR, Q3+1,5×IQR]**.

**Como provar nos dados**  
- `artifacts/revenue_thresholds_summary.csv` → **%≥€1M**, **%≥€10M**, **%≤€1k**.  
- Antes/depois: `artifacts/yearly_summary.csv` (mediana/média de `sales`) + `figs/sales_log10_hist.png`.  
- Outliers/winsor: `artifacts/winsor_caps_by_ind_year.csv` (limites por ind×ano) + `artifacts/fix_summary_all.csv` (contar `WINSOR[...]`).

**Decisão & porquê (trade-offs)**  
- **Winsor por ind×ano (1–99%)** estabiliza caudas sem “apagar” informação e respeita regimes setoriais/temporais.  
- Degraus comunicam **cobertura** (micro vs. mid vs. grandes).  
- `Z=3` e `1,5×IQR` são **conservadores**: raramente cortam “bom sinal”.

**Riscos & alternativas**  
- Sazonalidade extrema: reduzir p99→p98 (`Policy.winsor_upper=0.98`).  
- Setor ralo: usar **winsor global** (fallback).  
- **Para slides:** mostre **%≥€1M**, **linhas winsorizadas**, **mudança de mediana** pós-winsor.

---

### 1.2 Janela de anos esperada

**Termos**  
- **Janela automática (IQR+almofada):** estima `[lo,hi]` de `year` via Q1/Q3 e abre `±pad_years`.  
- **Fallback:** se `year` não varia, usa `[EXPECTED_YEAR_MIN, EXPECTED_YEAR_MAX]`.

**Como provar nos dados**  
- `artifacts/flags_before.csv` → flags `year_out_of_[lo,hi]` (limites no nome da flag).  
- Compare **% fora da janela** antes vs. depois (`flags_after.csv`).  
- Se `Policy.drop_out_of_window=False`, persiste como **flag** (não drop).

**Decisão & porquê**  
- **Default = flag** (não drop) — ano antigo pode ser histórico válido.

**Riscos & alternativas**  
- Para ceticismo: mostrar que a janela é **auto-detectada**, não arbitrária.

---

### 1.3 Flags que “derrubam na hora” (sobrevivência)

**Termos**  
- **ROW-level:** contradições na **linha** (ex.: `begin≥end`, após `exit_year`, sobreposição).  
- **FIRM-level:** contradições da **firma** (ex.: equity ≤0 crônica ≥3 anos).  
- **Sobreposição:** um balanço invade o próximo (duplica somas).

**Como provar nos dados**  
- `artifacts/drop_log.csv` (linha→motivos) + `figs/drops_by_reason.png`.  
- Temporal: `figs/temporal_logic_violations.png` + `artifacts/temporal_logic_audit.csv`.  
- Sobreposição: `balsheet_period_overlap` cai a ~0% no **depois**.

**Decisão & porquê**  
- Sobrevivência exige **linha do tempo coerente** → **DROP**.

**Riscos & alternativas**  
- Casos limítrofes: manter **auditoria visual** e não afrouxar regra.

---

### 1.4 Tokens de NA e tipos preferidos

**Termos**  
- **NA tokens:** “—”, “NULL”, etc. → **NA real**.  
- **Int64 nulo-seguro:** inteiro com NA (evita “0 mágico”).  
- **Coerção tolerante:** tenta converter; se falha, **avisa** (não quebra).

**Como provar nos dados**  
- `artifacts/type_coercion_report.csv` → contagem de colunas convertidas (slide “Coerções”).  
- `figs/missingness_top30_before_clean.png` para panorama **antes**.

**Decisão & porquê**  
- Evita **erros silenciosos** (ex.: “0” string virando número).

**Riscos & alternativas**  
- Muito `object` residual → ampliar dicionário de coerções.

---

### 1.5 Não-negativas e negativas permitidas

**Termos**  
- **Não-negativa:** grandeza que não faz sentido <0 (ex.: `inventories`).  
- **Negativa permitida:** lucro/prejuízo podem ser <0.

**Como provar nos dados**  
- `artifacts/nonnegativity_audit.csv` (colunas `negatives`, `neg_pct`).  
- `figs/nonnegativity_audit_before.png` (antes) vs `figs/nonnegativity_audit.png` (depois).

**Decisão & porquê**  
- **Impossível → NA** (melhor faltante que absurdo).

**Riscos & alternativas**  
- Campo setorial que aceita negativo? Mover de **NONNEGATIVE_COLS** para **ALLOW_NEGATIVE**.

---

### 1.6 Policy (política central)

**Termos**  
- **Policy:** botões imutáveis de tolerâncias/limiares (auditáveis).  
- Ex.: `accounting_rel_tol=0.10`, `accounting_abs_tol=1000`, `zombie_min_years=3`, `winsor_upper=0.99`.

**Como provar nos dados**  
- Contabilidade: `artifacts/identity_assets_eq_liab_plus_eq_partial.csv` (residual).  
- Zumbi: `artifacts/zombie_truncation_log.csv`.  
- Winsor: `artifacts/winsor_caps_by_ind_year.csv` + `artifacts/fix_summary_all.csv`.

**Decisão & porquê**  
- **Uma fonte** evita “ajustes invisíveis”; facilita auditoria.

**Riscos & alternativas**  
- Ajustar por **setor** apenas com evidência.

---

## 2) Regras de qualidade que derrubam (tempo/realidade)

### 2.1 Período quebrado (begin≥end) / begin/end fora do ano

**Termos**  
- **Exposição:** fração do ano (dias/365,25) — base da anualização.  
- **Ano desalinhado:** `year` fora de `[begin.year, end.year]`.

**Como provar nos dados**  
- `artifacts/balance_sheet_period_audit.csv` (colunas `begin_after_end`, `duration_days`).  
- `artifacts/flags_before.csv` → `begin_end_outside_observation_year`.  
- **Antes→Depois:** taxa deve ir a ~0% (slide “Temporal”).

**Decisão & porquê**  
- Sem período válido, **exposição** não é confiável → **DROP**.

**Riscos & alternativas**  
- Se sistemático por país, revisar **parsing de datas** antes do drop.

---

### 2.2 Impossibilidades de tempo (depois do exit, antes da fundação)

**Termos**  
- **Exit_year:** ano de saída/fechamento.  
- **Fundação:** início da firma.  
- **Relógio coerente:** `founded ≤ year ≤ exit`.

**Como provar nos dados**  
- `artifacts/temporal_logic_audit.csv` (colunas `row_after_exit_year`, `founded_after_exit`, `row_before_founded_year`).  
- `figs/temporal_logic_violations.png` com %.

**Decisão & porquê**  
- Afeta diretamente a **sobrevivência** → **DROP**.

**Riscos & alternativas**  
- Duvidoso? Tratar como **censurado** se `exit<t0` (já implementado).

---

### 2.3 Períodos patológicos: <90d; sobreposição

**Termos**  
- **Stub:** período curtíssimo.  
- **Overlap:** fim do A > início do B (mesma firma).

**Como provar nos dados**  
- `balance_sheet_period_audit.csv`: `duration_lt_360` (contexto).  
- `flags_before.csv`: `balsheet_period_overlap` → deve cair a ~0% depois.

**Decisão & porquê**  
- <90d é **ruído**; **overlap duplica** totais → **DROP**.

**Riscos & alternativas**  
- Muitos stubs legítimos? Ajustar 90→120d (policy).

---

### 2.4 Identidade contábil estourada

**Termos**  
- **Identidade:** Ativos ≈ Passivos + Patrimônio. Proxy: `intang+fixed+curr_assets`.  
- **Regra dupla:** **relativo >10%** **e** **absoluto >€1000**.

**Como provar nos dados**  
- `identity_assets_eq_liab_plus_eq_partial.csv` → `residual`.  
- Compare hist **antes/depois** (`figs/identity_fixed_vs_tang_intang_residual.png` ou hist custom).  
- `flags_before.csv`: `accounting_identity_large_mismatch` → ~0% depois.

**Decisão & porquê**  
- Evita contaminar **liquidez/alavancagem** → **DROP** estrito.

**Riscos & alternativas**  
- Arredondamento alto? subir p/ 15%; microfirmas? baixar €1000→€500.

---

### 2.5 Shell company (RPE absurdo, fora de tech)

**Termos**  
- **RPE:** `sales / labor_avg`.  
- **Whitelist tech (NACE 58–63):** admite produtividade muito alta.

**Como provar nos dados**  
- Contar **% RPE>limiar fora de tech** e **quanto cai após limpeza** (filtro/contagem).  
- `flags_before.csv`: `shell_company_extreme_rev_per_emp`.

**Decisão & porquê**  
- Protege contra “sinal mágico” por **reporte atípico**.

**Riscos & alternativas**  
- Ajustar limiar por indústria (benchmarks).

---

### 2.6 Firm-level (equity≤0 crônica; exit não-monótono; fundação mutante)

**Termos**  
- **Crônica (spell):** sequência de anos consecutivos.  
- **Não-monótono:** `exit_year` “volta no tempo”.

**Como provar nos dados**  
- `flags_before.csv`: `chronic_negative_equity_ge3y`, `exit_year_nonmonotonic`, `founded_year_nonconstant_within_firm`.  
- `zombie_truncation_log.csv` é outro truncamento (por vendas/emprego).

**Decisão & porquê**  
- Ruptura estrutural → **DROP da firma**.

**Riscos & alternativas**  
- Muitas reclassificações? Subir limiar de anos.

---

## 3) Regras que tratam (consertos locais e seguros)

### 3.1 Zeros estruturais → NA

**Termos**  
- **Zero estrutural:** quase todo mundo reporta **0**, sem negativos → “não se aplica/não reportado”.  
- Critérios: **≥95% zeros**, **≤5% positivos**, **sem negativos**, **n≥500**.

**Como provar nos dados**  
- `artifacts/structural_zero_candidates.csv` → `zeros_pct`, `positives_pct`, `recommend_zero_as_missing`.  
- `artifacts/fix_summary_all.csv` → `STRUCT_ZERO: 0→NaN`.

**Decisão & porquê**  
- Evita **puxar médias para baixo** por zeros informacionais.

**Riscos & alternativas**  
- Ajustar 95/5 ou `min_n`; documentar.

---

### 3.2 Shares [0,1], binários {0,1}, não-negativos

**Termos**  
- **Share/Proporção:** [0,1].  
- **Binário:** {0,1}.

**Como provar nos dados**  
- `inconsistency_per_column.csv` → % fora do domínio.  
- `fix_summary_all.csv` → `SHARE_OUTSIDE[0,1]: NaN`, `BINARY_OUTSIDE_01→NaN`, `NONNEGATIVE: neg→NaN`.

**Decisão & porquê**  
- **Impossível → NA** (sem chute).

**Riscos & alternativas**  
- Mapear {S/N, Y/N} antes da coerção.

---

### 3.3 Anualização (entrada/saída)

**Termos**  
- **Exposição:** dias/365,25.  
- **Anualizar:** escalar fluxos p/ ano cheio (**cap 2×**).  
- **Peso:** `exposure_weight` ∈ [0,1].

**Como provar nos dados**  
- `artifacts/partial_year_annualization_log.csv` → `rows_annualized`, `mean_multiplier`.

**Decisão & porquê**  
- Justo com meio-ano; evita viés “parece menor”.

**Riscos & alternativas**  
- Subir `min_days` p/ 120; baixar **cap** p/ setores calmos.

---

### 3.4 Winsorização por ind×ano

**Termos**  
- **Winsorizar:** trazer extremos até o percentil (sem remover).  
- **Caps:** limites lo/hi por grupo (reaplicáveis sem vazamento).

**Como provar nos dados**  
- `winsor_caps_by_ind_year.csv` + `fix_summary_all.csv`.  
- Slide “antes/depois” da **mediana/variância** de `sales` por ano.

**Decisão & porquê**  
- Estabiliza sem “apagar” cauda informativa.

**Riscos & alternativas**  
- Ajustar percentis por ruído setorial.

---

## 4) Inventário de flags (mapa de risco)

**Termos**  
- **Flag:** teste booleano.  
- **Inventário:** `% flagged` por regra (antes/depois).

**Como provar nos dados**  
- `flags_before.csv` vs `flags_after.csv` + `figs/flags_inventory_before_clean.png` e `figs/residual_issues_after_clean.png` (esperado 0% no que é drop-eligible).

**Decisão & porquê**  
- Transparência e priorização por impacto.

**Riscos & alternativas**  
- Se “explode”, investigar **upstream** (parse/origem).

---

## 5) Limpeza focada em sobrevivência (motor)

**Termos**  
- **Zumbi:** ≥3 anos com `sales≈0` **e** `labor_avg=0`.  
- **Gap truncation:** sumiço **>1 ano** → cortar a partir do retorno.  
- **Censura:** sem evento, duração até último ano observado.

**Como provar nos dados**  
- `drop_log.csv` + `figs/drops_by_reason.png`.  
- `zombie_truncation_log.csv` e `panel_gap_truncation_log.csv` (quando existirem).

**Decisão & porquê**  
- **Cortar > remendar**: só trajetória confiável.

**Riscos & alternativas**  
- Ajustar zumbi 3→2 anos (cuidado com falso positivo).

---

## 6) Artefatos de QA e diagnósticos

**Termos**  
- **Skim:** resumo por coluna (contagens/percentis/assimetria).  
- **Multicolinearidade:** |cor|>0,95, VIF>10.  
- **Leakage:** informação do futuro.

**Como provar nos dados**  
- `skim_per_column.csv`, `asymmetry_per_column.csv`, `multicollinearity_high_corr.csv`, `multicollinearity_vif_gt10.csv`, `leakage_risk_table.csv`.  
- Slides: “Skim – top únicos”, “Assimetrias”, “Leakage risk counts”.

**Decisão & porquê**  
- Garante dataset **modelável** e **sem vazamento**.

**Riscos & alternativas**  
- Leakage alto? usar `lag_predictors` (defasados).

---

## 7) Dataset de sobrevivência (uma linha por firma, pronto p/ R)

**Termos**  
- **t0_year:** primeiro ano observado.  
- **event:** 1 se `exit_year≥t0`, senão 0 (censurado).  
- **duration_years:** `t_end_year − t0_year`.  
- **Split** determinístico (estrat. por `event` + indústria).  
- **Imputação simples (baseline):** mediana/moda ind×ano + `miss__*`.

**Como provar nos dados**  
- `artifacts/survival_readiness_report.csv` → `firms, events, event_rate_pct, events_train, events_test, events_ge_min_required`.  
- Arquivos: `<base>__SURVIVAL.csv` (+ `TRAIN/TEST` opcionais).

**Decisão & porquê**  
- **CSV único** simples; baseline imutado e `miss__*` para modelos.

**Riscos & alternativas**  
- Sem eventos suficientes → readiness alerta.  
- **Bug conhecido:** alinhar `print` (2 vs 4 caminhos).

---

## 8) Imputação avançada (MICE+RF) — opcional

**Termos**  
- **MICE:** imputa coluna usando as demais (iterativo).  
- **RF Regressor/Classifier:** árvore p/ números/categorias.  
- **Máscara de avaliação:** mede MAE/RMSE/ACC em valores escondidos.

**Como provar nos dados (se ligado)**  
- `impute_summary.csv`, `impute_eval_scores.csv`, `impute_feature_importances.csv`.

**Decisão & porquê**  
- **Desligado por padrão**; ligar só com avaliação objetiva.

**Riscos & alternativas**  
- Garantir `FE_POLICY/FEConfig/_HAVE_SKLEARN`; senão, manter off.

---

## 9) Duplicatas, multicolinearidade, infinito, leakage

**Como provar**  
- Duplicatas: `figs/dedup_dropped_rows.png` + `drop_log.csv`.  
- Multicolinearidade: `multicollinearity_*`.  
- Infinito: razões aparecem em `drop_log.csv`.  
- Leakage: `figs/leakage_risk_counts.png`.

**Mensagem**  
- Fecha as “pontas soltas” de **qualidade** e **modelabilidade**.

---

## 10) Fluxo executivo (resumo)

```text
NA tokens → schema → flags (ANTES)
→ cortes estritos + consertos + anualização + winsor
→ (opcional) imputação/lag
→ flags (DEPOIS) → QA visuais → SURVIVAL.csv + readiness