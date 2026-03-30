{% docs aaq_process_Nik_query_aaq_ind_data %}

aaq_process_Nik_query_aaq_ind_data corresponds to:
--   dplyr verbs: mutate (derived field), filter (non-null predicates), and select (column projection).
-- upstream dependencies: source('pricing_operations','ind_pricing2')
-- expected grain: 1 row per source row; no dedupe/windowing present
-- primary key: not specified by the R/SQL; candidates may include (acct_nbr, site_nbr, container_grp_nbr) but not asserted here
-- outputs/consumers: unknown
-- materialization: table (stable, reproducible)
-- determinism: no ORDER BY (models are unordered); no windows; outputs repeat given same inputs
-- semantics preserved: SQL Server SUBSTRING() when start<1 is emulated exactly (see CASE in cte_final)
-- assumptions: infopro_div_nbr is numeric or castable to INTEGER; values may be 1–n digits; negatives are not expected
-- optimizations:
--   OPTIMIZATION: predicate pushdown—replace "WHERE <derived> IS NOT NULL" with the minimal equivalent
--                 "infopro_div_nbr IS NOT NULL" (i.e., len_param IS NOT NULL) to avoid recomputing the expression.
-- anti-pattern FIX: explicit CASTs to avoid implicit type coercion
-- arrange/order: not applicable; dbt model outputs are unordered by design

{% enddocs %}
