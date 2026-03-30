{% docs ymp_field_budget_new %}

-- Source R/SQL lines: ymp_field_budget_new.R
-- Upstream deps (dbt sources/refs): {{ source('FIELD','OFC_Budget_CBN') }}
-- Expected grain & PK: one row per source row after filters; primary key unknown (none enforced here).
-- Materialization: VIEW (lightweight filter/rename; stable outputs given same inputs and vars).
-- Consumers: Downstream reporting/models using budget “Net PI” sales stats for OM/MFG segments (not provided here).
-- Tests to add:
--   - not_null: Div_Nbr, Fiscal_Year, Acct_Period, Units_Amount
--   - accepted_values: LOB_Cat in ('RES','COM','IND')
-- Assumptions:
--   - Source ran under SQL Server CI collations; Snowflake is CS. Equality normalized via UPPER(TRIM(...)) to match SQL Server’s case & trailing‑space insensitivity.
--   - Pattern matches are intended case‑insensitive → use ILIKE.
--   - FISCAL_YEAR is numeric (or castable) and compared inclusively between provided bounds.
--   - dbt vars `prev_year` and `next_year` are supplied and represent inclusive bounds.
-- Optimizations:
--   - Column pruning & predicate pushdown. -- OPTIMIZATION: read only needed columns; push filters to source scan.
-- NO DIRECT TRANSLATION vs. original T‑SQL:
--   - Case sensitivity/trailing spaces: normalized via UPPER(TRIM(...)) for equality, ILIKE for LIKE.
--   - SQL Server parameters @prev_year/@next_year → dbt vars {{ var('prev_year') }} / {{ var('next_year') }} (typed to int).
--   - Per your spec, SYS_* metadata columns use CURRENT_TIMESTAMP (run‑variant by design).
-- Unresolved dependencies:
--   - Ensure a dbt source named 'FIELD' with table 'OFC_Budget_CBN' exists in sources YAML.
--   - Ensure pipeline provides dbt vars: prev_year, next_year.
-- Determinism:
--   - No windows/ordering; final row order undefined (as in dbt models). Results repeatable for identical inputs/vars; SYS_* columns vary per run by design.


{% enddocs %}