{% docs ymp_pre_rule_lock %}

-- Source R file(s): -- NO DIRECT TRANSLATION: R script content was not provided.
-- Upstream deps (dbt sources/refs): {{ source('YMP','YMP_rules') }}
-- Expected grain & PK: one row per (rule_id, rule_lock) active "as of" the dbt run anchor.
-- Materialization: TABLE (freeze SYS_* at build time; deterministic per run).
-- Consumers: Any downstream logic needing active rule locks for "YMP PI Logic".
-- Tests to add:
--   - not_null: rule_id
--   - accepted_values or domain tests for rule_lock (if domain known)
-- Assumptions:
--   - eff_dt/disc_dt are timestamp types (or castable) defining an inclusive active window.
--   - is_updated/is_deleted are 0/1 flags; NULL should be treated as not passing (SQL three-valued logic).
--   - SQL Server collation was CI; Snowflake is CS → normalize for exact parity.
-- Optimizations:
--   - Column pruning for predicate pushdown. -- OPTIMIZATION: read only needed columns.
-- NO DIRECT TRANSLATION vs. original T‑SQL:
--   - Case sensitivity normalized via UPPER(TRIM()) to mimic SQL Server CI behavior.
--   - Volatile "now": replaced filter CURRENT_TIMESTAMP with the dbt run anchor {{ run_started_at }} for reproducibility.
--     (SYS_* columns still use CURRENT_TIMESTAMP per your spec; see notes.)
-- Unresolved dependencies:
--   - Ensure a dbt source named 'YMP' with table 'YMP_rules' exists in sources YAML.
-- Determinism:
--   - Window ordering not applicable; final result ordering is undefined, as in dbt models (documented).

{% enddocs %}