{% docs int_nrd_realign_corp_config %}

-- Translation Report (condensed)
-- Source R file(s): nrd_realign.R
-- R lineage (Lines ~15–35): Reads /query/YMP/nrd_realign_config.sql against YMP; pivots name/value into one-row config.
-- Upstream sources: {{ source('YMP','ymp_config') }}  -- attribute/value store filtered to corporate scope and current rows
-- Downstream refs: used by int_nrd_realign_aaq (Is_New_Acct months threshold), int_nrd_realign_rcf_btd (rcf threshold),
--                  last-PI/FSA parameterizations (when those ephemerals are added).
-- Expected grain: single row (corporate config at run time)
-- Materialization: ephemeral (consumed by other models)
-- Determinism: all business-time filters anchored to {{ run_started_at }} (UTC). SYS_* metadata use CURRENT_TIMESTAMP by spec.
-- NO DIRECT TRANSLATION: R runs a SQL file; we read the canonical YMP config table with equivalent semantics (attribute/value).
-- Tests to add (in schema.yml):
--   - tests: not_null for each pivoted column; expect exactly one row.
-- Optimizations: predicate pushdown on effectivity; selective projection; CASE-pivot only for attributes used by R.
-- Heuristics: NA↔NULL preserved via TRY_* casts; no SELECT *; no ORDER BY at top level.

{% enddocs %}