{% docs nrd_realign_config %}

-- Source R file(s):            nrd_realign_config.R  -- inferred from context
-- Upstream dependencies:       {{ source('dwcore','ymp_config') }}
-- Purpose (business):          Return key/value configuration pairs currently effective
--                              for process types 'Global' and 'NRD Realign'.
-- Expected grain (rows):       Not guaranteed unique by attribute; duplicates are possible if
--                              multiple process types define the same attribute/value.
-- Primary key:                 None guaranteed.
-- Materialization choice:      view (the R script appears to query-at-runtime; view mirrors that behavior).
-- Tests to add (recommended):  - not_null: attribute
--                              - not_null: value
-- Determinism note:            To preserve *exact* R behavior, we use CURRENT_TIMESTAMP in the filter.
--                              This is intentionally volatile across runs, matching the R snippet.
--                              If reproducible replays are later required, see the optional note below.
-- Optimizations:               Predicate pushdown; only required columns are projected.
-- NO DIRECT TRANSLATION:       None required for logic; exact parity maintained.
-- Unresolved dependencies:     Assumes source('dwcore','ymp_config') is defined in sources.yml.

{% enddocs %}