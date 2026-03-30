{% docs nrd_realign_aaq_source %}

-- Source R/T‑SQL snippet name: nrd_realign_aaq_source
-- Upstream dependencies (dbt sources):
--   - {{ source('ymp','active_account_query') }}
--   - {{ source('ymp','nrd_push') }}
-- Expected grain & primary key:
--   - Follows the grain of {{ source('ymp','active_account_query') }}; 1+ rows per account
--     are possible if multiple qualifying pushes exist (preserved behavior).
-- Materialization: view (safe default). Change to 'table' if you persist the result.
-- Determinism:
--   - Business logic uses a run-anchored timestamp ({{ run_started_at }}) for reproducibility.
--     This is a deliberate, documented deviation from literal CURRENT_TIMESTAMP in the R/T‑SQL.
--   - Required audit fields intentionally use CURRENT_TIMESTAMP (per spec).
-- Lineage status:
--   - Inputs reproduced exactly (active_account_query, nrd_push).
--   - Output is this model only; no additional objects are created (besides the two mandated
--     audit columns). Downstream consumers are not specified in the snippet.
-- Key parity notes:
--   - Case sensitivity: SQL Server default collations are CI (case-insensitive). Snowflake is
--     case-sensitive by default. We use ILIKE for case-insensitive parity on push_origination.
-- Tests to add (suggested):
--   - not_null: infopro_div_nbr, acct_nbr
--   - accepted_values: ymp_eligible_rate IN (0,1)
--   - relationship: if you constrain duplicates later, add uniqueness tests
-- NO DIRECT TRANSLATION:
--   - The R code embeds T‑SQL. We implement the same logic in Snowflake SQL.
--   - Final column list must be explicit; we enumerate source columns at compile time to avoid
--     SELECT * while not guessing schema (uses adapter metadata).
-- Risks:
--   - LEFT JOIN may multiply rows if multiple qualifying pushes exist (preserved).

{% enddocs %}