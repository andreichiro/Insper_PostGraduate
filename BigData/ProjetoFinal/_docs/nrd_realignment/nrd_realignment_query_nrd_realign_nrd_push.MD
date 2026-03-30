{% docs nrd_realign_nrd_push %}


-- Translation Report (condensed)
-- Source R file: nrd_realign.R
-- R lineage (Lines ~112–126):
--   * Executes /query/YMP/nrd_realign_nrd_push.sql
--   * Substitutes @cur_ymp_nrd with cur_ymp_nrd (YYMM) from current_ymp_model
--   * Filters: active rows (CURRENT_TIMESTAMP between eff_dt and disc_dt),
--              Push_Origination = 'Field Submitted NRD',
--              and nrd > threshold built from @cur_ymp_nrd
--   * Output merged into AAQ: (Infopro_Div_Nbr, Acct_Nbr, NRD_Push, Push_Origination)
-- dbt lineage:
--   upstream source: {{ source('YMP','nrd_push') }}
--   upstream ref   : {{ ref('int_nrd_realign__current_ymp') }}  -- supplies cur_ymp_nrd (YYMM → YYYYMMDD threshold)
--   downstream     : {{ ref('int_nrd_realign_aaq') }} (will JOIN and compute push_exists as an existence flag)
-- Expected grain: ≤ 1 row per (Infopro_Div_Nbr, Acct_Nbr); we aggregate deterministically if multiple rows appear.
-- Determinism:
--   * Replace volatile CURRENT_TIMESTAMP with {{ run_started_at }} (UTC) in filters.
--   * SYS_* metadata use CURRENT_TIMESTAMP (metadata only), per project spec.
-- Comments (requirements):
--   * Line ranges mapped to R; NO DIRECT TRANSLATION notes for case sensitivity.
-- Anti-pattern FIX:
--   * We return the exact R column names and do not add fields R did not produce (no push_exists here).

{% enddocs %}