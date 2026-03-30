
{% docs nrd_realign_old_process %}

-- Source file(s): "nrd_realign_old_process" (T‑SQL excerpt, treated as the script name)
-- Upstream dependency: source('Pricing_Operations','NRD_alignment_Dailyload')
-- Expected grain & PK (assumption): one row per (infopro_div_nbr, acct_nbr) active at the run time; not guaranteed by source
-- Materialization: view (lightweight; recompiled each run to refresh the time anchor)
-- Determinism: business filter anchored to {{ run_started_at }} (anti-pattern FIX vs CURRENT_TIMESTAMP)
-- Time semantics: compared as TIMESTAMP_NTZ; BETWEEN is inclusive (parity with original)
-- Outputs/consumers: not specified in the snippet
-- Tests to add (suggested):
--   - not_null: infopro_div_nbr, acct_nbr, nrd_shouldbe
--   - accepted_values (on source): is_deleted, is_updated, is_current ∈ {0,1}
--   - (if SCD): conditional uniqueness on (infopro_div_nbr, acct_nbr, eff_dt, disc_dt)
-- Optimizations:
--   - OPTIMIZATION: select only necessary columns post-filter to reduce data movement
-- NO DIRECT TRANSLATION:
--   - SQL Server brackets + 3‑part name -> dbt source(); ensure sources.yml maps to [Pricing_Operations].[dbo].[NRD_alignment_Dailyload]
-- Anti-pattern FIX:
--   - Replaced volatile CURRENT_TIMESTAMP with {{ run_started_at }} in business logic; SYS_* audit fields keep CURRENT_TIMESTAMP per requirement
-- Lineage note:
--   - Upstream: NRD_alignment_Dailyload (source). Downstream: not provided; no extra objects introduced.

{% enddocs %}