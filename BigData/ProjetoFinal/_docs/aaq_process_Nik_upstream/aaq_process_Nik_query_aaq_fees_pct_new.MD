{% docs aaq_process_Nik_query_aaq_fees_pct_new %}

-- Source: T‑SQL SSMS snippet joining Container_Group_Master_Detail to Fee_Detail
-- Behavior: inner join on 5 keys; filter to target YYYYMM based on day>15 logic
-- Grain/PK: (infopro_div_nbr_dcg, acct_nbr_dcg, site_nbr_dcg, container_grp_nbr_dcg, snapshot_dt)
-- Output: exact 11 selected columns (names/order preserved)
-- Determinism: as_of_ts := COALESCE(var('as_of_ts'), run_started_at) [assumed UTC]
--   -- NO DIRECT TRANSLATION: local time vs UTC; set var('as_of_ts') if business-local time is required
-- Tests to add: not_null keys; relationships across 5 keys+snapshot_dt; accepted pattern for 8-digit snapshot_dt; uniqueness checks as applicable
-- Anti-pattern FIX: no type coercion on join keys; no implicit date parsing
-- Result ordering: models are unordered; no ORDER BY (by design)

{% enddocs %}
