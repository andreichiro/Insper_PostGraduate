{% docs aaq_process_Nik_query_bipcc_future_rates %}

-- Source R/SQL (Lines 01-24): inline snippet provided (T-SQL style).
-- Upstream dependency:  source('ods','vw_BIPCC')   -- preserves original object name
-- Expected grain (output): one row per (cccomp, ccacct, ccsite, ccctgr, ccchcd, ccchty, cccgmt)
-- Primary key candidate: the 7-key combination above.
-- Materialization: table (stable aggregate; no incremental logic needed).
-- Determinism: 
--   * NO DIRECT TRANSLATION: replaced volatile CURRENT_TIMESTAMP in the filter with a
--     run-anchored timestamp  var('as_of_ts', run_started_at)  to ensure repeatable
--     outputs across retries/backfills; functionally equivalent within a run.
--   * Per requirement, SYS_EXEC_ID and SYS_CREATE_DTM use CURRENT_TIMESTAMP exactly.
-- Assumptions:
--   * CCEFDT is comparable to a timestamp and is timezone-naïve; we default to TIMESTAMP_NTZ.
--   * '' should remain empty string (not NULL) in the target warehouse.
-- Tests to add:
--   * not_null + unique on the 7-key combination.
--   * not_null on future_eff_dt.
--   * accepted_values on flags if modeled downstream (e.g., SYS_DEL_IND in {'N'}).
-- Unresolved dependencies: none.

{% enddocs %}
