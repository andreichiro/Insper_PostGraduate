{% docs aaq_process_Nik_query_rev_funnel_last_pi %}
-- Original verbs:      SELECT (projection), FROM (single table), WHERE (filters).
-- Upstream deps:        source('ymp','YMP_Revenue_Funnel')   -- preserves physical table name
-- Expected grain:      Likely per (infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr, str_period).
-- Materialization:     view (read-only filtered slice).
-- Determinism:         -- NO DIRECT TRANSLATION: The original used CURRENT_TIMESTAMP in a filter.
--                      We anchor "now" to  run_started_at  (UTC) for reproducibility. This matches
--                      the determinism guidance while preserving behavior within a run.
-- Tests to add:        not_null on output cols; optional uniqueness on the 5-col set if business rules allow;
--                      freshness on source.
-- Assumptions:         (1) eff_dt/disc_dt are comparable timestamp columns; treated as TIMESTAMP_NTZ.
--                      (2) str_period is numeric YYYYMM; BETWEEN is inclusive as in the original.
-- Anti-pattern FIX:    Dropped "WHERE 1=1" (no-op often used for dynamic SQL).
-- Optimizations:       -- OPTIMIZATION: Column pruning + predicate pushdown for minimal IO.
-- NO DIRECT TRANSLATION:
--    * Using Snowflake TIMESTAMP_NTZ to match project norms; pure ANSI has no NTZ/TZ types.
--    * The original commented MAX()/GROUP BY and static period window remain intentionally unused.
-- Consumers:           Unknown.
-- machine_learning_candidate: Not applicable; this is a deterministic filter, no predictive task.
-- Unresolved deps:     Ensure sources.yml maps source('ymp','YMP_Revenue_Funnel') to ymp.dbo.YMP_Revenue_Funnel.

{% enddocs %}
