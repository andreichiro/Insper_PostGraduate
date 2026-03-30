{% docs aaq_process_Nik_query_bipccm_future_rates %}
-- Upstream deps:          source('ods','vw_bipccm') 
-- Expected grain:        one row per (CMCOMP, CMACCT, CMSITE, CMCTGR, CMCHCD, CMCHTY, CMCGMT, CMWMAT, CMUOM)
-- Primary key candidate: the composite above (validate uniqueness)
-- Output/consumers:      not specified (likely intermediate consumption by pricing/charges marts)
-- Materialization:       table (from config)
-- Tests to add:          - unique + not_null on composite key
--                        - not_null on future_eff_dt
-- Assumptions:           - CMEFDT and CURRENT_TIMESTAMP are comparable in warehouse (timezone semantics as per platform)
--                        - R logic is intentionally volatile; we preserve CURRENT_TIMESTAMP to match the R job
-- Optimizations:         -- OPTIMIZATION: predicate pushdown in CTE_filtered; only required columns retained early
-- NO DIRECT TRANSLATION: R used "!="; ANSI uses "<>" for not-equal
-- Unresolved deps:       none

{% enddocs %}
