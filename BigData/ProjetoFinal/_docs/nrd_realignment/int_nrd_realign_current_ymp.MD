{% docs current_ymp_model %}

-- Source R file(s):      current_ymp_model (embedded SQL; see "Lines 01-04" mapping below)
-- Upstream dependencies: source('YMP','ymp_form')  -- mirrors [YMP].[dbo].[ymp_form]
-- Purpose:               Compute the previous NRD month (YYMM) from the latest successful YMP model run.
-- Expected grain:        Single row (0 rows if no successful load exists).
-- Primary key:           N/A (single-row output).
-- Materialization:       table  -- freezes SYS_* audit fields at run time; a view would be volatile.
-- Outputs/consumers:     Downstream models needing the prior NRD month as numeric YYMM.
-- Tests to add:          - row_count <= 1
--                        - if row_count = 1 then exclude_nrd_month BETWEEN 100 AND 9912 (coarse YYMM bound)
--                        - (optional) freshness on source('YMP','ymp_form')
-- Assumptions:           - Column YMP_MODEL ends with 4 chars encoding YYMM (e.g., '2401').
--                        - Upload_Status should be compared case-insensitively (SQL Server default behavior).
-- Determinism notes:     - Use lexicographic ORDER BY on the 4-char alias to mirror SQL Server exactly.
--                        - Defer numeric casts until after TOP-1 selection to match original error surface area.
--                        - Required audit fields use CURRENT_TIMESTAMP (run-time variance) as per spec.
-- NO DIRECT TRANSLATION: - Case-insensitive equality via UPPER(...) in Snowflake.
--                        - Audit columns (SYS_EXEC_ID, SYS_CREATE_DTM) are added per project requirement.
-- Anti-pattern FIX:      - Avoid early/implicit casts; do not quote identifiers to dodge case-sensitivity pitfalls.
-- OPTIMIZATION:          - Predicate pushdown on Upload_Status and minimal column projection reduce scan cost.
-- Machine learning:      - Not a meaningful candidate; purely deterministic transformation.
-- Unresolved deps:       - Ensure sources.yml defines source('YMP','ymp_form') -> [YMP].[dbo].[ymp_form].

{% enddocs %}