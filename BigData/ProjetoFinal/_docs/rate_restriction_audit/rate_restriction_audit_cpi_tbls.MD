{% docs rate_restriction_audit_cpi_tbls %}
-- Source R file(s): rate_restriction_audit.R (Lines 121–226 build and cache cpi_tbls.RObj)
-- Upstream deps:  source('dwcore','bls_cpi_monthly')   -- snapshot of BLS API for three series
-- Expected grain: per (seriesid, year, period). Output is “melted” rows per (seriesid, periodname, variable) for the last 12 rows per series.
-- Primary key at output: (seriesid, periodname, variable) is a key within the 12-row snapshot per series.
-- Consumers: models that need CPI lagop/yoyop by series+month (e.g., rate_restriction_audit main model).
-- Materialization: table (stable snapshot semantics).
-- Tests to add (recommended): not_null on seriesid/periodname/variable/value; accepted_values on seriesid;
--   uniqueness on (seriesid, periodname, variable) scoped to the snapshot; relationships from downstream.
-- Assumptions:
--   *  source('dwcore','bls_cpi_monthly')  has: seriesid, year, period ('M01'..'M13'), periodname, value (index level).
--   * Computations replicate R exactly:
--       lagop = (sum(t-11..t) - sum(t-23..t-12)) / sum(t-23..t-12)   (row-based, not month-aware)
--       yoyop = (value(t) - value(t-12)) / value(t-12)
-- Determinism:
--   * No CURRENT_DATE logic; we compute from the source snapshot only. Given the same source, output is stable.
-- NO DIRECT TRANSLATION:
--   * R hits the BLS API on some days and otherwise loads a cached RObj. Here we always read the governed source snapshot.
-- Anti-pattern FIX:
--   * Do not construct calendar dates (M13 is 'Annual'). We order by a deterministic ordinal key (year*100 + period_num).

{% enddocs %}