{% docs ymp_charge_cd_class_db %}

-- Source file(s): /ymp_charge_cd_class_db.sql  (SQL Server T-SQL, labeled "original R code")
-- Upstream dependencies (dbt sources/refs):
--   - source('ymp','ymp_charge_cd_class')   -- preserves original table name; no hard-coded DB/schema.
-- Expected grain & PK: each row is a valid (effective) charge code record; likely grain = Charge_Cd at a time slice.
--   *Recommend tests*: not_null(Charge_Cd); if unique by Charge_Cd within validity window, add a custom uniqueness test.
-- Purpose: return only non-updated, non-deleted, currently-effective charge code classifications.
-- Materialization: view (to preserve always-fresh validity filtering; change to table only if snapshotting semantics are desired).
-- Determinism: Replaced volatile CURRENT_TIMESTAMP in the filter with a run-anchored timestamp ( run_started_at ), cast to TIMESTAMP_NTZ.
--   -- NO DIRECT TRANSLATION: Your final SELECT requires CURRENT_TIMESTAMP for SYS_* cols, which is volatile across retries.
--      We keep it (per your non-negotiable requirement) and document that these two fields are not deterministic.
-- Timezones: assume source eff_dt/disc_dt are interpreted in Snowflake as TIMESTAMP_NTZ. If source stores TZ-aware values,
--            cast behavior must be revisited. Documented below; explicit casts used.
-- Anti-pattern FIX: SQL Server brackets [col] → Snowflake-compatible identifiers/aliases; no engine-dependent behavior.
-- Optimizations: predicate pushdown on source; only required columns are projected early.
-- Machine learning candidate: not applicable; this is deterministic filtering, no predictive uplift expected.
-- Unresolved dependencies: Requires a sources.yml entry defining source('ymp','ymp_charge_cd_class').

{% enddocs %}