{% docs ymp_pre_config %}

-- Source R/SQL lines: 01-03 (single SQL block executed by the R job)
-- Upstream dependency: {{ source('dwcore','ymp_config') }}
-- Outputs/consumers: Acts like an in-memory R object; with `ephemeral` no DB object is created.
-- Expected grain: rows from ymp_config that are active at evaluation time and pass filters
-- Primary key: unknown; do not dedupe (preserve parity)
-- Materialization: ephemeral (mirrors R "resultset only"; avoids creating DB objects the R job didn’t create)
-- Determinism: replace volatile CURRENT_TIMESTAMP in the filter with run anchor {{ run_started_at }} (UTC)
--   NOTE: Required observability columns use CURRENT_TIMESTAMP (volatile) by spec; documented below.
-- Tests to add (in schema.yml, optional): not_null(attribute), not_null(value); uniqueness only if guaranteed upstream
-- Assumptions:
--   - eff_dt/disc_dt are timestamps that define an inclusive validity window
--   - process_type and hierarchy_type comparisons are case-sensitive as written
-- NO DIRECT TRANSLATION:
--   - Original filter uses CURRENT_TIMESTAMP; we use {{ run_started_at }} to ensure stable, repeatable builds.
-- anti-pattern FIX:
--   - Eliminated volatile "now()" in the time filter; avoids run-to-run drift for the same build.
-- OPTIMIZATION:
--   - Predicate pushdown & column pruning only to necessary fields.
-- DAG Protocol:
--   - Ensure sources.yml declares dwcore.ymp_config; no unresolved refs in this model.

{% enddocs %}