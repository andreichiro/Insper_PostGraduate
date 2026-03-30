{% docs rate_restriction_audit_query_rr_audit_cpq_overrides %}
-- Source R/T-SQL: rate_restriction_audit_query_rr_audit_cpq_overrides (CPQ overrides filter)
-- Upstream deps:  source('ymp','rate_restrictions_cpq_overrides') 
-- Expected grain: one row per override record effective "as of" anchor (no enforced PK)
-- Business logic: keep non-deleted/non-updated rows whose SCD window includes "now";
--   infer contract length from Y1–Y4 presence; compute exp_dt = add years - 1 day;
--   keep rows strictly after "now".
-- Determinism: ANTI-PATTERN FIX — replace volatile CURRENT_TIMESTAMP in business
--   predicates with a run-anchored timestamp ( run_started_at ) to ensure
--   repeatable outputs; see NO DIRECT TRANSLATION note below for exact T-SQL
--   subtraction semantics.
-- Outputs/consumers: downstream audits of rate restriction overrides (active set)
-- Tests to add: not_null on CSA_Contract_Nbr, Contract_Effective_Dt; accepted_values on
--   is_deleted/is_updated; freshness checks where applicable.
-- NO DIRECT TRANSLATION: DATEADD(year, ...)-1 day in T-SQL implemented as
--   DATEADD(day, -1, DATEADD(year, ...)) for explicitness/portability.
-- Assumptions: business logic is timezone-agnostic; we compare NTZ timestamps.

{% enddocs %}