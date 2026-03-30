{% docs rate_restriction_audit_ymp_rules_overlay %}
-- Purpose: Mirror the effect of YMP/business_rule_function.R (R ~ lines 520–648) by emitting curated field overrides.
-- Behavior: Provide adjustments per (infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr) with *_adj columns and a reason.
-- Upstream deps (keys only):  ref('rate_restriction_audit_query_rr_audit_aaq_ifp_source')  supplies the keyspace for validation.
-- Inputs: Curated rule assignments (external to the R code). This model defines the CONTRACT for those adjustments.
-- Determinism: Purely data-driven (no 'now'); stable for identical inputs. Final SYS_* columns use CURRENT_TIMESTAMP per standard.
-- NO DIRECT TRANSLATION:
--   The R function consumes curated rule inputs not embedded in code. Here we expose a table contract for those inputs.
--   Load assignments via a governed table or dbt seed and replace CTE_rules with a SELECT from that source.
-- Tests (recommended): unique on the 4-key, not_null on keys and adjustment_reason, accepted_values for boolean fields.

{% enddocs %}