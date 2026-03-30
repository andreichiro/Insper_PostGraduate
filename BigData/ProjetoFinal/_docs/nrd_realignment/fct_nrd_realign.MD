{% docs fct_nrd_realign %}

-- Translation Report (condensed)
-- Source R file: nrd_realign.R
-- R lineage covered here (Lines ~337–625):
--   • Start from enriched AAQ (w/ push, NRD_Adj, RR flags)         [337–362]
--   • Derive account-level RCF/BTD                                  [245–274] (joined in)
--   • Call YMP_rules_function twice (approved & test); precedence   [~337–457]
--   • Compute upload_needed, generate upload set                    [405–417]
--   • Revenue alignment by Site→NRD, account rollups, final filter  [458–506]
--   • Persist output (types/format)                                 [507–625]
--
-- dbt lineage:
--   upstream refs : ref('int_nrd_realign_aaq') ,  ref('int_nrd_realign_rcf_btd') ,  ref('business_rule_function') 
--   sources used upstream: YMP.active_account_query, YMP.nrd_push, YMP.ymp_config, YMP.current_ymp_model
--   downstream     :  ref('rpt_nrd_realign_cur_vs_new') 
--
-- NO DIRECT TRANSLATION (transparent):
--   1) The provided dbt model `business_rule_function` (port of YMP_rules_function) currently projects
--      eligibility outputs and assignment overlays (e.g., All_Elig_Rules), but not the realign-specific
--      fields R consumes here (Realigned_NRD, Rule_Id/Rule_Nbr, *_Test, override_phase).
--      → We preserve the deterministic restricted-vs-default precedence implemented earlier:
--           If Is_NRD_RR=1 & Next_RR_PI_yymm present → Realigned_NRD := (PI month - 2)
--           Else Realigned_NRD := NRD_Adj
--         Rule_Id / Rule_Nbr are assigned accordingly.
--      → We LEFT JOIN `business_rule_function` to enrich All_Elig_Rules (and keep it portable: if/when
--        the engine starts exposing Realign fields, a single CTE swap will lift them in; no DAG changes).
--   2) External legs used in R but not provided as SQL (FSA pending txn, Last_PI v3, Future Rate diff)
--      are not wired because their upstream SQL wasn’t supplied. Comments indicate the exact hook points.
--
-- Determinism & Reproducibility:
--   • All “today/now” logic uses  run_started_at  (UTC). Only SYS_* metadata uses CURRENT_TIMESTAMP.
--   • All ROW_NUMBER() include stable tiebreakers. Equality on dates uses NULL-safe <=> when required.

{% enddocs %}