{% docs int_nrd_realign_aaq %}


-- Translation Report (condensed)
-- Source R file: nrd_realign.R
-- R lineage covered here (Lines ~53–152, then ~112–126 merge of NRD push immediately after):
--   1) Load AAQ (from /query/YMP/nrd_realign_aaq_source.sql in R)
--   2) (In in-memory fallback only) join short-term push flags and filter YMP_Eligible_Rate==1 OR push_exists==1
--   3) Compute NRD_Adj (first-of-month bucketing by day<=15), Exclusion_Reason_Cur_YMP (NA)
--   4) Compute rate restriction flags: Is_NRD_RR and Next_RR_PI (min YYMM per acct) using cur_ymp_pi_eff
--   5) Normalize/adjust account open date, compute Is_New_Acct by config window (months)
--   6) Merge NRD_Push & Push_Origination (from /query/YMP/nrd_realign_nrd_push.sql in R)
-- dbt lineage:
--   upstream sources:    source('YMP','active_account_query')  
--   upstream refs   :    ref('int_nrd_realign_current_ymp')  ,   ref('int_nrd_realign_corp_config')  ,   ref('int_nrd_realign_nrd_push')  
--   downstream refs :    ref('int_nrd_realign_rcf_btd')  ,   ref('fct_nrd_realign')  
-- Expected grain: AAQ row-level (container/site detail) with per-row enrichments; no artificial dedupe performed.
-- Determinism & reproducibility:
--   * All “now” is anchored to   run_started_at   (UTC). Only SYS_* use CURRENT_TIMESTAMP (metadata-only), per spec.
--   * Windows include explicit ordering/ties where used.
-- NO DIRECT TRANSLATION:
--   * The SQL in /query/YMP/nrd_realign_aaq_source.sql is unavailable. We implement the same filter R uses in its fallback:
--     keep rows where YMP_Eligible_Rate = 1 OR a push exists for the account. This is the safest documented parity.
-- Anti-pattern FIX:
--   * Do not retain 'push_exists' in outputs (R explicitly drops it before continuing).
-- Heuristics:
--   * NA↔NULL via TRY_* casts; no SELECT *; explicit column list in the final SELECT.

{% enddocs %}