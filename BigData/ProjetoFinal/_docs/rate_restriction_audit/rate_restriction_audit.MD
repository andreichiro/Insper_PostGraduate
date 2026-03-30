{% docs rate_restriction_audit %}

-- Materialization: table
--
-- R Sections & Line Ranges (approx):
--   001–120 : setup/libs/config
--   121–226 : CPI pull/cache (blscrapeR or load RObj mirror)
--   227–430 : AAQ/IFP base acquisition & cleaning (eligible sites)
--   431–720 : Capture approvals + CPQ overrides (precedence)
--   721–1460: IFP parsing (route notes) + CPI join + repairs + CG/site roll-up
--   1461–1665: unique CG counts; 1711–2010: merge, audits, rounding
--   2011–2400: future gaps, unrestricted placeholders, outputs
--
-- Expected grain & key:
--   Site-Year (Infopro_Div_Nbr, Acct_Nbr, Site_Nbr, restriction_year).
--   Multi rows per site-year are allowed (e.g., placeholders); we do not dedup across years by design (parity with R).
--
-- Determinism & reproducibility:
--   * All “now” logic is anchored to  run_started_at  (UTC). R used Sys.Date()/Sys.time().
--   * Every window uses explicit PARTITION/ORDER with stable tiebreakers.
--   * R’s banker's rounding emulated with ROUND(..., 2, 'HALF_TO_EVEN').
--   * Tables are unordered; only window ORDER BY defines order; final SELECT has no ORDER BY (not needed for a table).
--
-- Dependencies (dbt):
--   -  ref('rate_restriction_audit_query_rr_audit_aaq_ifp_source')          -- R: ./query/rr_audit_aaq_ifp_source.sql
--   -  ref('rate_restriction_audit_query_rr_audit_capture_restrictions')    -- R: ./query/rr_audit_capture_restrictions.sql
--   -  ref('rate_restriction_audit_query_rr_audit_cpq_overrides')           -- R: ./query/rr_audit_cpq_overrides.sql
--   -  ref('rate_restriction_audit_query_rr_audit_cpq_overrides')                      -- NO DIRECT TRANSLATION: output of YMP/business_rule_function.R
--   -  ref('business_rule_function')                               -- NO DIRECT TRANSLATION: governed mirror of cpi_tbls.RObj
--
-- NO DIRECT TRANSLATION (documented & production-safe):
--   * CPI snapshot: the R job fetched BLS (blscrapeR) or loaded //.../cpi_tbls.RObj. Here we depend on a governed,
--     audited snapshot model  ref('rate_restriction_audit_cpi_tbls')  with columns:
--       seriesid, cpi_type in ('cpi_u','cpi_wst','cpi_gt'), periodname (lower 'jan'...'dec'),
--       variable in ('lagop','yoyop'), value (fraction like 0.031).
--     This mirrors the R object semantics ("last 12 months per series" already prepared upstream).
--   * Business rules overlay: R sources YMP/business_rule_function.R to produce adjusted notes/fields when
--     Adjustment_Reason is present. We apply the overlay from  ref('rate_restriction_audit_ymp_rules_overlay') :
--     it must expose adjusted columns for (Infopro_Div_Nbr, Acct_Nbr, Site_Nbr, Container_Grp_Nbr).
--
-- Anti-pattern FIX:
--   * One-time casts/renames in a single base CTE (avoid repetitive CASTs downstream).
--   * Deterministic CPI month fallback: when CPI tag lacks explicit month, use the latest month present in the snapshot,
--     selected deterministically via an explicit month index (see CTE_cpi_month_latest).
--   * Implement all critical string repairs the R applies (7/8/5 digit fixes; reversed fields; month validity).
--   * CPI 'YYYYXX' perpetual handling: normalize date and replicate +1..+4 years (R 980–1065).
--
-- Machine-learning candidate (tagged):
--   Ambiguity detection in route-note tokens (e.g., sub‑1% PI) can be pre-flagged with a shallow decision tree or
--   regularized logistic regression on tokenized features (ngrams + numeric patterns). Interpretable, low-risk,
--   and improves reviewer focus (precision on "Ambiguous PI percent" and similar heuristics).

Use case: Automatically classify free‑text route notes & special handling into one of: {CPI‑R12, CPI‑YOY, numeric percent, date+99.99 fallback, unusable} and extract the numeric percent when present.

Interpretable models: Single decision tree (max depth ~4–5), or logistic regression with engineered regex features (presence of tokens, token counts, pct pattern flags, date length flags). These are textbook‑style interpretable and can reduce false positives in “CPI vs numeric pct” detection, and lower manual audits.

Why substantial: This part drives downstream restriction classification. Small classification gains (2–5 pp) materially reduce error cases and audit load.
(I inserted a short rationale in the header comment of the main model during Patch 3.)


{% enddocs %}