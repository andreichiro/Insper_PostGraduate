{% docs rate_restriction_audit_lineage %}

Files (models / queries)

rate_restriction_audit.sql                                    -- main model (merges Capture + InfoPro route-note logic)
rate_restriction_audit_query_rr_audit_aaq_ifp_source.sql      -- AAQ/IFP base extract (div/acct/site/cg + route notes, flags)
rate_restriction_audit_ymp_rules_overlay.sql                  -- business-rule overlay (approved “adjustments” to AAQ fields)
rate_restriction_audit_query_rr_audit_capture_restrictions.sql-- Capture site restrictions (year1…year4 → date/%)
rate_restriction_audit_query_rr_audit_cpq_overrides.sql       -- CPQ overrides (MSA & site/account-wide)
rate_restriction_audit_cpi_tbls.sql                           -- CPI snapshot mirror (computes lagop / yoyop)
rate_restriction_audit_query_rr_audit_bi_future_rates.sql     -- FR: future-rate restrictions (ancillary, not feeding MAIN)
rate_restriction_audit_query_rr_audit_bi_future_rates2.sql    -- FR: pending/future rates 2 (ancillary, not feeding MAIN)
rate_restriction_audit_query_rr_audit_aaq_restricted_col_adds_acct.sql  -- ops step to stamp acct flags (ancillary)
rate_restriction_audit_query_rr_audit_aaq_restricted_col_adds_site.sql  -- ops step to stamp site flags (ancillary)
rate_restriction_audit_query_current_ymp_model.sql            -- current YMP model date (for ops/vars only)


Sources

ymp_staging.active_account_query — AAQ base (containers/sites with notes, CSA, broker flag, etc.)

ymp.rate_restrictions_cpq_overrides — approved CPQ overrides (MSA, anniversaries Y1–Y4)

cpq.rpt_quote_detail — Capture quotes (effective date & anniversary percentages)

dwcore.bls_cpi_monthly → fed through rate_restriction_audit_cpi_tbls (compute lagop/yoyop, expose yyyymm)

dwcore.dim_site, dwcore.dim_corp_hier — dims used inside the capture/override queries

ods.bipccr, ods.bipac, ods.bipsd, ods_staging.cdpbirate — FR inputs (ancillary)

ymp.ymp_form — current YMP model (for ops/vars only)

Final dbt lineage (DAG) — Correct & reproducible

Sources: 
  ├─ ymp_staging.active_account_query (AAQ) ───► rate_restriction_audit_query_rr_audit_aaq_ifp_source (A_IFP)
  │                                             ├──► rate_restriction_audit_ymp_rules_overlay (A_OVL)
  │                                             └──► rate_restriction_audit (MAIN)
  ├─ cpq.rpt_quote_detail ───────────────────────► rate_restriction_audit_query_rr_audit_capture_restrictions (A_CAP)
  │                                                └──► rate_restriction_audit (MAIN)
  ├─ ymp.rate_restrictions_cpq_overrides ────────► rate_restriction_audit_query_rr_audit_cpq_overrides (A_OVR)
  │                                                └──► rate_restriction_audit (MAIN)
  └─ dwcore.bls_cpi_monthly ─► rate_restriction_audit_cpi_tbls (R_CPI)
                               └──► rate_restriction_audit (MAIN)

Ancillaries / ops (do not feed MAIN):
  ├─ rr_audit_bi_future_rates(.sql|2.sql)  ──► FR result tables
  ├─ rr_audit_aaq_restricted_col_adds_acct.sql ─► AAQ acct flag stamp
  └─ rr_audit_aaq_restricted_col_adds_site.sql ─► AAQ site flag stamp


Helper refs consumed by the main model

rate_restriction_audit_query_rr_audit_aaq_ifp_source (A_IFP)

rate_restriction_audit_ymp_rules_overlay (A_OVL)

rate_restriction_audit_query_rr_audit_capture_restrictions (A_CAP)

rate_restriction_audit_query_rr_audit_cpq_overrides (A_OVR)

rate_restriction_audit_cpi_tbls (R_CPI)

What the main model does (abridged)

AAQ/IFP base & casting — Standardizes types (div/acct/site/cg, CSA dates, broker flag) and creates cpi_eligible_idx default for CPI references.

Business-rule overlay — Applies governed adjustments from rate_restriction_audit_ymp_rules_overlay (mirrors R’s YMP_rules_function) and stamps ymp_bus_rule.

Active site scaffolding — Separates a “clean” site grain (drop T/X containers for eligibility-only views) and keeps a distinct list of active sites (to fan out account-wide CPQ overrides).

CPI snapshot — Pulls CPI lagop/YOY measures by (cpi_type, variable, yyyymm, periodname). Fallback month (when route note says CPI-*-*-<no month>) uses the latest yyyymm per (cpi_type,variable) (see § Fixes below).

Route-note parsing — Normalizes route notes & special handling text to (date,pct) tokens; aligns CPI tokens; repairs common formats (7/8/5‑digit dates, swapped fields).

Container-group resolution — For each site/container group/month: resolve CPI (join to snapshot) or numeric %; compute perpetual (YYYYXX) behavior and produce 4‑year expansion; de‑dupe to min % per (site,cg,date,perp); add a synthetic 99.99 one year after the last non‑perpetual; enforce bounds (>=1601, <= +5y).

Site roll‑up — Reduce to site/year: min site %; carry CPI series when the min is CPI; for terminal year inject 99.99 unless perpetual.

Capture + overrides — Expand CPQ overrides to sites (account‑wide when site_nbr = -1), derive Y1–Y5 (date,% ) pairs per site; mark expired capture sequences; filter out temp‑only sites unless MSA.

Merge & precedence — Full join Capture vs IFP by site/year; precedence is Perpetual Broker → InfoPro (CPI) → Capture (if both, Capture wins).

Percent normalization — Normalize/scale odd‐looking values (e.g., 0.03 → 3, 300 → 3) with guardrails (cap at 5 when excessive magnitudes show up).

Audits & flags — Flag non‑eligible Capture, ambiguous low % (<1), unusually high (>15 and ≠99.99), perpetual non‑broker; materialize severity.

Fills & placeholders —

Fill missing intermediate future years (when first restriction starts too far ahead): carry month from first restriction and % (0 if Capture‐precedent).

Add unrestricted site placeholders within restricted accounts (99.99 at account’s max month for that year).

Counts — Attach count of unique container‑group restrictions per site/year.

Final shape — One row per (Infopro_Div_Nbr, Acct_Nbr, Site_Nbr, restriction_year) with Capture & IFP details, final (date,% ), audits, counts, CPI series, perpetuity, contract fields, and metadata.

Grain / primary keys per node

rate_restriction_audit_query_rr_audit_aaq_ifp_source: container grain (div, acct, site, cont_grp); later reduced to site grain for eligibility.

rate_restriction_audit_ymp_rules_overlay: same as A_IFP join keys; overlay only.

rate_restriction_audit_cpi_tbls: (cpi_type, variable, yyyymm) with periodname ∈ {jan…dec}.

rate_restriction_audit_query_rr_audit_capture_restrictions: site‑year after expansion (div, acct, site, rr_date_yymm_cap).

rate_restriction_audit_query_rr_audit_cpq_overrides: site level (fan‑out from account‑wide when site_nbr=-1).

rate_restriction_audit (MAIN): site‑year: (Infopro_Div_Nbr, Acct_Nbr, Site_Nbr, restriction_year) unique.

Differences vs R (intentional & parity‑preserving)

Deterministic CPI fallback month (per (cpi_type,variable) and yyyymm), not “last row of a long table”.

Tokenization uses explicit UNION ALL + split_to_table to make alignment reproducible and debuggable.

Perpetual expansion uses generator rows (1..4 years).

Missing‑year and unrestricted placeholders implemented with explicit generators and anti‑joins to avoid accidental dupes.

Explicit % normalization & rounding is retained, with the same thresholds and special 99.99 handling.

Risks / notes

Keys for div/acct/site/cont_grp are treated as strings downstream (to match InfoPro formats). Ensure numeric keys are cast consistently before joining.

99.99 is a sentinel (“restriction expires”): take care when doing min/avg.

CPI fallback month must be derived from yyyymm, not periodname alone (see fixes below).

{% enddocs %}