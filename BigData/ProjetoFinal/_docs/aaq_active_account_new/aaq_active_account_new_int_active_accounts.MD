{% docs aaq_active_account_new %}

aaq_active_account_new_0_int_active_accounts
- Source R routine(s): active_accounts() — "out_dt" assembly portion only (Lines ~001–220, 260–320)
- Purpose: Build the shared base dataset (R: out_dt) exactly once. Downstream consumers:
           aaq_active_account_new_1_int_active_accounts (CGMS branch),
           aaq_active_account_new_2_int_active_accounts (Active Account branch).
- Upstream dependencies (sources()): dwcore.dim_container_grp, dwcore.dim_site, dwcore.dim_acct, dwcore.dim_corp_hier
- Expected grain / natural key: (infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr)
- Materialization choice: table — ensures a stable, run-scoped snapshot for deterministic refs.
- Determinism:
  -- Time anchor: use  run_started_at  (UTC) instead of volatile CURRENT_TIMESTAMP to mirror
     “point-in-time” behavior within a dbt run. R used CURRENT_TIMESTAMP at query time;
     anchoring to run_started_at *stabilizes* evaluation for all sources within the model.
  -- Result ordering: not defined for tables; consumers must ORDER BY if needed.
- Anti-pattern FIX:
  -- R repeatedly opens connections and sleeps (while/Sys.sleep). We do not replicate; dbt handles
     connections atomically (compile+run). Behavior on row selection is unchanged.
  -- Avoid duplicate aliasing (R re-selects site_nbr from both sides). We retain the container_grp
     site_nbr and avoid re-aliasing to keep a single, unambiguous column.
- NO DIRECT TRANSLATION:
  -- R’s in-memory object 'out_dt' is not persisted; here we persist as a table to ensure stable,
     re-usable DAG edges and deterministic SYS_* fields at build time.
- Optimizations:
  -- DISTINCT on corp hierarchy (mirrors R’s unique()) to trim fanout pre-join.
  -- Predicate pushdown via WHERE clauses to minimize intermediate row counts.
- Tests to add (in schema.yml): unique combination key on (infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr);
  not_null on critical join columns and date-effectivity fields.

aaq_active_account_new_1_int_active_accounts
aaq_active_account_new_2_int_active_accounts
   - Source R: active_accounts() — CGMS branch (Lines ~001–600 for CGMS build)
   - Upstream sources: dwcore.dim_container_grp, dwcore.dim_site, dwcore.dim_acct, dwcore.dim_corp_hier, dwcore_staging.RPT_CUST_MKTG_INFO
   - Expected grain: (Infopro_Div_Nbr_Dcg, Acct_Nbr_Dcg, Site_Nbr_Dcg, Container_Grp_Nbr_Dcg) after dedupe
   - Primary key policy: deterministic survivor per the window spec (see NO DIRECT TRANSLATION note)
   - Materialization: table
   - Determinism: run-anchored clock via CTE_source.as_of_ts/.as_of_date; avoid CURRENT_TIMESTAMP directly
   - OPTIMIZATION: DISTINCT on dim_corp_hier to control fanout; deterministic dedupe (row_number) vs R’s memory-order unique()
   - NO DIRECT TRANSLATION:
       * iconv transliteration omitted (we clean/truncate only)
       * R unique(by=...) keeps first in current memory order; we select a deterministic survivor
   - Tests to add (dbt schema tests): unique & not_null on the 4-key grain after dedupe

account_new_2
/* machine_learning_candidate
   Rationale: interpretable churn/upsell propensity scoring with transparent features already present:
   (is_broker, is_fee_charged_{frf,evr,inv}, contract_term/status, acct_cat, NAICS, recurring_months_adv_bill).
   Candidate models: shallow decision tree (max_depth=4) or L1-logistic regression.
   Integration: materialize a sidecar scores table keyed by (infopro_div_nbr, acct_nbr); never alter base selections.

{% enddocs %}