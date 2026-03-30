{% docs business_rule_function %}

  Source R file:
    - projects/AAQ/R/YMP_rules_function.R  (function: YMP_rules_function)

![Business rule function overview](assets/business_rule_function (level 1, overview).jpg)

  Quick summary:

    Upstream dependencies (dbt):
      - source('ymp_staging','active_account_query')                -- R Lines 32–59
      - source('ymp_staging','ymp_test_rules_join')                 -- R Lines 61–78 (test_rules only)
      - ref('business_rule_function_query_nomenclature')            -- R Lines 86–137 (tokens, hints)
      - ref('business_rule_function_query_business_rules')          -- R Lines 139–152 (non-test rules)
      - ref('business_rule_function_query_testing_business_rules')  -- R Lines 153–168 (test rules)
      - ref('business_rule_function_query_aaq_data_types')          -- R Lines 170–174 (column data types)

    Grain & PK:
      - One row per AAQ account record (Infopro_Div_Nbr, Acct_Nbr).

    Outputs:
      - All AAQ columns (original names, with optional lowercasing where requested),
        plus YMP_Eligible_Rate, YMP_Elig_Bus_Rule,
        and (for 'YMP PI Logic' with rule_impact=true): All_Elig_Changes.

    Materialization: table (stable outputs).

    Determinism:
      - Time filters anchored to {{ run_started_at }} (UTC).
      - Guard that exactly one AAQ model date is present when evaluating testing rules.

    ANTI-PATTERN FIX:
      - The R code uses eval(parse(...)) to run rule code constructed as strings.
        Here we (a) standardize filter text, (b) parse into atoms (IN/NOT IN, CMP, GREP, SUBSTR),
        and (c) evaluate atoms against a per-row state object. No string-eval in SQL.

    NO DIRECT TRANSLATION (documented):
      - The R “nomenclature DSL → R code” multi-pass rewrite is *not* re-implemented character-for-character.
        We assume filter_code / assignment_code already use normalized constructs:
        IN/NOT IN, comparisons (==,!=,<,<=,>,>=), grepl("…", COL), substr(COL,start,end) ==/!= "…".
        If your upstream still stores a higher-level DSL, add a small pre-model to rewrite DSL into those forms
        using the nomenclature table. This mirrors the R steps and preserves intent.

    Tests to add (schema.yml):
      - unique: (Infopro_Div_Nbr, Acct_Nbr)
      - not_null: Infopro_Div_Nbr, Acct_Nbr
      - accepted_values: YMP_Eligible_Rate in (0,1) or null
      - relationships: to source('ymp_staging','active_account_query')

  Context and deep dive

  1) What business_rule_function is

  business_rule_function is the Pricing team’s rule‑engine model that implements the “YMP rules” over the Active Account Query (AAQ). It receives a set of switches/parameters and deterministically evaluates Filter rules (to set eligibility) and Assign rules (to set derived fields, e.g., rounding policy or recommended PI). It reproduces legacy R behavior but is implemented as a dbt SQL model (materialized table) for reproducibility, auditability, and downstream consumption.

  2) Where it sits in the Pricing flow (Pricing-only perspective)

  Upstream (Pricing inputs only):

  dwcore tables

  ODS rate tables

  ODS minimum lift & special handling tables

  Revenue funnel

  ind_pricing2

  Container group master detail
  → combined to form the base table

  Enrichments → staging for rules:

  base table → rate restriction

  base_table_rate_restriction → adds model class & YMP eligibility

  base_table_rate_restriction_model_eligibility → feeds business_rule_function

  base_table_rate_restriction_model_eligibility_business_rule → monthly revenue calculator → final AAQ table

  Zoom on the rule engine (Pricing’s purview):

  Input is AAQ and/or a restriction (SQL predicate), and/or model class, and/or YMP eligibility.

  Active rules (from ymp_rules in PROD or ymp_test_rules_new in TEST) are applied in global order; Filter sets eligibility, Assign writes derived fields.

  Conflicts are resolved by deterministic precedence and last‑write‑wins semantics.

  3) Contract: inputs, parameters, outputs

  Primary input (AAQ): ymp_staging.active_account_query (all columns preserved; optional lower‑casing of selected columns for case‑sensitive parity with R).

  Optional joins (TEST runs): ymp_staging.ymp_test_rules_join to align AAQ rows to test rules within their effective window while excluding _H0 tests.

  Rule catalogs & compiler tokens:

  PROD rules: ymp_staging.ymp_rules

  TEST rules: ymp.ymp_test_rules_new

  Nomenclature tokens: ymp.bus_rule_nom

  AAQ type map (Information Schema snapshot) drives typed comparisons.

  Key switches (dbt var(...)):

  prcs_type (default ['YMP Eligibility']): controls which rulesets run; include 'YMP PI Logic' to enable PI stubs/outputs.

  approved (default [1]): filter by approval flag.

  test_rules (default false): switch to TEST catalog + lineage join; requires a single model date anchor.

  budget_run (default false): exclude Corporate owners and restrict to rule_nbr >= 5100 (budget path parity).

  aaq_filter (default ''): WHERE suffix to subset AAQ reads.

  cols_tolower (default []): list of AAQ columns to lowercase only in the seed state (not for assigned values).

  Model-date controls for TEST alignment
  cur_model_date (explicit), or use_model_date_from_aaq=true with model_date_source_col (default 'CUR_YMP_NRD_YYYYMMDD'), with allow_multi_model_dates_in_test=false ensuring a single anchor.

  Outputs (broadcast to AAQ grain):

  YMP_Eligible_Rate (0/1) — eligibility flag.

  YMP_Elig_Bus_Rule — rule_id that set eligibility.

  When 'YMP PI Logic' is in prcs_type: stubs Adjustment_Reason, Recommended_PI_Percentage, Recommended_PI_Amount are added if missing; assignments may also set Rec_PI_Adj_Type, Recommended_PI_Pct, Recommended_PI_Amt, Rounding_Policy, All_Elig_Rules.

  When rule_impact=true with 'YMP PI Logic': emits All_Elig_Changes trail for impact auditing.

  Audit fields: SYS_EXEC_ID, SYS_CREATE_DTM.

  Grain note (correction): Eligibility is computed at (Infopro_Div_Nbr, Acct_Nbr) and then broadcast back to the full AAQ row grain. If AAQ carries customer/container/site, you’ll see eligibility duplicated across those rows. The underlying decision is not per-container/site.

  4) How rules are evaluated (R parity without eval(parse))

  Pipeline:

  Load AAQ (+ TEST lineage if test_rules=true), apply aaq_filter, add PI “shell” columns when prcs_type includes 'YMP PI Logic'.

  Load rules (PROD or TEST) and filter by prcs_type, approved; for TEST, also restrict by the model date anchor (effective window).

  Load nomenclature tokens and AAQ type map (maps AAQ columns to as.character|as.integer|as.numeric|as.Date).

  Normalize & parse Filter code (Exclude/Include, OR/AND groups) into typed atoms:

  IN / NOT IN

  CMP (==, !=, <, <=, >, >=) with numeric/string coercion based on AAQ column type

  GREP (regex) and SUBSTR with inclusive end index (R semantics)

  Normalize & parse Assign code into (target,value) pairs with type‑aware coercion.

  Build a per‑row state object (variant map of AAQ columns in uppercase) and apply rules in global order:
  process_type ASC → rule_nbr ASC → precedence ASC → sf_insert_ts ASC → rule_id ASC.

  Filters: last matching filter wins, sets YMP_Eligible_Rate (0 for Exclude, 1 for Include) + YMP_Elig_Bus_Rule.
  Assigns: when their filter matches, merge their assignment object into the state (last write wins by key).

  Project the final state back to explicit columns with strict typing; assigned values are not lower‑cased.

  Determinism / hard guards:

  Unsupported filter atoms → hard fail (parity with R erroring on parse/eval).

  TEST runs require a single model date (unless explicitly allowed).

  If no rules exist, we return the seed state (R parity).

  5) Rule conflicts, precedence, and “rules about rules”

  Conflicts are resolved by the global order and last‑write‑wins semantics:

  Among Filters: the last matching rule determines eligibility and the winning rule id.

  Among Assigns: within the same target column, the last applied assignment prevails.

  budget_run=true narrows the catalog (exclude Corporate; rule_nbr >= 5100) to mirror the R “budget” path.

  6) PI (Price Increase) logic and outcomes

  One common outcome is to determine whether the account is eligible for a PI (boolean).

  When you feed PI eligibility (e.g., from a prior step) back into AAQ/state, Assign rules can emit the recommended PI:

  Percentage (Recommended_PI_Percentage / Recommended_PI_Pct)

  Amount (Recommended_PI_Amount / Recommended_PI_Amt)

  Rationale (Adjustment_Reason, Rounding_Policy)

  With rule_impact=true under 'YMP PI Logic', the engine writes an impact trail in All_Elig_Changes, encoding each change with suffixed markers (_p for percent, _a for amount, _n for none).

  7) Pricing cycle & analyst workflow (operational cadence)

  There’s a defined review window when analysts input recommendations.

  At window close, analyst inputs are combined with rule outcomes in business_rule_function to build final recommendations.

  The function then calculates the next review period for each account (date anchoring via AAQ model date is enforced for TEST paths).

  8) Implemented models, lineage, and materializations (dbt)

  Final model (table):

  business_rule_function → AAQ + derived outputs (stable, reproducible table)

  Supporting models:

  business_rule_function_query_business_rules (table; PROD rules “as‑of” snapshot)

  business_rule_function_query_testing_business_rules (view; TEST rules pass‑through; effective‑window filtering happens inside business_rule_function using the model date anchor, not inside this view)

  business_rule_function_query_aaq_data_types (table; Information Schema snapshot)

  business_rule_function_query_nomenclature (table; normalized tokens)

  Sources consumed:

  ymp_staging.active_account_query (AAQ; primary input)

  ymp_staging.ymp_test_rules_join (TEST lineage; time‑bounded; excludes _H0)

  ymp_staging.ymp_rules (PROD rules)

  ymp.ymp_test_rules_new (TEST rules)

  ymp.bus_rule_nom (DSL token catalog)

  Information Schema for AAQ (drives typed comparisons)

  Materializations:

  Final + reference tables are materialized as tables (except the TEST rules view).

  Time filters are anchored to run_started_at for PROD.

  TEST lineage anchors to AAQ model date (single‑date guard), keeping runs reproducible.

  9) Validation, parity choices, and case sensitivity

  Sequential semantics: strict application order; last write wins — R parity.

  Case sensitivity: GREP/SUBSTR are case‑sensitive unless a column is listed in cols_tolower (only the seed gets lower‑cased; assigned values are preserved).

  Type coercion: numeric vs. string comparisons follow AAQ type map. SUBSTR uses inclusive end index (matches R).

  Unsupported tokens: fail fast (better than silent drift).

  No eval(parse): the DSL is compiled into typed atoms and evaluated in SQL — safer and auditable.

  Recommended tests (schema.yml):

  Final table:

  unique on (Infopro_Div_Nbr, Acct_Nbr) (or AAQ grain if stricter)

  not_null on (Infopro_Div_Nbr, Acct_Nbr)

  accepted_values on YMP_Eligible_Rate ∈ {0,1, null}

  relationships to AAQ

  Reference tables: not_null on keys; accepted values on flags where appropriate.

  10) Edge cases & error handling to know about

  Multiple AAQ model dates in TEST → hard fail (unless allow_multi_model_dates_in_test=true).

  No active rules → engine returns input AAQ rows with null eligibility (seed state).

  Rules with blank assignment_code → safely ignored.

  Columns created by assignments are added stubs only when required ('YMP PI Logic'), otherwise they remain null unless assigned.

  AAQ subsetting via aaq_filter pushes down as a WHERE suffix (use with caution; it participates in model date selection).

  11) Performance & operations (practical guidance)

  Predicate pushdown: aaq_filter reduces working set early.

  Information Schema scan is constrained by table_schema + table_name to minimize metadata cost.

  The recursive rule application operates on variant objects per row — powerful but sensitive to rule volume; prune rules via prcs_type, approved, and budget_run where suitable.

  Monitor row counts, nullability, and rule coverage over time; add data tests for critical invariants (e.g., % rows with eligibility).

  Log/alert when CTE_fail_if_unsupported or model-date guards trigger.

  12) Open issues & clarifications (skeptical pass)

  Grain ambiguity in the informal note: it claimed PI eligibility at acct/customer/container/site. The implemented code computes at (div, acct) and broadcasts — confirm downstream consumers expect duplication across container/site rows.

  Historical doc drift: an older doc for the TEST rules view referenced var('cur_model_1/2'). The current implementation anchors effective windows inside business_rule_function using CTE_model_anchor; the TEST rules view is a pass‑through with flags and basic filters.

  Case sensitivity: confirm any columns added to cols_tolower still meet matching needs across GREP/SUBSTR vs. business expectations.

  Owner filtering in budget_run: “Corporate” detection is substring on owner_name; verify this convention is stable across catalogs.

![Business rule function detailed](assets/business_rule_function (level 2, detailed).jpg)

    {% enddocs %}