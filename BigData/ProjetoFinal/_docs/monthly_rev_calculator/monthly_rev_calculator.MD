{% docs monthly_rev_calculator %}

-- Upstream deps:  ref('aaq_tbl')   -- Lines 01–13 (input table parameter)
-- Expected grain: same as input (no aggregation). PK unknown (TODO: document).
-- Outputs: updated revenue fields per business rules; both non-inverse and inverse modes supported.
-- Materialization: table (stable, repeatable; no reliance on ORDER BY outside windows).
-- Assumptions:
--   * post_elig_run defaults to TRUE (as in R). Controlled via var('post_elig_run', true).
--   * inverse defaults to FALSE (as in R). Controlled via var('inverse', false).
--   * All arithmetic uses DECIMAL semantics where feasible; banker’s rounding emulated.
-- Determinism:
--   * No window ordering used; all logic is row-wise. Outputs repeat given same inputs & vars.
--   * SYS_* fields use CURRENT_TIMESTAMP per requirement (not deterministic across runs).
-- Optimizations:
--   * Predicate pushdown by gate flag avoids unnecessary CASE evaluations.
--   * No macros; minimal, readable CTE chain.
-- NO DIRECT TRANSLATION notes:
--   * R's eval(parse()) dynamic filter replaced with deterministic gate flag (see CTE_source).  -- Lines 11
--   * Rounding uses ANSI-friendly emulation; see CTE_round_* (documented below).              -- Lines 40–41, 71–72
-- Anti-pattern FIX:
--   * R inverse branch uses is.na(Estimated_Monthly_Rev) when filling New_Charge_Rate; this likely
--     overrides prior assignments unintentionally. We guard on New_Charge_Rate IS NULL instead.      -- Lines 66–67
-- Tests to add (dbt tests):
--   * not_null on key business columns (e.g., charge identifiers).
--   * accepted_values on enumerations (Charge_Type, Charge_Method).
--   * numeric range tests on rates/frequencies.
--   * conditional tests (e.g., if Charge_Method='Q' and Charge_Cd='CMP' then Estimated_Monthly_Rev=0 in non-inverse).

{% enddocs %}