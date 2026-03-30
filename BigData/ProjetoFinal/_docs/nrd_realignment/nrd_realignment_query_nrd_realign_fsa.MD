{% docs nrd_realign_fsa %}

-- Source R file(s): nrd_realign_fsa.R  -- Lines 01-19 (see inline references)
-- Purpose: Flag (infopro_div_nbr, acct_nbr) pairs having CURRENT FSA txns:
--          SUM(monthly_sales_change_amt + revenue_expect_disposal_change_amt) > 0
--          for included txn codes and NOT excluded reasons since a start period.
-- Upstream sources: dwcore.fact_sales_activity, dwcore.dim_corp_hier,
--                   dwcore.dim_container_grp, dwcore.dim_txn_reason_cd
-- Expected grain (output): 1 row per (Infopro_Div_Nbr, Acct_Nbr) meeting HAVING predicate
-- Primary key (business): (Infopro_Div_Nbr, Acct_Nbr)  -- recommend uniqueness test
-- Materialization: view (lightweight; mirrors upstream changes)
-- Determinism: Stable given same inputs & vars. SYS_* are intentionally volatile metadata per spec.
-- Suggested tests:
--   - not_null: Infopro_Div_Nbr, Acct_Nbr
--   - unique: (Infopro_Div_Nbr, Acct_Nbr)
--   - relationships: fsd.corp_hier_sk -> dch.corp_hier_sk; fsd.container_grp_sk -> dcg.container_grp_sk; fsd.txn_reason_cd_sk -> dtx.txn_reason_cd_sk
-- Assumptions:
--   - sales_activity_period_sk is numeric; var('fsa_start_date') comparable (period key).
--   - NO DIRECT TRANSLATION: SQL Server is typically case-insensitive; Snowflake is case-sensitive.
--     To preserve typical T-SQL semantics, we normalize both sides with UPPER(). If your source
--     system was case-sensitive, remove UPPER() calls to match that environment.
-- Optimizations:
--   - Predicate pushdown on fact (period + flags) occurs in CTE_source. -- OPTIMIZATION: minimizes join/agg cost.
-- NO DIRECT TRANSLATION:
--   - T-SQL parameters (@fsa_start_date, @config_incl_fsa_txn_cd, @config_excl_fsa_rsn_cd)
--     are dbt vars: var('fsa_start_date', 0), var('config_incl_fsa_txn_cd', []), var('config_excl_fsa_rsn_cd', []).
--   - Required SYS_* columns use CURRENT_TIMESTAMP exactly as mandated by spec (volatile by design).
-- Unresolved dependencies: none (all read via source())
-- Lineage: Inputs (4 dwcore tables) → this model (nrd_realign_fsa) → downstream consumers (not provided here).

{% enddocs %}