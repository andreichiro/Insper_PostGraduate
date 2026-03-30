
{% docs aaq_process_Nik_query_bipccr_future_rates %}

-- Upstream deps:         source('ods','vw_bipccr')         -- preserve original object name
-- Purpose:              Produce the earliest *future* effective charge date per (company, account, site,
--                       container_group, charge_code, charge_type, charge_method).
-- Expected grain:       One row per (company_number, account_number, site, container_group, charge_code,
--                       charge_type, charge_method).
-- Primary key (output): The 7-key above (deterministic due to aggregation).
-- Materialization:      table
-- Determinism notes:    Uses CURRENT_TIMESTAMP exactly as original (time-volatile by design). Final audit
--                       columns also use CURRENT_TIMESTAMP as mandated by the spec. This is an intentional
--                       parity choice; outputs vary over time as in the original.
-- Tests to add:         - not_null: all key columns and future_eff_dt
--                       - unique: (company_number, account_number, site, container_group, charge_code,
--                                  charge_type, charge_method)
-- Assumptions:          - The source view  source('ods','vw_bipccr')  exposes the referenced columns with
--                         timestamp-compatible types for EFFECTIVE_DATE.
-- Optimizations:        Column pruning in CTE_source; predicate pushdown in CTE_filtered.
-- NO DIRECT TRANSLATION: The two audit fields in the final SELECT are added per the spec (not present in R).

{% enddocs %}
