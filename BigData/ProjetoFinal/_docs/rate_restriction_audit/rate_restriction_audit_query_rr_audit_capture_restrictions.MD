{% docs rate_restriction_audit_query_rr_audit_capture_restrictions  %}

"rate_restriction_audit_query_rr_audit_capture_restrictions"
Upstream deps (sources):   source('cpq','rpt_quote_detail') ,
                           source('dwcore','dim_site') ,
                           source('dwcore','dim_corp_hier') 
Expected grain:           Post-join DISTINCT over (quote_number, div, acct, site, eff_dt, approval_dt, created_dt, rate_year_1..3).
Primary key:              None (join may be many-to-many) — see DISTINCT note.
Consumers:                Rate restriction audit (CPQ ↔ DS).
Materialization:          table (stable snapshot per run).
Determinism:              Replace GETDATE() with run-anchored  run_started_at  (UTC).
Assumptions:              Warehouse: Snowflake; timestamps are NTZ; outputs are DATEs.
Rounding:                 ROUND(...,0) => HALF_AWAY_FROM_ZERO (Snowflake/SQL Server parity).
NO DIRECT TRANSLATION:    GETDATE() → run_anchored timestamp; SQL Server collation → explicit UPPER(); implicit casts → TRY_TO_NUMBER.
Optimizations:            Predicate pushdown; single-pass normalization.
Tests to add:             not_null(quote_number, div, acct, site, eff_dt, approval_dt, created_dt);
                          accepted_values(signed_contract={'Signed'}) [if modeled];
                          relationship(quote_number → contract_nbr on active subset).
Machine Learning candidate: Tag added. Use a simple decision tree to flag anomalies where 2nd/3rd-year nulling
                            deviates from rules; interpretable, useful for audit triage.
{% enddocs %}