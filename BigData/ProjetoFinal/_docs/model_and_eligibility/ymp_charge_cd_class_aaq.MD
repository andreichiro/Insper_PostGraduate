{% docs ymp_charge_cd_class_aaq %}
-- Source R/SQL file: ymp_charge_cd_class_aaq
-- Upstream: source('ymp','active_account_query')
-- Grain: DISTINCT pairs (Charge_Cd, Charge_Cd_Desc)
-- Materialization: table (anchors SYS_* to run time; see determinism note)
-- Tests suggested: not_null on Charge_Cd, Charge_Cd_Desc; unique on (Charge_Cd, Charge_Cd_Desc)
-- NO DIRECT TRANSLATION: use source() instead of hard-coded 'ymp.dbo.active_account_query'
-- NO DIRECT TRANSLATION: enforce case-insensitive comparison for Rate_Source via UPPER() to emulate common SQL Server collations
-- Determinism note: per spec, SYS_* fields use CURRENT_TIMESTAMP (volatile); data rows are deterministic given inputs

{% enddocs %}