/*aaq_active_account_new_query_aaq_employee_data_1.sql */

- Source R/T-SQL file(s): aaq_active_account_new_query_aaq_employee_data (excerpt provided)
- Upstream dependencies: {{ source('dwcore','dim_employee') }}
- Purpose: Base layer — single place to select and minimally normalize the required fields for employee records.
- Expected grain: row-per-record from DWCORE.dbo.Dim_Employee (no dedup, no filtering).
- Primary key: none enforced here; downstream model filters/dedups.
- Outputs/consumers: aaq_active_account_new_query_aaq_employee_data_2 (joins later to ods.ifp.BIPAS on ASSLNO per R comment).
- Materialization: table (stable snapshot per run).
- Tests to add (suggested): not_null on employee_ein; values accepted for is_current_raw.
- Assumptions: is_Current in source is a string field whose literal '1' denotes “current”; we do not coerce booleans to preserve exact R semantics.
- Determinism: no ordering applied; model is deterministic given the same source snapshot. No NOW() used.
