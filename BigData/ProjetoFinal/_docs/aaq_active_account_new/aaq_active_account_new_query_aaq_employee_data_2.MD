/*aaq_active_account_new_query_aaq_employee_data_2.sql */

- Source R/T-SQL file(s): aaq_active_account_new_query_aaq_employee_data (excerpt)
- Upstream dependencies: {{ ref('aaq_active_account_new_query_aaq_employee_data_1') }}
- Purpose: Exact translation of the R query: filter to current employees and return DISTINCT six columns with Employee_EIN aliased to ASSLNO.
- Expected grain: DISTINCT tuples of (ASSLNO, Complete_Name, Job_Cd, Job_Desc, Emp_Job_Family, Emp_Job_Family_Grp).
- Primary key: none (distinct set defined by the six-column combination).
- Consumers: downstream joins to ods.ifp.BIPAS on ASSLNO (per R comment).
- Materialization: table.
- Tests to add (suggested):
    * not_null: ASSLNO
    * accepted_values for is_current_raw in upstream base
    * informational: uniqueness on the full six-column set (tautologically true by DISTINCT; included only to guard regressions)
- Assumptions: keep literal string comparison is_current_raw = '1' to match R; ignore the commented Div_SK filter.
- Determinism: DISTINCT result; no ordering guaranteed (same as R).
- NO DIRECT TRANSLATION: n/a (exact mapping).
