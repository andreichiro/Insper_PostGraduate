{% docs aaq_active_account_new_lineage %}

aaq_active_account_new_0_int_active_accounts.sql
aaq_active_account_new_1_int_active_accounts.sql
aaq_active_account_new_2_int_active_accounts.sql

Sources:
  dwcore.dim_container_grp
  dwcore.dim_site
  dwcore.dim_acct
  dwcore.dim_corp_hier
  dwcore.dim_acct_attribute
  dwcore.dim_acct_fee
  dwcore.dim_customer
  dwcore_staging.RPT_CUST_MKTG_INFO

Final dbt lineage (DAG) — Correct and Reproducible
Sources (dwcore):
  ├─ dim_container_grp
  ├─ dim_site
  ├─ dim_acct
  └─ dim_corp_hier
            │
            ▼
aaq_active_account_new_0_int_active_accounts           -- reproduces R's out_dt (shared base)
   (ref consumers)
      ├───────────────────────────────────────┐
      │                                       │
      ▼                                       ▼
aaq_active_account_new_1_int_active_accounts  aaq_active_account_new_2_int_active_accounts
   (CGMS output; ref _0)                         (Active Account output; ref _0)
      │                                          │
      │                                          ├─ Sources (dwcore): dim_acct_attribute
      │                                          ├─ Sources (dwcore): dim_acct_fee
      │                                          └─ Sources (dwcore): dim_customer
      │
      └─ Source (dwcore_staging): RPT_CUST_MKTG_INFO   (RCMI email parse)


Independent/parallel models (from R runner):

aaq_active_account_new_query_aaq_charge_cds
aaq_active_account_new_query_aaq_employee_data_1
aaq_active_account_new_query_aaq_employee_data_2
aaq_active_account_new_query_aaq_sql_infopro_region


(These run independently from the active_account lineage, just like in R.)

📌 Grain / primary keys per node (for tests & contracts)

aaq_active_account_new_0_int_active_accounts (R out_dt):
Grain: (infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr).

aaq_active_account_new_1_int_active_accounts (R cgms):
Grain: (Infopro_Div_Nbr_Dcg, Acct_Nbr_Dcg, Site_Nbr_Dcg, Container_Grp_Nbr_Dcg) after deterministic dedupe.

aaq_active_account_new_2_int_active_accounts (R active_account):
Grain: still at the container group level
(infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr); fee flags are pivoted per (div,acct) and broadcast across container groups for that account (matches R).

{% enddocs %}
