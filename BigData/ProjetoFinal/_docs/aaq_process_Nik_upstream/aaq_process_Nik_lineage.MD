{% docs aaq_process_Nik_lineage %}

R: Active Account new (0)     ──►  R: Active Account new (1)   ──►  R: Active Account new (2)
         │                                   │                              │
         ▼                                   ▼                              ▼
dbt: aaq_active_account_new_0_int_active_accounts
                                        └─► dbt: aaq_active_account_new_1_int_active_accounts (CGMS export path)
dbt: aaq_active_account_new_2_int_active_accounts  (main active account surface)
         │
         ▼
dbt: aaq_process_Nik  (the main AAQ build; consumes *_2_int_active_accounts)
         │
         ▼
dbt: aaq_process_Nik_downstream  (structure-driven rename/reorder, SYS_*)

Secondary “query_*” models consumed by aaq_process_Nik (mirror of R’s sqlProc/run_dwcore_table):

aaq_process_Nik_query_aaq_ind_data → IND inputs (rates, volumes, names, disposal hints).

aaq_process_Nik_query_aaq_fees_pct_new → fee_detail (FRF/ERF flags & %; DCG fallback disposal codes).

aaq_process_Nik_query_aaq_hierarchy_new → lineage only (R queried, later unused).

ODS RST:

aaq_process_Nik_query_cg_w_rst_bipcc

aaq_process_Nik_query_cg_w_rst_bipccm

aaq_process_Nik_query_cg_w_rst_bipccr

aaq_process_Nik_query_cg_w_rst_bipcv

CDP deleted rates:

aaq_process_Nik_query_cg_w_cdpbirate

Future rates:

aaq_process_Nik_query_bipcc_future_rates

aaq_process_Nik_query_bipccr_future_rates

aaq_process_Nik_query_bipccm_future_rates

PI alignment window:

aaq_process_Nik_query_rev_funnel_last_pi

{% enddocs %}
