{% docs aaq_process_Nik %}

-- - Upstream deps via ref()/source(): dwcore.active_account/infopro_region/employee_data/charge_cd_data,
--   aaq_ind_data, aaq_fees_pct_new, cg_w_rst_bipcc/bipccm/bipccr/bipcv, cg_w_cdpbirate,
--   bipcc*_future_rates, rev_funnel_last_pi, aaq_min_lift, aaq_special_handling, aaq_hierarchy_new (lineage),
--   aaq_structure (structure metadata)
-- - Grain / PK: 9-key (infopro_div_nbr_dcg, acct_nbr_dcg, site_nbr_dcg, container_grp_nbr_dcg,
--                      charge_code, charge_type, charge_method, unit_of_measure, wmt)
-- - Determinism: all windows ordered; banker’s rounding HALF-EVEN emulation; 1700→1753 only; volatile SYS_* fields documented
-- - NO DIRECT TRANSLATION (handled via comments below): Start/Collect/EndAudit, backups & restores,
--   RepopulateSQLTable/table swaps/emails, rate_restriction_audit(), monthly_rev_calc(), model_class_elig(),
--   budget_allocation(), nrd_realign(), ymp_execute(), dynamic rename/order driven by aaq_structure
-- - Tests to add: unique + not_null on 9-key; not_null(container_grp_nbr); length(rev_distrib_desc) ≤ 30; % ranges as needed
-- - Machine-learning candidate: interpretably predicting Is_True_Acquisition when date token is missing/ambiguous


aaq_process_Nik expects the following (non‑exhaustive) from each “query_*”:

RST (bipcc/bipccm/bipccr/bipccv): accomp, acacct, acsite, acctgr, charge_code, charge_method, charge_type, effective_date, rate_source, wmt, unit_of_measure, asslno, cgfran, cgmunp (+ rate columns used downstream).

CDP: same join keys + rate_source.

FR: infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr, charge_cd, charge_type, charge_method, waste_material_type, unit_of_measure, future_eff_dt.

IND: charge_code, charge_type, charge_method, unit_of_measure, wmt, effective_date, charge_rate, rate_eff_dt, hauls_per_month, units_per_haul, dsp_cost_per_unit, container_size, acct_nm, site_nm, addr_line_1, addr_line_2, cnrnt1, cnrnt2, cucrof, cuaddr, rev_distrib_desc, acshfl, acct_acquisition_desc, acquisition_cd, dsp_site_cd, land_fill_price_cd.

FEE_DETAIL: is_fee_charged_frf, is_fee_charged_evr, frf_rate_pct_dcf, erf_rate_pct_dcf, land_fill_cd_dcg, land_fill_price_cd_dcg, infopro_div_nbr_dcg, acct_nbr_dcg, site_nbr_dcg, container_grp_nbr_dcg.

PI: infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr, str_period.

{% enddocs %}
