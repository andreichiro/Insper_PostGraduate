{% docs model_class_and_eligibility_lineage %}

Sources:
  ymp.active_account_query
  ymp.ymp_charge_cd_class
  (separate branch) ymp_staging.active_account_query
  (separate branch) ymp_staging.ymp_rules
  (plus) nomenclature & data_types refs for rules

DAG (current)
  source(ymp.ymp_charge_cd_class) ──▶ ymp_charge_cd_class_db (view)
  source(ymp.active_account_query) ──▶ ymp_charge_cd_class_aaq (table)
  source(ymp.active_account_query) ──▶ model_class_aaq (table)

  source(ymp.active_account_query) ─┐
  ref(ymp_charge_cd_class_db)      ├─▶ model_class_and_eligibility (table)
  ref(ymp_charge_cd_class_aaq)     ┤
  ref(business_rule_function)      ┘     ← expects 2 shapes (Model Class; AAQ-grain Eligibility)

  source(ymp_staging.active_account_query) ─┐
  ref(business_rule_function_query_business_rules) ─▶ business_rule_function (recursive rules engine)
  ref(business_rule_function_query_nomenclature)   ─┘
  ref(business_rule_function_query_aaq_data_types) ─┘

{% enddocs %}