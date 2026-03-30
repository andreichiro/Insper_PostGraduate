{% docs model_class_and_eligibility %}
Source R file(s): model_class_elig.R  (sourced: YMP/business_rule_function.R)
Upstream deps:
  - source('ymp','active_account_query')         -- AAQ base
  - ref('ymp_charge_cd_class_db')                -- active charge-code classes (windowed eff/disc)
  - ref('ymp_charge_cd_class_aaq')               -- AAQ charge-codes (eligible & non-zero; excludes BIPCV)
  - ref('business_rule_function')                -- rules engine output (Model Class, YMP Eligibility)
Expected grain:
  - Row grain = AAQ rows (infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr, charge_cd, charge_cd_desc, …)
Materialization: table
Determinism:
  - All windows have explicit PARTITION BY; dedupe uses a stable, preference-aware tiebreaker.
  - SYS_* fields intentionally volatile per requirement.
Assumptions & NO DIRECT TRANSLATION:
  - The rules model must emit for prcs_type='Model Class': (infopro_div_nbr_norm, acct_grp_key_int, acct_model_class_mcf).
    For prcs_type='YMP Eligibility': (infopro_div_nbr, acct_nbr, site_nbr, container_grp_nbr, charge_cd, ymp_eligible_rate).
  - R computes `elig_cc` BEFORE calling rules; the rules may rely on it. We compute it as a pre-feature and
    document that the rules model should either reproduce or read this feature. We drop it from the final select to
    match the R lifecycle.
  - R has two branches for AAQ charge codes (DB helper excludes BIPCV; in-memory may include). We standardize on the DB
    helper for deterministic dbt runs. If exact parity across both branches is ever required, apply a union here.
Tests (recommend):
  - not_null: infopro_div_nbr, acct_nbr, charge_cd
  - accepted_values: ymp_eligible_* ∈ {0,1}
  - data test: rollup monotonicity (cont ≤ site ≤ acct ≤ div)
Interpretable ML candidate:
  - The rule-based `acct_model_class`/eligibility can be complemented with a CART or shallow linear model to provide
    reason codes and contour checks. Low-dimensional, highly interpretable → tagged machine_learning_candidate.

{% enddocs %}