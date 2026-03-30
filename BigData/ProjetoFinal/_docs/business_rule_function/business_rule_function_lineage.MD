{% docs business_rule_function_lineage %}

**Purpose (matches R):**
Implements the YMP business‑rule engine over Active Account Query (AAQ), reproducing the R scripts’ behavior:
- reads AAQ + (optionally) the test lineage table
- pulls the active rules (either PROD or TEST flavor)
- loads rule nomenclature tokens and AAQ data types
- evaluates Filter and Assign rules in the defined global order
- writes back derived assignments (Rounding Policy, Adjustment Reason, optional PI fields)
- emits account‑level eligibility (rate + winning rule) broadcast back to AAQ grain

### Models & files

**Final model (this file):**
- *(filename in repo)* → **business_rule_function** (materialized **table**)

**Supporting models used by the final model:**
- `business_rule_function_query_business_rules.sql` (materialized **table**)
- `business_rule_function_query_testing_business_rules.sql` (materialized **view**)
- `business_rule_function_query_aaq_data_types.sql` (materialized **table**)
- `business_rule_function_query_nomenclature.sql` (materialized **table**)

---

### Sources consumed (as in R)

- `ymp_staging.active_account_query`  *(AAQ; primary input)*
- `ymp_staging.ymp_test_rules_join`   *(only when `var('test_rules')` = true; time‑bounded join)*
- `ymp_staging.ymp_rules`             *(rule catalog for PROD branch)*
- `ymp.ymp_test_rules_new`            *(rule catalog for TEST branch)*
- `ymp.bus_rule_nom`                  *(nomenclature tokens for rule parsing)*
- **Information Schema** for the AAQ table (drives R‑style `as.*` type mapping used by the parser):
  ` source('ymp_staging','active_account_query').database .information_schema.columns`

### Final dbt lineage (DAG) — Correct and Reproducible

Sources:
├─ ymp_staging.active_account_query
├─ ymp_staging.ymp_test_rules_join (optional when test_rules=true)
├─ ymp_staging.ymp_rules
├─ ymp.ymp_test_rules_new
└─ ymp.bus_rule_nom
│
├─────────────┬───────────────────────────┬─────────────────────────────┐
│ │ │ │
▼ ▼ ▼ ▼
business_rule_function_query_business_rules business_rule_function_query_testing_business_rules
(PROD rules view over ymq_rules) (TEST rules view over ymp_test_rules_new)

business_rule_function_query_aaq_data_types business_rule_function_query_nomenclature
(AAQ INFO_SCHEMA mapping) (nomenclature tokens)
│ │ │
└──────────────┬─────┴───────────┬──────┘
▼ ▼
business_rule_function (this model; materialized table)
│
└─ Output: AAQ + derived columns

perl
Copy code

**Branching / switches:**
- `var('test_rules', false)`  
  → if **true**: use `business_rule_function_query_testing_business_rules` + join `ymp_staging.ymp_test_rules_join`  
  → if **false**: use `business_rule_function_query_business_rules`
- `var('prcs_type', ['YMP Eligibility'])`  
  → filters rules by process_type; when it includes `'YMP PI Logic'`, extra outputs (PI%) and (PI$) are produced and AAQ rows are extended with nullable stubs if missing
- `var('approved', [1])` filters rules by approval flag
- `var('budget_run', false)` excludes Corporate owners and restricts rule_nbr ≥ 5100 (mirrors R budget path)
- `var('aaq_filter', '')` optional AAQ row subset (propagates to current model timestamp selection)
- `var('test_as_of_current', false)` controls the as‑of timestamp used in the **test** lineage join
- `var('cols_tolower', [])` lower‑cases selected AAQ columns for case‑sensitive match parity with R

---

### Inputs → internal “R objects” → outputs (1:1 mapping to R)

| R object / step (commented in SQL)              | dbt / SQL construct                                                                                     |
|-------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| AAQ read (+ NULL stubs for PI/Adj Reason)       | `CTE_source` from `source('ymp_staging','active_account_query')`, conditional stubs for PI/Adj Reason    |
| Optional test lineage join                      | `CTE_source` LEFT JOIN `source('ymp_staging','ymp_test_rules_join')` (when `test_rules=true`)            |
| Current model timestamp (R: cur_ymp_nrd_yyyymmdd)| `CTE_cur_model` (min date across filtered AAQ; deterministic warning when multiple)                      |
| Rule catalog (PROD/TEST)                        | `CTE_rules` from `ref('..._business_rules')` or `ref('..._testing_business_rules')`                      |
| Global order setorderv                           | `CTE_rules_ordered` (+ `CTE_rules_maxpos`)                                                               |
| Nomenclature tokens                              | `CTE_nom_tokens` from `ref('business_rule_function_query_nomenclature')`                                 |
| AAQ type mapping (R as.*)                        | `CTE_supported_cols` from `ref('business_rule_function_query_aaq_data_types')` + explicit extra columns  |
| Filter code normalization                        | `CTE_norm_rules`, `CTE_expr_std`, `CTE_expr_std2`                                                        |
| Filter parsing (OR/AND/atoms)                    | `CTE_or_groups`, `CTE_and_atoms`, `CTE_atoms_classified` + `CTE_in_values`/`CTE_cmp_atoms`/…             |
| Assign parsing (OR/AND/atoms)                    | `CTE_or_groups_assign`, `CTE_and_atoms_assign`, `CTE_atoms_classified_assign` + *_assign CTEs            |
| Per‑account state (R: rates_tbl)                 | `CTE_seq_seed`                                                                                            |
| Recursive loop (apply rules in order)            | `CTE_seq_state` (mutates Assign targets; computes eligibility from Filter matches)                       |
| Final state per (div,acct)                       | `CTE_seq_last`                                                                                            |
| Eligibility output (per account)                 | `CTE_eligibility`                                                                                         |
| Apply Assign to AAQ rows                         | `CTE_with_assign` (joins seq_last back to full AAQ rows)                                                 |
| Final projection                                 | `CTE_final_base` → `CTE_final` → **final SELECT**                                                        |

---

### Grains / primary keys (for tests & contracts)

- **business_rule_function** (final output):  
  **Grain = AAQ row grain** (whatever `ymp_staging.active_account_query` emits).  
  New columns broadcast per *(Infopro_Div_Nbr, Acct_Nbr)*:  
  - `YMP_Eligible_Rate` (0/1)  
  - `YMP_Elig_Bus_Rule` (rule_id that set the eligibility)  
  - Optional: `Adjustment_Reason`, `Recommended_PI_Percentage`, `Recommended_PI_Amount` (if enabled)

- **Eligibility internal state** (not materialized):  
  `CTE_seq_last` / `CTE_eligibility` grain = *(Infopro_Div_Nbr, Acct_Nbr)*

- **Rules tables**:  
  `business_rule_function_query_business_rules` & `..._testing_business_rules` grain = *(rule_id)*

- **Nomenclature**: pass‑through (same as `ymp.bus_rule_nom`)

- **AAQ data types**: per *column_name* in AAQ

---

### Materializations (match R outputs)

- **business_rule_function** → **table** (final reproducible output; used downstream by YMP and ML candidates)  
- **business_rule_function_query_business_rules** → **table** (stable as‑of snapshot semantics)  
- **business_rule_function_query_testing_business_rules** → **view** (refreshed view of test rule catalog)  
- **business_rule_function_query_aaq_data_types** → **table** (snapshot of AAQ column typing used by parser)  
- **business_rule_function_query_nomenclature** → **table** (stable copy to avoid token drift during runs)

---

### Validation notes to mirror R semantics

1. **Sequential semantics:** The loop in `CTE_seq_state` applies rules in **global order** (process_type, rule_nbr, precedence, rule_id); last write wins—same as R.  
2. **Case sensitivity:** GREP and SUBSTR behave case‑sensitively unless a column is listed in `var('cols_tolower')`.  
3. **Eligibility grain:** Computed per (div,acct) and then joined back to AAQ rows, matching the R broadcast.  
4. **Time windows:** PROD rules honor `run_started_at` (as‑of); TEST rules honor model date from AAQ (`CTE_cur_model`) and the `test_as_of_current` knob for the lineage join window.  

{% enddocs %}
