{% docs business_rule_function_query_testing_business_rules %}

- Purpose: Reproduce the exact filter of active test rules within an effective window; exclude IDs matching pattern '%_H0' where '_' is a single-character wildcard.
- Upstream dependencies (dbt source()): source('ymp','ymp_test_rules_new')
- Expected grain / PK: Not enforced; pass-through rows from source after filtering. Downstream should not assume uniqueness.
- Materialization: view (read-only selection). -- OPTIMIZATION: a view avoids storage and stays aligned with upstream freshness.
- Determinism & time anchors:
  * Business filters require var('cur_model_1') and var('cur_model_2'); no defaults are injected (run will fail if they are missing).
  * Audit columns (SYS_EXEC_ID, SYS_CREATE_DTM) are volatile by design and do not affect selection results. -- NO DIRECT TRANSLATION: non-ANSI formatting functions mandated by spec.
- LIKE/case-sensitivity:
  * SQL Server collations are typically case-insensitive. To preserve that behavior across warehouses, we normalize with UPPER() (ANSI) so the comparison is case-insensitive while retaining '_' as a wildcard. -- NO DIRECT TRANSLATION: cross-engine approximation documented.
- Timestamp semantics: ASSUMPTION — variables represent timezone-naïve instants; CAST(... AS TIMESTAMP) uses the warehouse default (e.g., TIMESTAMP_NTZ if available).
- Ordering: No ORDER BY in R; result set ordering is intentionally unspecified (tables are unordered).
- Tests to add (recommended): not_null(rule_nbr), not_null(test_id); accepted_values(is_updated in (0), is_deleted in (0))

{% enddocs %}