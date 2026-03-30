
{% docs business_rule_function_query_aaq_data_types %}
-- Upstream deps: ref('active_account_query')  -- resolve db/schema/table deterministically (no hard-coding)
-- Expected grain: 1 row per column of the referenced table
-- Output: column_name, data_type, r_data_type + SYS columns
-- Materialization: table
-- Determinism: No top-level ORDER BY (tables are unordered). No window functions used.
-- Optimizations: 
--   * OPTIMIZATION: Predicate pushdown to (table_schema, table_name) for INFORMATION_SCHEMA.COLUMNS to limit scan.
-- NO DIRECT TRANSLATION notes:
--   * Snowflake normalizes many numeric aliases to NUMBER; we use NUMERIC_SCALE to reproduce the R intent
--     of distinguishing integers from decimals/floats. This is required for parity. (Docs: data type aliases) 
-- Anti-pattern (documented, preserved):
--   * TIME → 'as.Date' matches the original R CASE, even though it discards time-of-day. Consider revisiting downstream.
-- Tests to add:
--   * not_null: column_name
--   * accepted_values: r_data_type ∈ {'as.character','as.integer','as.numeric','as.Date'} OR NULL


{% enddocs %}