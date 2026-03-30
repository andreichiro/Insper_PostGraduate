{% docs model_class_aaq %}

-- Source R/T‑SQL: model_class_aaq (selection from AAQ)
-- Purpose: reproduce T‑SQL selection with computed flags and filters
-- Grain: deduplicated across selected columns (SELECT DISTINCT parity)
-- Materialization: table (freezes SYS_* at build time)
-- Determinism: SYS_* fields use CURRENT_TIMESTAMP per requirement (volatile by design)
-- Note on string comparisons:
--   -- NO DIRECT TRANSLATION: SQL Server default collations are case-insensitive; Snowflake is case-sensitive.
--   We apply UPPER() to preserve original semantics without relying on database collation.

{% enddocs %}