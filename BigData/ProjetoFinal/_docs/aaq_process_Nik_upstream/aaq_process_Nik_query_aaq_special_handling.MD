{% docs aaq_process_Nik_query_aaq_special_handling %}

-- Upstream dependencies:  source('ODS','vw_BIPTB')   -- binds [ODS].[IFP].[vw_BIPTB] via sources.yml
-- Expected grain & PK: unknown; passthrough rows where TBTBLN='SPH' and flags; no dedupe
-- Outputs/consumers: not specified
-- Materialization: table
-- Tests to add (schema.yml):
--   - accepted_values: { column_name: TBTBLN, values: ['SPH'] }
--   - not_null: [TBTBLN, TBDESC, TBELEM, Region]  -- adjust if nulls are valid
--   - sources.yml freshness test on source('ODS','vw_BIPTB') if applicable
-- Assumptions:
--   * Case sensitivity: Many T‑SQL deployments use CI collations.
--     -- NO DIRECT TRANSLATION (collation): Cross‑warehouse parity achieved via UPPER() comparisons.
--   * Trailing spaces: SQL Server equality often ignores trailing spaces; Snowflake/ANSI may not.
--     -- NO DIRECT TRANSLATION (trailing spaces): If your source pads with trailing spaces, consider RTRIM()
--        in the predicates to mirror SQL Server behavior. Not applied here to avoid unintended leading-space changes.
--   * Audit columns: You require CURRENT_TIMESTAMP‑based fields.
--     -- NO DIRECT TRANSLATION (ANSI): The mandated TO_VARCHAR/TO_NUMBER format is platform‑specific by spec.
-- Determinism & reproducibility:
--   * Data outputs are stable for identical inputs; audit fields vary per run by design (CURRENT_TIMESTAMP).
-- Optimizations:
--   -- OPTIMIZATION: Project only needed columns for predicate pushdown and reduced I/O.
-- Unresolved dependencies:
--   * Ensure sources.yml defines source name 'ODS' (schema 'IFP') with table 'vw_BIPTB'.

{% enddocs %}
