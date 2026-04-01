{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_reports') }}

select * from {{ source('modeled_base', 'fct_mari_reports_resolved') }}
