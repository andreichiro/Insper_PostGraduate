{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}
-- depends_on: {{ source('raw_conceptual', 'stg_formation') }}

select * from {{ source('modeled_base', 'fct_formation_clean') }}
