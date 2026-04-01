{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_fct_teacher_month') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}

select * from {{ source('modeled_base', 'base_modelada_v2') }}
