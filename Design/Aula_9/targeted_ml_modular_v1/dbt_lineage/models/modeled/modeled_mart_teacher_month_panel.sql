{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_base_modelada_v2') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}

select * from {{ source('modeled_base', 'mart_teacher_month_panel') }}
