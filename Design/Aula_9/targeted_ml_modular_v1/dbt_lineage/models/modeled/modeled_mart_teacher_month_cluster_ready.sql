{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_mart_teacher_month_persona_ready') }}

select * from {{ source('modeled_base', 'mart_teacher_month_cluster_ready') }}
