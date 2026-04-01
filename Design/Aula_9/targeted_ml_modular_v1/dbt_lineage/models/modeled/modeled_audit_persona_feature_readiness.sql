{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_mart_teacher_month_persona_ready') }}
-- depends_on: {{ ref('modeled_mart_teacher_persona_ready') }}

select * from {{ source('modeled_base', 'audit_persona_feature_readiness') }}
