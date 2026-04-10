{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_fct_interaction_clean') }}
-- depends_on: {{ ref('modeled_fct_session_raw') }}

select * from {{ source('modeled_base', 'fct_teacher_month') }}
