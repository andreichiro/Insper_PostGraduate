{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_fct_teacher_month') }}
-- depends_on: {{ ref('modeled_fct_session_clean') }}
-- depends_on: {{ ref('modeled_fct_interaction_clean') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}

select * from {{ source('ml_outputs', 'mart_onboarding_population_v1') }}
