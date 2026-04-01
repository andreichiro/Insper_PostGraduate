{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_mart_onboarding_population') }}
-- depends_on: {{ ref('modeled_fct_session_clean') }}
-- depends_on: {{ ref('modeled_fct_interaction_clean') }}

select * from {{ source('ml_outputs', 'mart_first_session_journey_v1') }}
