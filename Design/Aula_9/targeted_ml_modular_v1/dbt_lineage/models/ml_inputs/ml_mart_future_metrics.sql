{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_mart_first_session_journey') }}
-- depends_on: {{ ref('modeled_fct_session_clean') }}
-- depends_on: {{ ref('modeled_fct_interaction_clean') }}
-- depends_on: {{ ref('modeled_fct_formation_clean') }}
-- depends_on: {{ ref('modeled_fct_mari_conversation_resolved') }}
-- depends_on: {{ ref('modeled_fct_mari_help_resolved') }}

select * from {{ source('ml_outputs', 'mart_future_metrics_v1') }}
