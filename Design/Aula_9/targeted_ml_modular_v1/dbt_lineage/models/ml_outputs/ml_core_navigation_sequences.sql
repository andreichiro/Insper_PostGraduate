{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_mart_first_session_journey') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_core_definition_selection') }}
-- depends_on: {{ ref('ml_core_definition_frontier') }}

select * from {{ source('ml_outputs', 'core_navigation_sequences_v1') }}
