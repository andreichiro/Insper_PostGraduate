{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_mart_first_session_journey') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_core_definition_selection') }}
-- depends_on: {{ ref('ml_core_definition_frontier') }}
-- depends_on: {{ ref('ml_governance_feature_registry') }}
-- depends_on: {{ ref('ml_governance_track_registry') }}

select * from {{ source('ml_outputs', 'core_scoring_scenarios_v1') }}
