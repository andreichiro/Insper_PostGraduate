{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_post_model_reference_selection') }}
-- depends_on: {{ ref('ml_post_model_feature_importance') }}
-- depends_on: {{ ref('ml_mart_first_session_journey') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_core_definition_selection') }}
-- depends_on: {{ ref('ml_governance_feature_registry') }}
-- depends_on: {{ ref('ml_core_scoring_scenarios') }}

select * from {{ source('ml_outputs', 'governance_post_model_output_status_v1') }}
