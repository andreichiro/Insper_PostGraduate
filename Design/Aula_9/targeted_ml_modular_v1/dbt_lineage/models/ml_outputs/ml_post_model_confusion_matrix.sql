{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_post_model_threshold_metrics') }}
-- depends_on: {{ ref('ml_core_model_predictions') }}
-- depends_on: {{ ref('ml_post_model_reference_selection') }}
-- depends_on: {{ ref('ml_mart_first_session_journey') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_core_definition_selection') }}
-- depends_on: {{ ref('ml_governance_feature_registry') }}
-- depends_on: {{ ref('ml_core_scoring_scenarios') }}

select * from {{ source('ml_outputs', 'post_model_confusion_matrix_v1') }}
