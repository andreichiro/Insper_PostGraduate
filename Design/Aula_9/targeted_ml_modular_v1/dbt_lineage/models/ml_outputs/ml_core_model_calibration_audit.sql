{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_mart_first_session_journey') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_core_definition_selection') }}
-- depends_on: {{ ref('ml_governance_feature_registry') }}
-- depends_on: {{ ref('ml_core_scoring_scenarios') }}

select * from {{ source('ml_outputs', 'core_model_calibration_audit_v1') }}
