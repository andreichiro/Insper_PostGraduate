{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_governance_definition_candidate_metric_registry') }}

select * from {{ source('ml_outputs', 'core_definition_selection_v1') }}
