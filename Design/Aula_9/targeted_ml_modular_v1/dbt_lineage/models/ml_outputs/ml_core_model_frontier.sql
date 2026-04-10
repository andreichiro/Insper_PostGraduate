{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_core_model_fold_metrics') }}
-- depends_on: {{ ref('ml_core_model_predictions') }}

select * from {{ source('ml_outputs', 'core_model_frontier_v1') }}
