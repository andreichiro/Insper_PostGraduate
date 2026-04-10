{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_core_model_predictions') }}

select * from {{ source('ml_outputs', 'core_prediction_bootstrap_v1') }}
