{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_post_model_heavy_user_scores') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_core_model_predictions') }}
-- depends_on: {{ ref('ml_post_model_reference_selection') }}

select * from {{ source('ml_outputs', 'post_model_heavy_user_summary_v1') }}
