{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_post_model_heavy_user_scores') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}

select * from {{ source('ml_outputs', 'post_model_heavy_user_profile_v1') }}
