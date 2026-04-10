{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_core_cv_score_folds') }}
-- depends_on: {{ ref('ml_core_model_predictions') }}

select * from {{ source('ml_outputs', 'core_cv_score_summary_v1') }}
