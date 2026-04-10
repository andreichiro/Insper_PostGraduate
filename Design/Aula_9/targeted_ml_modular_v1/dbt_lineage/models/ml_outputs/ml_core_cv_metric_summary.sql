{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_core_cv_metric_folds') }}
-- depends_on: {{ ref('ml_core_model_fold_metrics') }}

select * from {{ source('ml_outputs', 'core_cv_metric_summary_v1') }}
