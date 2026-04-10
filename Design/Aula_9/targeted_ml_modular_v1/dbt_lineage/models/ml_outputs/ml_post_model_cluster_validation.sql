{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_mart_teacher_cluster_ready') }}
-- depends_on: {{ ref('ml_post_model_cluster_assignment') }}

select * from {{ source('ml_outputs', 'post_model_cluster_validation_v1') }}
