{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_core_model_frontier') }}

select * from {{ source('ml_outputs', 'post_model_reference_selection_v1') }}
