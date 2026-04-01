{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_core_model_frontier') }}

select * from {{ source('ml_outputs', 'core_definition_b_excessive_separation_v1') }}
