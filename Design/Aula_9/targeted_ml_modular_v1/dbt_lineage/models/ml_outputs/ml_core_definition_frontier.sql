{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_mart_future_metrics') }}
-- depends_on: {{ ref('ml_core_definition_selection') }}

select * from {{ source('ml_outputs', 'core_definition_frontier_v1') }}
