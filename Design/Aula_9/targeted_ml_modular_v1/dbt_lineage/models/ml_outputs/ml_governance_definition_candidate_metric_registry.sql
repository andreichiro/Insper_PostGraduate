{{ config(materialized='view') }}

select * from {{ source('ml_outputs', 'governance_definition_candidate_metric_registry_v1') }}
