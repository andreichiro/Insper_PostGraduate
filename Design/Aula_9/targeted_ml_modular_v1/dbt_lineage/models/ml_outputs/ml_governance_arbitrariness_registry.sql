{{ config(materialized='view') }}

select * from {{ source('ml_outputs', 'governance_arbitrariness_registry_v1') }}
