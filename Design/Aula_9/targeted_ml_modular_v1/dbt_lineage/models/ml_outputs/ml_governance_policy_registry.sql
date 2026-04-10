{{ config(materialized='view') }}

select * from {{ source('ml_outputs', 'governance_policy_registry_v1') }}
