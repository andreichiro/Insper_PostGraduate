{{ config(materialized='view') }}

select * from {{ source('ml_outputs', 'governance_feature_registry_v1') }}
