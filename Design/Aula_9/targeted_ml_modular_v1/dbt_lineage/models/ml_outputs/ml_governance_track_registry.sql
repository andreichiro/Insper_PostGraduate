{{ config(materialized='view') }}

select * from {{ source('ml_outputs', 'governance_track_registry_v1') }}
