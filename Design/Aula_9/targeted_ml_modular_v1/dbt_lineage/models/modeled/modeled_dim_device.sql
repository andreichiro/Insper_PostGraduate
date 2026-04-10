{{ config(materialized='view') }}

select * from {{ source('modeled_base', 'dim_device') }}
