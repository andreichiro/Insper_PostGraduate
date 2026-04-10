{{ config(materialized='view') }}
-- depends_on: {{ source('raw_conceptual', 'school_calendar') }}

select * from {{ source('modeled_base', 'dim_calendar') }}
