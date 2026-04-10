{{ config(materialized='view') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_entries') }}
-- depends_on: {{ source('raw_conceptual', 'dim_teachers') }}

select * from {{ source('modeled_base', 'fct_session_raw') }}
