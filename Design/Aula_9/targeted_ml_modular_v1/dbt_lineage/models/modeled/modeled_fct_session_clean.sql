{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_fct_session_raw') }}

select * from {{ source('modeled_base', 'fct_session_clean') }}
