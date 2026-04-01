{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_fct_interaction_clean') }}
-- depends_on: {{ source('raw_conceptual', 'stg_lessons') }}

select * from {{ source('modeled_base', 'dim_lesson') }}
