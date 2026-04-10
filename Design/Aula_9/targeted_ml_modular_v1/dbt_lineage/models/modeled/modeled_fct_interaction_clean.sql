{{ config(materialized='view') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_contents_interactions') }}
-- depends_on: {{ source('raw_conceptual', 'dim_teachers') }}
-- depends_on: {{ source('raw_conceptual', 'stg_lessons') }}

select * from {{ source('modeled_base', 'fct_interaction_clean') }}
