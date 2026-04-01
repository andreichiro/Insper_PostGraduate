{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_fct_interaction_clean') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_contents_interactions') }}

select * from {{ source('modeled_base', 'dim_event') }}
