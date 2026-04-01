{{ config(materialized='view') }}
-- depends_on: {{ source('raw_conceptual', 'dim_teachers') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_entries') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_contents_interactions') }}
-- depends_on: {{ source('raw_conceptual', 'stg_formation') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_conversation') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_reports') }}

select * from {{ source('modeled_base', 'dim_teacher') }}
