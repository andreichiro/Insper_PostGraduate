{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_conversation') }}

select * from {{ source('modeled_base', 'fct_mari_conversation_resolved') }}
