{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_bridge_mari_conversation_teacher') }}
-- depends_on: {{ source('raw_conceptual', 'fct_mari_ia_eventos_isso_ajudou') }}

select * from {{ source('modeled_base', 'fct_mari_help_resolved') }}
