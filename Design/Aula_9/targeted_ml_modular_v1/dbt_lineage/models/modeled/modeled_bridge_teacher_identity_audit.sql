{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_bridge_mari_conversation_teacher') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_entries') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_contents_interactions') }}
-- depends_on: {{ source('raw_conceptual', 'stg_formation') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_conversation') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_reports') }}
-- depends_on: {{ source('raw_conceptual', 'fct_mari_ia_eventos_isso_ajudou') }}

select * from {{ source('modeled_base', 'bridge_teacher_identity_audit') }}
