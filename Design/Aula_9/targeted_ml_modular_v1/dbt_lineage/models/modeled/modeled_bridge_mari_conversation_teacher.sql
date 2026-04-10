{{ config(materialized='view') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_reports') }}
-- depends_on: {{ source('raw_conceptual', 'stg_mari_ia_conversation') }}

select * from {{ source('modeled_base', 'bridge_mari_conversation_teacher') }}
