{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_core_definition_selection') }}

select * from {{ source('ml_outputs', 'governance_label_registry_v1') }}
