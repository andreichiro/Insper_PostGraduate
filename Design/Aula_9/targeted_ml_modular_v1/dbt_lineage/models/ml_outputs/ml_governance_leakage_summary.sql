{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_governance_leakage_audit') }}

select * from {{ source('ml_outputs', 'governance_leakage_summary_v1') }}
