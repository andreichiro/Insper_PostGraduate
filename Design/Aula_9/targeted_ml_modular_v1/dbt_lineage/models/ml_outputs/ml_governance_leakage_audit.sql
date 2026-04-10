{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_governance_feature_registry') }}
-- depends_on: {{ ref('ml_governance_label_registry') }}
-- depends_on: {{ ref('ml_core_scoring_scenarios') }}

select * from {{ source('ml_outputs', 'governance_leakage_audit_v1') }}
