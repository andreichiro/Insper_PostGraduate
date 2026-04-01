{{ config(materialized='view') }}
-- depends_on: {{ ref('ml_governance_feature_registry') }}
-- depends_on: {{ ref('ml_governance_track_registry') }}

select * from {{ source('ml_outputs', 'governance_feature_eligibility_v1') }}
