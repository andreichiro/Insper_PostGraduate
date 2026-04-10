{{ config(materialized='view') }}
-- depends_on: {{ ref('modeled_base_modelada_v2') }}
-- depends_on: {{ ref('modeled_fct_teacher_month') }}
-- depends_on: {{ ref('modeled_dim_teacher') }}
-- depends_on: {{ ref('modeled_dim_event') }}
-- depends_on: {{ ref('modeled_dim_device') }}
-- depends_on: {{ ref('modeled_dim_calendar') }}
-- depends_on: {{ ref('modeled_fct_session_raw') }}
-- depends_on: {{ ref('modeled_fct_session_clean') }}
-- depends_on: {{ ref('modeled_fct_interaction_clean') }}
-- depends_on: {{ ref('modeled_fct_formation_clean') }}
-- depends_on: {{ ref('modeled_dim_lesson') }}
-- depends_on: {{ ref('modeled_bridge_teacher_identity_audit') }}
-- depends_on: {{ ref('modeled_bridge_mari_conversation_teacher') }}
-- depends_on: {{ ref('modeled_fct_mari_conversation_resolved') }}
-- depends_on: {{ ref('modeled_fct_mari_reports_resolved') }}
-- depends_on: {{ ref('modeled_fct_mari_help_resolved') }}
-- depends_on: {{ ref('modeled_mart_teacher_month_persona_ready') }}
-- depends_on: {{ ref('modeled_mart_teacher_persona_ready') }}
-- depends_on: {{ ref('modeled_mart_teacher_month_cluster_ready') }}
-- depends_on: {{ ref('modeled_mart_teacher_cluster_ready') }}
-- depends_on: {{ ref('modeled_mart_teacher_month_panel') }}
-- depends_on: {{ ref('modeled_audit_persona_feature_readiness') }}
-- depends_on: {{ ref('modeled_dim_persona_range_candidates') }}
-- depends_on: {{ source('raw_conceptual', 'fct_teachers_contents_interactions') }}
-- depends_on: {{ source('raw_conceptual', 'dim_teachers') }}
-- depends_on: {{ source('raw_conceptual', 'stg_lessons') }}

select * from {{ source('modeled_base', 'audit_base_modelada_validation') }}
