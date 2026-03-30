{% docs business_rule_function_query_nomenclature %}
-- Source R file: YMP_rules_function (nomenclature loader; R lines ~145–176)
-- Upstream source(): ymp.bus_rule_nom
-- Purpose: Stage the nomenclature table used by the rules compiler.
-- Grain/PK: whatever the source provides (no changes).
-- Determinism: full snapshot at run-time; no transforms.
-- Tests to add: not_null on (Class, Translation_Type, Nomenclature, Rule_Order)

{% enddocs %}