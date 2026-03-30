{% docs aaq_process_Nik_query_aaq_hierarchy_new %}

Source R file(s): aaq_process_Nik_query_aaq_hierarchy_new (SQL inside R; Lines 01–30)
Upstream dependencies (source()):
  -  source('ods','ods_ifp_rs600wf@b')   -- original: ods.dbo.ods_ifp_rs600wf@b  (Lines 09)
  -  source('ods','ods_ifp_rs600wf@a')   -- original: ods.dbo.ods_ifp_rs600wf@a  (Lines 10)

Expected grain & join risk:
  - Result grain = rows from an INNER JOIN on t0.dwfblwscmp = t1.dwfalwscmp (Lines 11)
  - Many‑to‑many risk if either side is non‑unique on the join key; behavior preserved.

Outputs / consumers: not specified (unknown)

Materialization choice: table (default; aligns with stable, reusable output)

Tests to add (outside this model):
  - not_null: {dwfainflb, dwfainfdvs, dwfbarea, dwfbbu, dwfbreg, dwfblwscmp, dwfalwsnac}
  - relationship tests on join keys to detect orphaned keys
  - (Optional) accepted_values on sys_del_ind = 'N' upstream if modeled

Assumptions:
  - Case/locale‑specific ORDER BY in the original is for presentation only; models are unordered.

Optimizations:
  - Removed redundant NOT NULL predicates and unnecessary wrapper subquery (see notes).

NO DIRECT TRANSLATION notes:
  - ORDER BY ... COLLATE latin1_general_bin (Lines 25–30): omitted to avoid false determinism.
    Downstream readers should ORDER BY as needed with explicit collation.
  - Required runtime columns use CURRENT_TIMESTAMP per spec; this introduces run‑time variance.
    If perfect replay is ever required, consider swapping to  run_started_at  (documented here)
    but kept CURRENT_TIMESTAMP to satisfy the “always include” requirement verbatim.

Unresolved dependencies:
  - If your sources.yml uses a different source name (e.g., 'dwcore'), update only the source
    name in  source('…', 'ods_ifp_rs600wf@a|b') . Do not rename the table identifiers.

Machine learning candidate: no (pure relational join/filter; no predictive step to justify ML).

{% enddocs %}
