#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from common import html_to_text, load_manifest_spec, read_json, strip_survival_sections_from_html, utc_now_iso, write_json


def extract_sections(html: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    for block in re.findall(r"<section>.*?</section>", html, flags=re.S | re.I):
        h = re.search(r"<h2>(.*?)</h2>", block, flags=re.S | re.I)
        title = re.sub(r"\s+", " ", h.group(1)).strip() if h else "(sem_titulo)"
        text = html_to_text(block)
        sections.append((title, text))
    return sections


def sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.!\?])\s+", text)
    out = []
    for p in parts:
        s = re.sub(r"\s+", " ", p).strip()
        if len(s) >= 30:
            out.append(s)
    return out


def select_claim_candidates(sentences: List[str], claim_rules: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for s in sentences:
        low = s.lower()
        matched_rule = any(re.search(rule.get("match_regex", ""), low, flags=re.I) for rule in claim_rules)
        has_result_marker = "resultado:" in low
        is_causal_like = classify_claim(s) == "causal_or_extrapolative"
        if matched_rule or has_result_marker or is_causal_like:
            out.append(s)
    # Preserve order and uniqueness.
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def classify_claim(sentence: str) -> str:
    s = sentence.lower()
    if re.search(r"\d", s):
        return "numeric"
    if any(w in s for w in ["caiu", "subiu", "aument", "redu", "domina", "tende", "baixa", "alta"]):
        return "directional"
    if any(w in s for w in ["causal", "causa", "prov", "implica", "portanto", "logo"]):
        return "causal_or_extrapolative"
    return "qualitative"


def evaluate_rule(rule: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[str, str]:
    key = rule.get("metric_key")
    if key not in metrics:
        return "not_verifiable", f"metric '{key}' not available"

    v = metrics.get(key)
    check = rule.get("check")

    if v is None:
        return "not_verifiable", "metric value is null"

    if check == "range":
        lo = float(rule.get("min"))
        hi = float(rule.get("max"))
        ok = (float(v) >= lo) and (float(v) <= hi)
        return ("supported" if ok else "unsupported", f"{key}={v} expected in [{lo}, {hi}]")
    if check == "lt":
        thr = float(rule.get("value"))
        ok = float(v) < thr
        return ("supported" if ok else "unsupported", f"{key}={v} expected < {thr}")
    if check == "lte":
        thr = float(rule.get("value"))
        ok = float(v) <= thr
        return ("supported" if ok else "unsupported", f"{key}={v} expected <= {thr}")
    if check == "gt":
        thr = float(rule.get("value"))
        ok = float(v) > thr
        return ("supported" if ok else "unsupported", f"{key}={v} expected > {thr}")
    if check == "gte":
        thr = float(rule.get("value"))
        ok = float(v) >= thr
        return ("supported" if ok else "unsupported", f"{key}={v} expected >= {thr}")
    if check == "eq":
        expected = rule.get("value")
        ok = v == expected
        return ("supported" if ok else "unsupported", f"{key}={v} expected == {expected}")

    return "not_verifiable", f"unknown check '{check}'"


def parse_args() -> argparse.Namespace:
    default_base = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
    parser = argparse.ArgumentParser(description="Audit non-survival narrative claims for support/overstatement.")
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--html-file", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--baseline-output-dir", type=Path, required=True)
    parser.add_argument("--spec-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    spec_file = (args.spec_file or (base_dir / "verification" / "spec" / "non_survival_manifest.yaml")).resolve()
    spec = load_manifest_spec(spec_file)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    html_raw = args.html_file.resolve().read_text(encoding="utf-8", errors="ignore")
    html = strip_survival_sections_from_html(html_raw)
    sections = extract_sections(html)

    truth_dir = args.truth_dir.resolve()
    truth_summary = read_json(truth_dir / "truth_summary_non_survival.json") if (truth_dir / "truth_summary_non_survival.json").exists() else {}
    truth_core = read_json(truth_dir / "truth_core_metrics.json") if (truth_dir / "truth_core_metrics.json").exists() else {}

    consolidated_path = args.baseline_output_dir.resolve() / "consolidated_status.json"
    consolidated = read_json(consolidated_path) if consolidated_path.exists() else {}

    metrics: Dict[str, Any] = {}
    metrics.update(truth_summary)
    metrics.update(truth_core)
    metrics["causal_claim_allowed"] = (
        consolidated.get("causal_diagnostic_assessment", {}).get("causal_claim_allowed")
    )

    claim_rules = spec.get("claim_rules", [])

    rows: List[Dict[str, Any]] = []
    claim_counter = 1

    for section_title, section_text in sections:
        candidates = select_claim_candidates(sentence_split(section_text), claim_rules=claim_rules)
        for sentence in candidates:
            sentence_l = sentence.lower()

            matched_rule = None
            for rule in claim_rules:
                pat = rule.get("match_regex", "")
                if pat and re.search(pat, sentence_l, flags=re.I):
                    matched_rule = rule
                    break

            if matched_rule is not None:
                support_status, note = evaluate_rule(matched_rule, metrics)
                evidence_ref = matched_rule.get("evidence_ref")
                claim_id = matched_rule.get("claim_id")
            else:
                support_status = "not_verifiable"
                note = "no explicit validation rule"
                evidence_ref = None
                claim_id = f"claim_{claim_counter:04d}"

            rows.append(
                {
                    "claim_id": claim_id,
                    "section": section_title,
                    "claim_text": sentence,
                    "claim_type": classify_claim(sentence),
                    "support_status": support_status,
                    "evidence_ref": evidence_ref,
                    "notes": note,
                }
            )
            claim_counter += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "claim_audit.csv", index=False)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "html_file": str(args.html_file.resolve()),
        "claims_total": int(len(df)),
        "supported": int((df["support_status"] == "supported").sum()) if not df.empty else 0,
        "unsupported": int((df["support_status"] == "unsupported").sum()) if not df.empty else 0,
        "not_verifiable": int((df["support_status"] == "not_verifiable").sum()) if not df.empty else 0,
        "by_type": df["claim_type"].value_counts().to_dict() if not df.empty else {},
    }
    write_json(out_dir / "claim_audit_summary.json", summary)
    print(str(out_dir / "claim_audit_summary.json"))


if __name__ == "__main__":
    main()
