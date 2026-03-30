#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from common import utc_now_iso, write_json


def run_cmd(cmd: List[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    return proc.stdout.strip()


def parse_args() -> argparse.Namespace:
    default_base = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
    parser = argparse.ArgumentParser(description="Run full non-survival verification workflow and produce verdict.")
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--python-exec", type=str, default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing run directory from run_repro_non_survival.py")
    parser.add_argument("--skip-repro", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    python_exec = args.python_exec or str((base_dir / ".venv" / "bin" / "python").resolve())

    if args.skip_repro:
        if args.run_dir is None:
            raise ValueError("--skip-repro requires --run-dir pointing to an existing reproduction run.")
        run_dir = args.run_dir.resolve()
    else:
        run_cmd_list = [
            python_exec,
            str((base_dir / "verification" / "run_repro_non_survival.py").resolve()),
            "--base-dir",
            str(base_dir),
            "--python-exec",
            python_exec,
        ]
        if args.run_id:
            run_cmd_list.extend(["--run-id", args.run_id])

        run_dir_out = run_cmd(run_cmd_list, cwd=base_dir)
        run_dir = Path(run_dir_out.splitlines()[-1].strip()).resolve()

    run_meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    baseline_output_dir = Path(run_meta["baseline_output_dir"])
    repro_output_dir = Path(run_meta["repro_output_dir"])

    compare_dir = run_dir / "results" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python_exec,
            str((base_dir / "verification" / "compare_artifacts.py").resolve()),
            "--base-dir",
            str(base_dir),
            "--baseline-output-dir",
            str(baseline_output_dir),
            "--repro-output-dir",
            str(repro_output_dir),
            "--out-dir",
            str(compare_dir),
        ],
        cwd=base_dir,
    )

    truth_dir = run_dir / "results" / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python_exec,
            str((base_dir / "verification" / "recompute_truth.py").resolve()),
            "--base-dir",
            str(base_dir),
            "--baseline-output-dir",
            str(baseline_output_dir),
            "--out-dir",
            str(truth_dir),
        ],
        cwd=base_dir,
    )

    claims_dir = run_dir / "results" / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python_exec,
            str((base_dir / "verification" / "audit_claims.py").resolve()),
            "--base-dir",
            str(base_dir),
            "--html-file",
            str(baseline_output_dir / "reports" / "analise_inicial_dos_dados_interativa.html"),
            "--truth-dir",
            str(truth_dir),
            "--baseline-output-dir",
            str(baseline_output_dir),
            "--out-dir",
            str(claims_dir),
        ],
        cwd=base_dir,
    )

    compare_summary = json.loads((compare_dir / "artifact_diff_summary.json").read_text(encoding="utf-8"))
    truth_summary = json.loads((truth_dir / "truth_recompute_summary.json").read_text(encoding="utf-8"))
    claim_summary = json.loads((claims_dir / "claim_audit_summary.json").read_text(encoding="utf-8"))

    compare_df = pd.read_csv(compare_dir / "artifact_diff_summary.csv")
    claim_df = pd.read_csv(claims_dir / "claim_audit.csv")
    truth_diff_df = pd.read_csv(truth_dir / "truth_vs_baseline_diff.csv")

    critical_or_major = compare_df[
        (compare_df["status"] == "fail") & (compare_df["severity"].isin(["critical", "major"]))
    ]

    # Treat only real disagreements/gaps as blockers; informational rows like
    # "extra_truth_metric" should not fail the verdict.
    truth_blocking_statuses = {"mismatch", "missing_in_truth", "missing_in_baseline"}
    truth_mismatches = truth_diff_df[truth_diff_df["status"].isin(truth_blocking_statuses)]
    unsupported_claims = claim_df[claim_df["support_status"] == "unsupported"]
    unverifiable_claims = claim_df[claim_df["support_status"] == "not_verifiable"]

    passes_all = (
        len(critical_or_major) == 0
        and len(truth_mismatches) == 0
        and len(unsupported_claims) == 0
        and len(unverifiable_claims) == 0
    )

    verdict = {
        "generated_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "objective": "Validate non-survival analysis correctness, reproducibility, and evidence support.",
        "comparison": compare_summary,
        "truth_recompute": truth_summary,
        "claim_audit": claim_summary,
        "decision": {
            "is_100_percent_verified": bool(passes_all),
            "verdict_label": "100% verified" if passes_all else "not 100% verified",
            "blocking_reasons": {
                "artifact_failures_critical_or_major": int(len(critical_or_major)),
                "truth_mismatches": int(len(truth_mismatches)),
                "unsupported_claims": int(len(unsupported_claims)),
                "not_verifiable_claims": int(len(unverifiable_claims)),
            },
        },
    }

    write_json(run_dir / "results" / "verification_verdict.json", verdict)

    report_lines = [
        "# Non-Survival Verification Verdict",
        "",
        f"- Generated at (UTC): {verdict['generated_at_utc']}",
        f"- Verdict: **{verdict['decision']['verdict_label']}**",
        f"- Critical/Major artifact failures: {len(critical_or_major)}",
        f"- Truth mismatches: {len(truth_mismatches)}",
        f"- Unsupported claims: {len(unsupported_claims)}",
        f"- Not-verifiable claims: {len(unverifiable_claims)}",
        "",
        "## Paths",
        f"- Run dir: `{run_dir}`",
        f"- Artifact diff: `{compare_dir / 'artifact_diff_summary.csv'}`",
        f"- Truth diff: `{truth_dir / 'truth_vs_baseline_diff.csv'}`",
        f"- Claim audit: `{claims_dir / 'claim_audit.csv'}`",
        f"- Verdict JSON: `{run_dir / 'results' / 'verification_verdict.json'}`",
    ]
    (run_dir / "results" / "verification_verdict.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(str(run_dir / "results" / "verification_verdict.json"))


if __name__ == "__main__":
    main()
