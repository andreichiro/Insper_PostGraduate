#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick local tester for /teacher/dynamics endpoint.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of local API (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--request",
        required=False,
        help="Teacher free-text request.",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=50,
        help="Duration in minutes.",
    )
    parser.add_argument(
        "--top-k-sources",
        type=int,
        default=8,
        help="Number of retrieved sources.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model passed to API.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save full response JSON in analysis_output/tests.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive loop mode: type multiple requests without restarting script.",
    )
    return parser.parse_args()


def print_summary(payload: Dict[str, Any], response: Dict[str, Any]) -> None:
    print("\n=== REQUEST ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    print("\n=== RESPONSE SUMMARY ===")
    print("plan_title:", response.get("plan_title"))
    print("objective:", response.get("objective"))

    inferred = response.get("inferred_context") or {}
    print("inferred_context:", inferred)

    verifier = response.get("verifier") or {}
    print("verifier:", verifier)

    steps = response.get("steps") or []
    print("steps_count:", len(steps))
    if steps:
        print("first_step:", json.dumps(steps[0], ensure_ascii=False, indent=2))

    citations = response.get("citations") or []
    print("citations_count:", len(citations))
    for c in citations[:5]:
        print("-", c.get("source_id"), "|", c.get("title"), "|", c.get("component"), "| year", c.get("year"))


def maybe_save_json(response: Dict[str, Any]) -> None:
    out_dir = Path('/Users/akatsurada/Documents/INSPER/Visualization/Aula 2/analysis_output/tests')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f"teacher_dynamics_response_{ts}.json"
    out_path.write_text(json.dumps(response, ensure_ascii=False, indent=2))
    print(f"\nSaved full response to: {out_path}")


def main() -> None:
    args = parse_args()
    endpoint = f"{args.base_url.rstrip('/')}/teacher/dynamics"

    def send_once(user_request: str) -> None:
        payload = {
            "request": user_request,
            "duration_minutes": args.duration_minutes,
            "top_k_sources": args.top_k_sources,
            "model": args.model,
        }

        try:
            r = requests.post(endpoint, json=payload, timeout=180)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return

        print("status_code:", r.status_code)
        try:
            data = r.json()
        except ValueError:
            print("Non-JSON response:")
            print(r.text[:2000])
            return

        if r.status_code >= 400:
            print("Error response:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return

        print_summary(payload, data)
        if args.save_json:
            maybe_save_json(data)

    if args.interactive:
        print("Interactive mode enabled. Type request text and press Enter. Empty line to exit.")
        while True:
            try:
                user_request = input("request> ").strip()
            except EOFError:
                break
            if not user_request:
                break
            send_once(user_request)
            print("\n" + "-" * 80 + "\n")
        return

    if not args.request:
        raise SystemExit("Either provide --request or use --interactive.")

    send_once(args.request)


if __name__ == "__main__":
    main()
