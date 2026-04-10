from __future__ import annotations

import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from jinja2 import Environment, FileSystemLoader, select_autoescape

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.orchestration.artifacts import ProjectPaths


def _run_python_module_script(script_path: Path, args: list[str]) -> None:
    cmd = ["python", str(script_path), *args]
    subprocess.run(cmd, check=True)


def _extract_title_head_and_body(html: str) -> tuple[str, str, str]:
    title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    head_match = re.search(r"<head[^>]*>(.*)</head>", html, flags=re.IGNORECASE | re.DOTALL)
    body_match = re.search(r"<body[^>]*>(.*)</body>", html, flags=re.IGNORECASE | re.DOTALL)
    title = title_match.group(1).strip() if title_match else "Targeted ML"
    head_html = head_match.group(1).strip() if head_match else ""
    body = body_match.group(1).strip() if body_match else html
    return title, head_html, body


def _render_shell(paths: ProjectPaths, title: str, head_html: str, body_html: str) -> str:
    env = Environment(
        loader=FileSystemLoader(str(paths.project_root / "targeted_ml" / "templates")),
        autoescape=select_autoescape(enabled_extensions=("html", "j2")),
    )
    template = env.get_template("report_shell.html.j2")
    return template.render(title=title, head_html=head_html, body_html=body_html)


def build_report(spec: AnalysisSpec, paths: ProjectPaths) -> Path:
    script = paths.project_root / "targeted_ml" / "runtime" / "html_report_engine.py"
    output_html = paths.reports_dir / spec.report.output_html_name
    with TemporaryDirectory(prefix="targeted-ml-report-", dir=paths.staging_dir) as temp_dir:
        compat_html = Path(temp_dir) / "compat_report.html"
        _run_python_module_script(
            script,
            [
                "--build-dir",
                str(paths.build_dir),
                "--output-html",
                str(compat_html),
            ],
        )
        compat_content = compat_html.read_text(encoding="utf-8")
        title, head_html, body_html = _extract_title_head_and_body(compat_content)
        output_html.write_text(_render_shell(paths, title=title, head_html=head_html, body_html=body_html), encoding="utf-8")
    return output_html
