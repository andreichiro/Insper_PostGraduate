#!/usr/bin/env python3
"""
Usage:
    python gen_requirements.py path/to/your_script.py  [--pin]

If --pin is given the script tries to look up the installed version
of each package and produces pinned specs (e.g. numpy==1.26.4).
Without --pin it emits an unpinned list.
"""
import ast
import sys
import builtins
from pathlib import Path

PIN = "--pin" in sys.argv
if PIN:
    sys.argv.remove("--pin")
try:
    import pkg_resources  # only needed when --pin is used
except ImportError:
    if PIN:
        sys.exit("Install setuptools (pkg_resources) or run without --pin")

if len(sys.argv) != 2:
    sys.exit("Give exactly ONE .py file path.\n  python gen_requirements.py script.py [--pin]")

target = Path(sys.argv[1])
if not (target.exists() and target.suffix == ".py"):
    sys.exit(f"{target} is not a Python file")

# ---------------- core logic -----------------
imports = set()

def add(name):
    root = name.split(".")[0]      # 'sklearn.metrics' -> 'sklearn'
    if root and root not in builtins.__dict__:
        imports.add(root)

tree = ast.parse(target.read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            add(alias.name)
    elif isinstance(node, ast.ImportFrom) and node.module:
        add(node.module)

# write requirements.txt
out_lines = []
for pkg in sorted(imports):
    if PIN:
        try:
            ver = pkg_resources.get_distribution(pkg).version
            out_lines.append(f"{pkg}=={ver}")
        except pkg_resources.DistributionNotFound:
            out_lines.append(pkg)          # package not installed locally
    else:
        out_lines.append(pkg)

Path("requirements.txt").write_text("\n".join(out_lines) + "\n")
print("Created requirements.txt with:")
print("\n".join(out_lines))