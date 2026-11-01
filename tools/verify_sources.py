"""Check extracted source files against the original notebook.

Two checks are made for every file listed in the notebook:

1. The file on disk must match the text stored in the notebook (FILES).
   A mismatch here is an error and always fails the run.
2. The file's SHA-256 must match the value the notebook declares (HASHES).
   The original notebook declares different values for two files, so this
   check reports a warning by default. Pass --strict to make it fail.

Standard library only.
"""

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "original_kaggle_notebook.ipynb"


def load_declarations() -> tuple[dict, dict]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join(nb["cells"][1]["source"])
    values = {}
    for node in ast.parse(code).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("FILES", "HASHES")):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values["FILES"], values["HASHES"]


def main(strict: bool) -> int:
    files, declared = load_declarations()
    errors, warnings = [], []

    for name, declared_hash in declared.items():
        path = ROOT / name
        if not path.exists():
            errors.append(f"Missing file: {name}")
            continue
        disk = path.read_bytes()
        if disk != files[name].encode("utf-8"):
            errors.append(f"File differs from the notebook text: {name}")
        if hashlib.sha256(disk).hexdigest() != declared_hash:
            warnings.append(f"SHA-256 differs from the value declared in the notebook: {name}")

    for line in warnings:
        print(f"WARNING: {line}")
    for line in errors:
        print(f"ERROR: {line}", file=sys.stderr)

    checked = len(declared)
    if errors or (strict and warnings):
        return 1
    print(f"Verified {checked} extracted source files against the notebook text "
          f"({len(warnings)} declared-hash warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main(strict="--strict" in sys.argv))
