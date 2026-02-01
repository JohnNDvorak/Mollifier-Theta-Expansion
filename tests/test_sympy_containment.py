"""Lint test: SymPy imports must be contained to allowed files (WI-6).

Per CLAUDE.md rule #7, SymPy is only allowed in:
  - core/scale_model.py (canonical location)
  - reports/ (rendering)
  - lemmas/di_kloosterman.py (DIExponentModel uses SymPy expressions internally)

All other files must go through ScaleModel for symbolic operations.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
PACKAGE_DIR = REPO_ROOT / "mollifier_theta"

# Files allowed to import sympy directly
ALLOWED_SYMPY_FILES = {
    PACKAGE_DIR / "core" / "scale_model.py",
    PACKAGE_DIR / "lemmas" / "di_kloosterman.py",
    # reports/ is allowed — rendering needs SymPy for display
    PACKAGE_DIR / "reports" / "render_md.py",
    PACKAGE_DIR / "reports" / "render_tex.py",
    PACKAGE_DIR / "reports" / "render_diagnose.py",
    PACKAGE_DIR / "reports" / "proof_certificate.py",
    PACKAGE_DIR / "reports" / "mathematica_export.py",
    # diagonal_extract.py uses sympy for polynomial evaluation (via lazy import)
    PACKAGE_DIR / "transforms" / "diagonal_extract.py",
}


def test_sympy_containment() -> None:
    """No `import sympy` outside allowed files."""
    violations: list[str] = []

    for py_file in PACKAGE_DIR.rglob("*.py"):
        if py_file in ALLOWED_SYMPY_FILES:
            continue

        text = py_file.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            if "import sympy" in stripped or "from sympy" in stripped:
                rel = py_file.relative_to(REPO_ROOT)
                violations.append(f"{rel}:{i}: {stripped}")

    if violations:
        msg = "SymPy imported outside allowed files:\n" + "\n".join(violations)
        raise AssertionError(msg)
