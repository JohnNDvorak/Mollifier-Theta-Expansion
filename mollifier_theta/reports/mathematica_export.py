"""Export diagonal main term to Mathematica .wl format."""

from __future__ import annotations

from pathlib import Path

from mollifier_theta.core.ir import Term, TermKind, TermStatus
from mollifier_theta.core.ledger import TermLedger


def format_main_term_wl(term: Term) -> str:
    """Format a diagonal main term as Mathematica code."""
    poly_data = term.metadata.get("main_term_poly", {})
    coefficients = poly_data.get("coefficients", [])

    lines = [
        "(* Diagonal main term from mollifier-theta framework *)",
        "(* Generated automatically — do not edit *)",
        "",
        f'(* {poly_data.get("description", "Diagonal main term")} *)',
        "",
        "(* Mollifier coefficients as function of theta *)",
    ]

    # Build the polynomial expression
    terms_wl = []
    for label, expr in coefficients:
        # Convert Python/SymPy syntax to Mathematica
        wl_expr = expr.replace("**", "^")
        terms_wl.append(f"  (* {label} *) {wl_expr}")

    lines.append(f"diagonalMainTermPoly[theta_] := {' + '.join(e.split('*)')[1].strip() for e in terms_wl)}")
    lines.append("")
    lines.append("(* Full main term: T * P(theta) * Log[T]^k *)")

    log_power = term.metadata.get("log_power", 0)
    lines.append(
        f"diagonalMainTerm[T_, theta_] := T * diagonalMainTermPoly[theta] * Log[T]^{log_power}"
    )
    lines.append("")
    lines.append("(* Mellin/residue integral representation *)")
    lines.append("(* The main term arises from the residue at s=1 of the Mellin transform *)")
    lines.append("(* of the AFE kernel convolved with the mollifier squared *)")
    lines.append(
        "mellinIntegralRep[s_, T_, theta_] := "
        "1/(2 Pi I) Integrate["
        "  T^s/s * zetaMollifierSquaredMellin[s, theta],"
        "  {s, 1/2 - I Infinity, 1/2 + I Infinity}"
        "]"
    )
    lines.append("")

    return "\n".join(lines)


def export_diagonal_main_term(
    ledger: TermLedger | None = None,
    output_dir: str | Path = "artifacts/mathematica",
) -> Path:
    """Export diagonal main term from ledger to .wl file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diagonal_main_term.wl"

    if ledger is None:
        # Try loading from default artifact location
        from mollifier_theta.core.serialize import import_ledger

        ledger_path = Path("artifacts/repro_conrey89/ledger.json")
        if ledger_path.exists():
            ledger = import_ledger(ledger_path)
        else:
            # Generate a template
            content = "(* No ledger found — run 'mollifier repro conrey89' first *)\n"
            output_path.write_text(content)
            return output_path

    main_terms = ledger.filter(
        kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM
    )

    if not main_terms:
        output_path.write_text("(* No diagonal main terms in ledger *)\n")
        return output_path

    parts = []
    for mt in main_terms:
        parts.append(format_main_term_wl(mt))

    output_path.write_text("\n\n".join(parts))
    return output_path
