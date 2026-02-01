"""JSON import/export helpers."""

from __future__ import annotations

import json
from pathlib import Path

from mollifier_theta.core.ledger import TermLedger


def export_ledger(ledger: TermLedger, path: str | Path) -> None:
    """Write ledger to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ledger.to_json())


def import_ledger(path: str | Path) -> TermLedger:
    """Read ledger from a JSON file."""
    path = Path(path)
    return TermLedger.from_json(path.read_text())


def export_dict(data: dict, path: str | Path) -> None:
    """Write arbitrary dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def import_dict(path: str | Path) -> dict:
    """Read JSON file as dict."""
    path = Path(path)
    return json.loads(path.read_text())
