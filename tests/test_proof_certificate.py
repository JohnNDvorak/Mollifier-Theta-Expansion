"""Tests for proof certificate exporter (includes WI-8 hardening)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mollifier_theta.analysis.slack import diagnose_pipeline
from mollifier_theta.core.ir import Term, TermKind, TermStatus
from mollifier_theta.core.ledger import TermLedger
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.reports.proof_certificate import (
    _environment_stamp,
    export_proof_certificate,
    generate_proof_certificate,
    render_proof_certificate_md,
)


class TestProofCertificate:
    def test_certificate_generated(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        assert isinstance(cert, dict)

    def test_certificate_has_required_keys(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        required = {
            "theta_val", "theta_admissible", "theta_max", "headroom",
            "transform_chain", "term_counts", "constraints",
            "binding_constraint", "verification",
        }
        assert required <= set(cert.keys())

    def test_transform_chain_nonempty(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        assert len(cert["transform_chain"]) > 0

    def test_constraints_nonempty(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        assert len(cert["constraints"]) > 0

    def test_binding_constraint_has_derivation_path(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        bc = cert["binding_constraint"]
        assert bc is not None
        assert "derivation_path" in bc
        assert len(bc["derivation_path"]) > 0

    def test_binding_constraint_traceable_to_initial(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        bc = cert["binding_constraint"]
        # The first step should be an early transform
        path = bc["derivation_path"]
        transforms = [step["transform"] for step in path]
        # Should include transforms from the full chain
        assert len(transforms) >= 3

    def test_term_counts_consistent(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        total = cert["term_counts"]["total"]
        by_status_total = sum(cert["term_counts"]["by_status"].values())
        assert total == by_status_total

    def test_verification_fields(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        v = cert["verification"]
        assert v["theta_max_symbolic"] is not None
        assert v["theta_max_numerical"] is not None
        assert v["is_supremum"] is True

    def test_json_serializable(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        text = json.dumps(cert, default=str)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)


class TestProofCertificateMarkdown:
    def test_markdown_nonempty(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        md = render_proof_certificate_md(cert)
        assert len(md) > 100

    def test_markdown_contains_theta(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        md = render_proof_certificate_md(cert)
        assert "0.56" in md

    def test_markdown_contains_binding(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        md = render_proof_certificate_md(cert)
        assert "Binding Constraint" in md


class TestProofCertificateExport:
    def test_export_creates_files(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "proof"
            export_proof_certificate(result, output_dir)
            assert (output_dir / "proof_certificate.json").exists()
            assert (output_dir / "proof_certificate.md").exists()


# ============================================================
# WI-8: Reproducible Proof Certificates
# ============================================================
class TestDeterministicTieBreak:
    def test_slack_ordering_deterministic(self) -> None:
        """Two runs produce identical slack ordering."""
        r1 = diagnose_pipeline(theta_val=0.56)
        r2 = diagnose_pipeline(theta_val=0.56)
        keys1 = [(ts.slack, ts.error_exponent) for ts in r1.term_slacks]
        keys2 = [(ts.slack, ts.error_exponent) for ts in r2.term_slacks]
        assert keys1 == keys2

    def test_bottleneck_deterministic(self) -> None:
        r1 = diagnose_pipeline(theta_val=0.56)
        r2 = diagnose_pipeline(theta_val=0.56)
        assert r1.bottleneck is not None
        assert r2.bottleneck is not None
        assert r1.bottleneck.error_exponent == r2.bottleneck.error_exponent
        assert r1.bottleneck.bound_family == r2.bottleneck.bound_family


class TestEnvironmentStamp:
    def test_has_python_version(self) -> None:
        stamp = _environment_stamp()
        assert "python_version" in stamp

    def test_has_platform(self) -> None:
        stamp = _environment_stamp()
        assert "platform" in stamp

    def test_has_pydantic_version(self) -> None:
        stamp = _environment_stamp()
        assert "pydantic_version" in stamp
        assert stamp["pydantic_version"] != "unknown"

    def test_has_sympy_version(self) -> None:
        stamp = _environment_stamp()
        assert "sympy_version" in stamp
        assert stamp["sympy_version"] != "unknown"

    def test_certificate_has_environment(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        assert "environment" in cert
        assert "python_version" in cert["environment"]


class TestCanonicalJSON:
    def test_sort_keys(self) -> None:
        """Exported JSON has sorted keys."""
        result = conrey89_pipeline(theta_val=0.56)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "proof"
            export_proof_certificate(result, output_dir)
            text = (output_dir / "proof_certificate.json").read_text()
            data = json.loads(text)
            canonical = json.dumps(data, indent=2, sort_keys=True, default=str)
            assert text == canonical

    def test_constraint_ordering_canonical(self) -> None:
        r1 = conrey89_pipeline(theta_val=0.56)
        r2 = conrey89_pipeline(theta_val=0.56)
        c1 = generate_proof_certificate(r1)
        c2 = generate_proof_certificate(r2)
        keys1 = [(c["slack"], c["error_exponent"]) for c in c1["constraints"]]
        keys2 = [(c["slack"], c["error_exponent"]) for c in c2["constraints"]]
        assert keys1 == keys2


class TestContentFingerprint:
    def test_fingerprint_present(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        cert = generate_proof_certificate(result)
        assert "content_fingerprint" in cert
        assert len(cert["content_fingerprint"]) == 16

    def test_fingerprint_stable_across_runs(self) -> None:
        """Two pipeline runs produce the same content fingerprint."""
        r1 = conrey89_pipeline(theta_val=0.56)
        r2 = conrey89_pipeline(theta_val=0.56)
        c1 = generate_proof_certificate(r1)
        c2 = generate_proof_certificate(r2)
        assert c1["content_fingerprint"] == c2["content_fingerprint"]

    def test_fingerprint_differs_for_different_theta(self) -> None:
        r1 = conrey89_pipeline(theta_val=0.56)
        r2 = conrey89_pipeline(theta_val=0.50)
        c1 = generate_proof_certificate(r1)
        c2 = generate_proof_certificate(r2)
        assert c1["content_fingerprint"] != c2["content_fingerprint"]


class TestTermLedgerPrune:
    def test_prune_removes_intermediates(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        ledger = result.ledger
        initial_count = ledger.count()
        removed = ledger.prune()
        assert removed > 0
        assert ledger.count() < initial_count

    def test_prune_keeps_bound_only(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        ledger = result.ledger
        bound_before = len(ledger.filter(status=TermStatus.BOUND_ONLY))
        ledger.prune()
        bound_after = len(ledger.filter(status=TermStatus.BOUND_ONLY))
        assert bound_after == bound_before

    def test_prune_keeps_main_terms(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        ledger = result.ledger
        main_before = len(ledger.filter(status=TermStatus.MAIN_TERM))
        ledger.prune()
        main_after = len(ledger.filter(status=TermStatus.MAIN_TERM))
        assert main_after == main_before

    def test_prune_custom_statuses(self) -> None:
        ledger = TermLedger()
        ledger.add(Term(kind=TermKind.INTEGRAL))
        ledger.add(Term(
            kind=TermKind.KLOOSTERMAN,
            status=TermStatus.BOUND_ONLY,
            lemma_citation="test",
        ))
        removed = ledger.prune(keep_statuses={TermStatus.BOUND_ONLY})
        assert removed == 1
        assert ledger.count() == 1

    def test_prune_returns_count(self) -> None:
        ledger = TermLedger()
        ledger.add(Term(kind=TermKind.INTEGRAL))
        ledger.add(Term(kind=TermKind.DIAGONAL))
        removed = ledger.prune()
        assert removed == 2
        assert ledger.count() == 0
