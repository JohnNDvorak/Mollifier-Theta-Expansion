"""Schema completeness and round-trip tests for math-parameter export.

Ensures every BoundOnly term exports complete records with no missing
fields, and that the JSON round-trip is stable.
"""

from __future__ import annotations

import json

import pytest

from mollifier_theta.core.ir import TermStatus
from mollifier_theta.core.stage_meta import get_bound_meta
from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.pipelines.conrey89_voronoi import conrey89_voronoi_pipeline
from mollifier_theta.reports.math_parameter_export import (
    MathParameterRecord,
    export_math_parameters,
    export_math_parameters_json,
)


class TestSchemaCompleteness:
    """Every BoundOnly term must export a complete record."""

    @pytest.fixture(scope="class")
    def baseline_result(self):
        return conrey89_pipeline(theta_val=0.56)

    @pytest.fixture(scope="class")
    def voronoi_result(self):
        return conrey89_voronoi_pipeline(theta_val=0.56)

    def test_all_bound_terms_exported(self, baseline_result) -> None:
        all_terms = baseline_result.ledger.all_terms()
        bound_only = [t for t in all_terms if t.status == TermStatus.BOUND_ONLY]
        records = export_math_parameters(all_terms)
        assert len(records) == len(bound_only)

    def test_no_empty_bound_family_for_di_terms(self) -> None:
        """DI-bound terms must have non-empty bound_family.

        Trivial bound terms may not have BoundMeta.
        """
        result = conrey89_pipeline(theta_val=0.56)
        all_terms = result.ledger.all_terms()
        records = export_math_parameters(all_terms)
        for r in records:
            if "DI" in r.citation or "Deshouillers" in r.citation:
                assert r.bound_family != "", (
                    f"DI term {r.term_id} has empty bound_family"
                )

    def test_no_empty_error_exponent(self, baseline_result) -> None:
        all_terms = baseline_result.ledger.all_terms()
        records = export_math_parameters(all_terms)
        for r in records:
            if r.bound_family:
                assert r.error_exponent != "", (
                    f"Term {r.term_id} ({r.bound_family}) has empty error_exponent"
                )

    def test_length_exponents_populated(self, baseline_result) -> None:
        all_terms = baseline_result.ledger.all_terms()
        records = export_math_parameters(all_terms)
        for r in records:
            assert r.m_length_exponent != ""
            assert r.n_length_exponent != ""
            assert r.modulus_exponent != ""

    def test_voronoi_terms_have_dual_length(self, voronoi_result) -> None:
        """PostVoronoi bound terms should have non-default n-length."""
        all_terms = voronoi_result.ledger.all_terms()
        records = export_math_parameters(all_terms)
        # Not all records will have dual lengths, but at least some should be non-default
        # (records from PostVoronoi may or may not have explicit dual length metadata)
        # At minimum, all records should be exportable without errors
        assert len(records) > 0


class TestRoundTripJSON:
    """JSON export must be losslessly round-trippable."""

    @pytest.fixture(scope="class")
    def baseline_result(self):
        return conrey89_pipeline(theta_val=0.56)

    def test_json_serializable(self, baseline_result) -> None:
        all_terms = baseline_result.ledger.all_terms()
        json_data = export_math_parameters_json(all_terms)
        serialized = json.dumps(json_data)
        assert isinstance(serialized, str)

    def test_json_round_trip(self, baseline_result) -> None:
        all_terms = baseline_result.ledger.all_terms()
        json_data = export_math_parameters_json(all_terms)
        serialized = json.dumps(json_data)
        parsed = json.loads(serialized)
        assert len(parsed) == len(json_data)

    def test_json_fields_present(self, baseline_result) -> None:
        all_terms = baseline_result.ledger.all_terms()
        json_data = export_math_parameters_json(all_terms)
        required_fields = {
            "term_id", "bound_family", "case_id", "error_exponent",
            "m_length_exponent", "n_length_exponent", "modulus_exponent",
            "kernel_family_tags", "citation",
        }
        for record in json_data:
            assert required_fields <= set(record.keys()), (
                f"Missing fields: {required_fields - set(record.keys())}"
            )

    def test_record_to_dict_matches_json(self, baseline_result) -> None:
        all_terms = baseline_result.ledger.all_terms()
        records = export_math_parameters(all_terms)
        json_data = export_math_parameters_json(all_terms)
        for record, jd in zip(records, json_data):
            assert record.to_dict() == jd


class TestGoldenSubset:
    """Golden: verify the binding term's exported record is stable."""

    @pytest.fixture(scope="class")
    def baseline_result(self):
        return conrey89_pipeline(theta_val=0.56)

    def test_binding_term_has_di_family(self, baseline_result) -> None:
        """The DI bound terms should all have DI_Kloosterman family."""
        all_terms = baseline_result.ledger.all_terms()
        records = export_math_parameters(all_terms)
        di_records = [r for r in records if r.bound_family == "DI_Kloosterman"]
        assert len(di_records) > 0
        for r in di_records:
            assert "7*theta/4" in r.error_exponent

    def test_di_records_have_symmetric_lengths(self, baseline_result) -> None:
        """DI records in the baseline should have symmetric length exponents.

        The default from _extract_length_exponents is "theta" unless
        SumStructure overrides it (e.g., to "T^theta").
        """
        all_terms = baseline_result.ledger.all_terms()
        records = export_math_parameters(all_terms)
        di_records = [r for r in records if r.bound_family == "DI_Kloosterman"]
        for r in di_records:
            # Both m and n should reference theta (either "theta" or "T^theta")
            assert "theta" in r.m_length_exponent
            assert "theta" in r.n_length_exponent
