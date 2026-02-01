"""Tests for report generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from mollifier_theta.pipelines.conrey89 import conrey89_pipeline
from mollifier_theta.reports.mathematica_export import export_diagonal_main_term, format_main_term_wl
from mollifier_theta.reports.render_md import render_report
from mollifier_theta.reports.render_tex import render_tex_report


class TestMarkdownReport:
    def test_report_nonempty(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        md = render_report(result)
        assert len(md) > 100

    def test_report_contains_four_sevenths(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        md = render_report(result)
        assert "4/7" in md or "4\\over7" in md or "theta < 4/7" in md.replace(" ", "")

    def test_report_contains_citations(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        md = render_report(result)
        assert "Conrey" in md
        assert "Deshouillers" in md or "Iwaniec" in md

    def test_report_contains_sub_exponent_table(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        md = render_report(result)
        assert "DI bilinear saving" in md

    def test_report_contains_transform_chain(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        md = render_report(result)
        assert "ApproxFunctionalEq" in md


class TestLaTeXReport:
    def test_tex_report_nonempty(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        tex = render_tex_report(result)
        assert len(tex) > 100
        assert r"\documentclass" in tex

    def test_tex_report_has_math(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        tex = render_tex_report(result)
        assert r"\theta" in tex


class TestMathematicaExport:
    def test_export_to_file(self, tmp_path: Path) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        output = export_diagonal_main_term(result.ledger, tmp_path)
        assert output.exists()
        content = output.read_text()
        assert len(content) > 0

    def test_export_contains_mathematica_syntax(self, tmp_path: Path) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        output = export_diagonal_main_term(result.ledger, tmp_path)
        content = output.read_text()
        # Should contain Mathematica function definitions
        assert "theta_" in content or "Log[" in content or ":=" in content

    def test_format_main_term_wl(self) -> None:
        result = conrey89_pipeline(theta_val=0.56)
        from mollifier_theta.core.ir import TermKind, TermStatus

        main_terms = result.ledger.filter(
            kind=TermKind.DIAGONAL, status=TermStatus.MAIN_TERM
        )
        assert len(main_terms) > 0
        wl = format_main_term_wl(main_terms[0])
        assert "diagonal" in wl.lower() or "MainTerm" in wl or ":=" in wl
