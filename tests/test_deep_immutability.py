"""Tests for deep immutability enforcement (WI-1).

Verifies that nested containers in IR objects cannot be mutated in-place.
"""

from __future__ import annotations

import pytest

from mollifier_theta.core.frozen_collections import (
    FrozenDict,
    FrozenList,
    deep_freeze_for_pydantic,
)
from mollifier_theta.core.ir import (
    HistoryEntry,
    Kernel,
    Phase,
    Range,
    Term,
    TermKind,
)
from mollifier_theta.core.sum_structures import (
    AdditiveTwist,
    CoeffSeq,
    SumIndex,
    SumStructure,
    WeightKernel,
)


class TestFrozenList:
    def test_cannot_append(self) -> None:
        fl = FrozenList([1, 2, 3])
        with pytest.raises(TypeError, match="append"):
            fl.append(4)

    def test_cannot_setitem(self) -> None:
        fl = FrozenList([1, 2, 3])
        with pytest.raises(TypeError, match="item assignment"):
            fl[0] = 99

    def test_cannot_extend(self) -> None:
        fl = FrozenList([1, 2, 3])
        with pytest.raises(TypeError, match="extend"):
            fl.extend([4, 5])

    def test_cannot_insert(self) -> None:
        fl = FrozenList([1, 2, 3])
        with pytest.raises(TypeError, match="insert"):
            fl.insert(0, 99)

    def test_cannot_pop(self) -> None:
        fl = FrozenList([1, 2, 3])
        with pytest.raises(TypeError, match="pop"):
            fl.pop()

    def test_equality_with_list(self) -> None:
        fl = FrozenList([1, 2, 3])
        assert fl == [1, 2, 3]

    def test_iteration(self) -> None:
        fl = FrozenList([1, 2, 3])
        assert list(fl) == [1, 2, 3]


class TestFrozenDict:
    def test_cannot_setitem(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="item assignment"):
            fd["b"] = 2

    def test_cannot_delitem(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="item deletion"):
            del fd["a"]

    def test_cannot_update(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="update"):
            fd.update({"b": 2})

    def test_cannot_pop(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="pop"):
            fd.pop("a")

    def test_equality_with_dict(self) -> None:
        fd = FrozenDict({"a": 1, "b": 2})
        assert fd == {"a": 1, "b": 2}

    def test_read_access_works(self) -> None:
        fd = FrozenDict({"a": 1})
        assert fd["a"] == 1
        assert fd.get("a") == 1
        assert fd.get("missing") is None


class TestDeepFreeze:
    def test_deep_freeze_for_pydantic(self) -> None:
        result = deep_freeze_for_pydantic([{"key": [1, 2]}])
        assert isinstance(result, FrozenList)
        assert isinstance(result[0], FrozenDict)
        assert isinstance(result[0]["key"], FrozenList)

    def test_deep_freeze_nested_sets(self) -> None:
        result = deep_freeze_for_pydantic({"s": {1, 2, 3}})
        assert isinstance(result, FrozenDict)
        assert isinstance(result["s"], frozenset)

    def test_deep_freeze_idempotent(self) -> None:
        """Already-frozen objects are returned as-is (fast path)."""
        original = deep_freeze_for_pydantic([{"a": 1}, {"b": [2, 3]}])
        refrozen = deep_freeze_for_pydantic(original)
        assert refrozen is original


class TestTermDeepImmutability:
    def test_metadata_cannot_be_mutated(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            metadata={"foo": "bar"},
        )
        with pytest.raises(TypeError, match="item assignment"):
            term.metadata["new_key"] = "value"

    def test_metadata_nested_dict_immutable(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            metadata={"nested": {"inner": "value"}},
        )
        with pytest.raises(TypeError, match="item assignment"):
            term.metadata["nested"]["new_key"] = "value"

    def test_variables_cannot_be_mutated(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            variables=["m", "n"],
        )
        with pytest.raises(TypeError, match="append"):
            term.variables.append("x")

    def test_phases_cannot_be_mutated(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            phases=[Phase(expression="e(x)")],
        )
        with pytest.raises(TypeError, match="append"):
            term.phases.append(Phase(expression="e(y)"))

    def test_kernels_cannot_be_mutated(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            kernels=[Kernel(name="W")],
        )
        with pytest.raises(TypeError, match="item assignment"):
            term.kernels[0] = Kernel(name="X")

    def test_history_cannot_be_mutated(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            history=[HistoryEntry(transform="T1")],
        )
        with pytest.raises(TypeError, match="append"):
            term.history.append(HistoryEntry(transform="T2"))

    def test_parents_cannot_be_mutated(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            parents=["id1"],
        )
        with pytest.raises(TypeError, match="append"):
            term.parents.append("id2")

    def test_ranges_cannot_be_mutated(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            ranges=[Range(variable="t")],
        )
        with pytest.raises(TypeError, match="append"):
            term.ranges.append(Range(variable="x"))


class TestKernelDeepImmutability:
    def test_properties_immutable(self) -> None:
        k = Kernel(name="W", properties={"smooth": True})
        with pytest.raises(TypeError, match="item assignment"):
            k.properties["new_prop"] = "value"

    def test_nested_properties_immutable(self) -> None:
        k = Kernel(name="W", properties={"config": {"inner": 1}})
        with pytest.raises(TypeError, match="item assignment"):
            k.properties["config"]["new_key"] = 2


class TestPhaseDeepImmutability:
    def test_depends_on_immutable(self) -> None:
        p = Phase(expression="e(x)", depends_on=["m", "n"])
        with pytest.raises(TypeError, match="append"):
            p.depends_on.append("c")


class TestJsonRoundTrip:
    def test_term_roundtrip_preserves_immutability(self) -> None:
        term = Term(
            kind=TermKind.INTEGRAL,
            variables=["m", "n"],
            metadata={"key": "value", "nested": {"inner": [1, 2]}},
            phases=[Phase(expression="e(x)", depends_on=["m"])],
        )
        data = term.model_dump()
        reconstructed = Term(**data)

        # Reconstructed term is also deeply immutable
        with pytest.raises(TypeError, match="append"):
            reconstructed.variables.append("x")
        with pytest.raises(TypeError, match="item assignment"):
            reconstructed.metadata["new"] = "val"
        with pytest.raises(TypeError, match="append"):
            reconstructed.phases.append(Phase(expression="e(y)"))

    def test_callers_can_pass_plain_lists(self) -> None:
        """Pydantic coercion still works — callers don't need to use FrozenList."""
        term = Term(
            kind=TermKind.INTEGRAL,
            variables=["a", "b"],
            metadata={"x": 1},
        )
        assert term.variables == ["a", "b"]
        assert term.metadata == {"x": 1}
        # But stored as frozen
        assert isinstance(term.variables, FrozenList)
        assert isinstance(term.metadata, FrozenDict)


class TestSumStructureDeepImmutability:
    def test_sum_indices_immutable(self) -> None:
        ss = SumStructure(
            sum_indices=[SumIndex(name="m"), SumIndex(name="n")],
        )
        with pytest.raises(TypeError, match="append"):
            ss.sum_indices.append(SumIndex(name="c"))

    def test_additive_twists_immutable(self) -> None:
        ss = SumStructure(
            additive_twists=[AdditiveTwist(modulus="c", numerator="a", sum_variable="m")],
        )
        with pytest.raises(TypeError, match="append"):
            ss.additive_twists.append(
                AdditiveTwist(modulus="c", numerator="b", sum_variable="n")
            )

    def test_coeff_seq_citations_immutable(self) -> None:
        cs = CoeffSeq(
            name="a_m", variable="m",
            citations=["Ref1", "Ref2"],
        )
        with pytest.raises(TypeError, match="append"):
            cs.citations.append("Ref3")

    def test_weight_kernel_parameters_immutable(self) -> None:
        wk = WeightKernel(
            kind="smooth",
            parameters={"type": "bessel"},
        )
        with pytest.raises(TypeError, match="item assignment"):
            wk.parameters["new_param"] = "value"


class TestFrozenListCopyLeaks:
    """Verify FrozenList copy-producing operations return FrozenList, not list."""

    def test_add_returns_frozen(self) -> None:
        fl = FrozenList([1, 2])
        result = fl + [3, 4]
        assert isinstance(result, FrozenList)
        assert result == [1, 2, 3, 4]

    def test_radd_returns_frozen(self) -> None:
        fl = FrozenList([3, 4])
        result = [1, 2] + fl
        assert isinstance(result, FrozenList)
        assert result == [1, 2, 3, 4]

    def test_mul_returns_frozen(self) -> None:
        fl = FrozenList([1, 2])
        result = fl * 2
        assert isinstance(result, FrozenList)
        assert result == [1, 2, 1, 2]

    def test_rmul_returns_frozen(self) -> None:
        fl = FrozenList([1, 2])
        result = 2 * fl
        assert isinstance(result, FrozenList)
        assert result == [1, 2, 1, 2]

    def test_slice_returns_frozen(self) -> None:
        fl = FrozenList([1, 2, 3, 4])
        result = fl[1:3]
        assert isinstance(result, FrozenList)
        assert result == [2, 3]

    def test_copy_returns_frozen(self) -> None:
        fl = FrozenList([1, 2, 3])
        result = fl.copy()
        assert isinstance(result, FrozenList)
        assert result == [1, 2, 3]
        # Mutation of copy should also fail
        with pytest.raises(TypeError):
            result.append(4)

    def test_iadd_blocked(self) -> None:
        fl = FrozenList([1, 2])
        with pytest.raises(TypeError, match="\\+="):
            fl += [3]

    def test_imul_blocked(self) -> None:
        fl = FrozenList([1, 2])
        with pytest.raises(TypeError, match="\\*="):
            fl *= 2

    def test_sort_blocked(self) -> None:
        fl = FrozenList([3, 1, 2])
        with pytest.raises(TypeError, match="sort"):
            fl.sort()

    def test_reverse_blocked(self) -> None:
        fl = FrozenList([1, 2, 3])
        with pytest.raises(TypeError, match="reverse"):
            fl.reverse()


class TestFrozenDictCopyLeaks:
    """Verify FrozenDict copy-producing operations return FrozenDict, not dict."""

    def test_or_returns_frozen(self) -> None:
        fd = FrozenDict({"a": 1})
        result = fd | {"b": 2}
        assert isinstance(result, FrozenDict)
        assert result == {"a": 1, "b": 2}

    def test_ror_returns_frozen(self) -> None:
        fd = FrozenDict({"b": 2})
        result = {"a": 1} | fd
        assert isinstance(result, FrozenDict)
        assert result == {"a": 1, "b": 2}

    def test_copy_returns_frozen(self) -> None:
        fd = FrozenDict({"a": 1, "b": 2})
        result = fd.copy()
        assert isinstance(result, FrozenDict)
        assert result == {"a": 1, "b": 2}
        with pytest.raises(TypeError):
            result["c"] = 3

    def test_fromkeys_returns_frozen(self) -> None:
        result = FrozenDict.fromkeys(["a", "b", "c"], 0)
        assert isinstance(result, FrozenDict)
        assert result == {"a": 0, "b": 0, "c": 0}
        with pytest.raises(TypeError):
            result["d"] = 1

    def test_ior_blocked(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="\\|="):
            fd |= {"b": 2}

    def test_hash_works(self) -> None:
        fd = FrozenDict({"a": 1, "b": 2})
        h = hash(fd)
        assert isinstance(h, int)

    def test_hash_with_nested_frozen_list(self) -> None:
        """Hash should work even with unhashable nested values."""
        fd = FrozenDict({"a": FrozenList([1, 2, 3])})
        h = hash(fd)
        assert isinstance(h, int)

    def test_setdefault_blocked(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="setdefault"):
            fd.setdefault("b", 2)

    def test_popitem_blocked(self) -> None:
        fd = FrozenDict({"a": 1})
        with pytest.raises(TypeError, match="popitem"):
            fd.popitem()
