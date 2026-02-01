"""Deep-freeze utilities for immutable IR objects.

Converts mutable containers to immutable equivalents recursively:
  list -> FrozenList (list subclass that blocks mutation)
  dict -> FrozenDict (dict subclass that blocks mutation)

Using subclasses of the original types means Pydantic serialization
works without warnings. Callers can still pass plain lists and dicts
to Pydantic constructors — the model_validator coerces them.

Operations that would return mutable copies (__add__, slicing, copy(),
__or__) are overridden to return frozen types, preventing mutable leaks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class FrozenList(list):
    """A list subclass that raises TypeError on mutation.

    Pydantic sees it as a list (no serialization warnings), but
    all mutating operations are blocked. Copy-producing operations
    (__add__, slicing, copy) return FrozenList to prevent leaks.
    """

    def __setitem__(self, index: Any, value: Any) -> None:
        raise TypeError("FrozenList does not support item assignment")

    def __delitem__(self, index: Any) -> None:
        raise TypeError("FrozenList does not support item deletion")

    def append(self, value: Any) -> None:
        raise TypeError("FrozenList does not support append()")

    def extend(self, values: Any) -> None:
        raise TypeError("FrozenList does not support extend()")

    def insert(self, index: int, value: Any) -> None:
        raise TypeError("FrozenList does not support insert()")

    def pop(self, index: int = -1) -> Any:
        raise TypeError("FrozenList does not support pop()")

    def remove(self, value: Any) -> None:
        raise TypeError("FrozenList does not support remove()")

    def clear(self) -> None:
        raise TypeError("FrozenList does not support clear()")

    def sort(self, **kwargs: Any) -> None:
        raise TypeError("FrozenList does not support sort()")

    def reverse(self) -> None:
        raise TypeError("FrozenList does not support reverse()")

    def __iadd__(self, other: Any) -> Any:
        raise TypeError("FrozenList does not support += assignment")

    def __imul__(self, other: Any) -> Any:
        raise TypeError("FrozenList does not support *= assignment")

    def __add__(self, other: Any) -> "FrozenList":
        return FrozenList(list.__add__(self, other))

    def __radd__(self, other: Any) -> "FrozenList":
        return FrozenList(list.__add__(list(other), self))

    def __mul__(self, other: Any) -> "FrozenList":
        return FrozenList(list.__mul__(self, other))

    def __rmul__(self, other: Any) -> "FrozenList":
        return FrozenList(list.__mul__(self, other))

    def __getitem__(self, index: Any) -> Any:
        result = list.__getitem__(self, index)
        if isinstance(index, slice):
            return FrozenList(result)
        return result

    def copy(self) -> "FrozenList":
        return FrozenList(self)

    def __repr__(self) -> str:
        return f"FrozenList({super().__repr__()})"


class FrozenDict(dict):
    """A dict subclass that raises TypeError on mutation.

    Pydantic sees it as a dict (no serialization warnings), but
    all mutating operations are blocked. Copy-producing operations
    (__or__, copy) return FrozenDict to prevent leaks.
    """

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("FrozenDict does not support item assignment")

    def __delitem__(self, key: Any) -> None:
        raise TypeError("FrozenDict does not support item deletion")

    def clear(self) -> None:
        raise TypeError("FrozenDict does not support clear()")

    def pop(self, *args: Any) -> Any:
        raise TypeError("FrozenDict does not support pop()")

    def popitem(self) -> tuple:
        raise TypeError("FrozenDict does not support popitem()")

    def setdefault(self, key: Any, default: Any = None) -> Any:
        raise TypeError("FrozenDict does not support setdefault()")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict does not support update()")

    def __ior__(self, other: Any) -> Any:
        raise TypeError("FrozenDict does not support |= assignment")

    def __or__(self, other: Any) -> "FrozenDict":
        merged = dict(self)
        merged.update(other)
        return FrozenDict(merged)

    def __ror__(self, other: Any) -> "FrozenDict":
        merged = dict(other)
        merged.update(self)
        return FrozenDict(merged)

    def copy(self) -> "FrozenDict":
        return FrozenDict(dict.copy(self))

    @classmethod
    def fromkeys(cls, iterable: Any, value: Any = None) -> "FrozenDict":
        return cls(dict.fromkeys(iterable, value))

    def __hash__(self) -> int:  # type: ignore[override]
        # Use a robust hash that handles unhashable nested values
        try:
            return hash(tuple(sorted(self.items())))
        except TypeError:
            # Fallback for unhashable values (e.g., nested FrozenList)
            return hash(tuple(sorted((k, repr(v)) for k, v in self.items())))

    def __repr__(self) -> str:
        return f"FrozenDict({super().__repr__()})"


class DeepFreezeModel(BaseModel):
    """Mixin that deep-freezes all mutable containers after construction.

    Subclasses get automatic conversion of list -> FrozenList,
    dict -> FrozenDict on all fields after Pydantic validation.
    """

    @model_validator(mode="after")
    def _deep_freeze_containers(self) -> "DeepFreezeModel":
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            frozen = deep_freeze_for_pydantic(val)
            if frozen is not val:
                object.__setattr__(self, field_name, frozen)
        return self


def _is_already_frozen(obj: Any) -> bool:
    """Check if an object and all its children are already frozen."""
    if isinstance(obj, FrozenDict):
        return all(_is_already_frozen(v) for v in obj.values())
    if isinstance(obj, FrozenList):
        return all(_is_already_frozen(item) for item in obj)
    if isinstance(obj, (dict, list, set)):
        return False
    return True  # scalars, tuples, frozensets, enums, etc.


def deep_freeze_for_pydantic(obj: Any) -> Any:
    """Recursively convert mutable containers to frozen equivalents.

    Uses FrozenList / FrozenDict (subclasses of list / dict) so that
    Pydantic serialization sees the expected types with no warnings.
    Fast-paths already-frozen subtrees to avoid redundant work.
    """
    # Fast-path: already fully frozen
    if isinstance(obj, (FrozenDict, FrozenList)) and _is_already_frozen(obj):
        return obj
    if isinstance(obj, dict):
        return FrozenDict({k: deep_freeze_for_pydantic(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return FrozenList(deep_freeze_for_pydantic(item) for item in obj)
    if isinstance(obj, tuple):
        return tuple(deep_freeze_for_pydantic(item) for item in obj)
    if isinstance(obj, set):
        return frozenset(deep_freeze_for_pydantic(item) for item in obj)
    return obj
