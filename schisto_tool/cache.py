
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, overload

F = TypeVar("F", bound=Callable[..., Any])


def cache_data(func: F | None = None, **kwargs: Any):
    """Use Streamlit cache_data when Streamlit is available; otherwise no-op.

    This keeps computational modules importable in test/CLI contexts where
    Streamlit is not installed, while preserving caching inside the app.
    """
    try:
        import streamlit as st  # type: ignore
    except ModuleNotFoundError:
        if func is not None:
            return func
        return lambda f: f

    if func is not None:
        return st.cache_data(func, **kwargs)
    return st.cache_data(**kwargs)
