"""
Shared test helpers for fuzzy.dashboard tests.
"""

from typing import Any, Set


def collect_component_ids(component: Any) -> Set[str]:
    """
    Recursively collect every Dash component id in a component tree.

    Returns:
        The set of component ids found.
    """
    component_ids = set()
    component_id = getattr(component, "id", None)
    if component_id is not None:
        component_ids.add(component_id)
    for child in getattr(component, "children", None) or []:
        if hasattr(child, "children") or hasattr(child, "id"):
            component_ids |= collect_component_ids(child)
    return component_ids
