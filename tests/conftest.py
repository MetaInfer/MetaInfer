"""Shared pytest configuration for smoke tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--case",
        action="append",
        default=None,
        help="Run cases whose name contains given substring (can repeat)",
    )
    parser.addoption(
        "--list-cases",
        action="store_true",
        help="List all case names and exit",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # --list-cases: print names and stop
    if config.getoption("--list-cases", default=False):
        names = sorted(
            {item.callspec.params["case"]["name"] for item in items if hasattr(item, "callspec")}
        )
        print("\nAvailable test cases:")
        for n in names:
            print(f"  - {n}")
        config.option.collectonly = True
        return

    # --case filter: keep only matching items
    case_patterns = config.getoption("--case", default=None) or []
    if not case_patterns:
        return

    remaining = []
    for item in items:
        if not hasattr(item, "callspec"):
            continue
        case_name = item.callspec.params["case"]["name"]
        if any(pat in case_name for pat in case_patterns):
            remaining.append(item)
    if not remaining:
        print(f"\nNo cases matched patterns: {case_patterns}")
        print("Available cases:")
        for item in items:
            if hasattr(item, "callspec"):
                print(f"  - {item.callspec.params['case']['name']}")
    items[:] = remaining
