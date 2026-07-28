"""Tests for Task 11: fusion param overrides must be in a named helper."""
import ast
import pathlib


def test_fusion_search_params_helper_exists():
    """FusionRAGPipeline must have a helper that builds fusion-specific search params."""
    src = pathlib.Path("rag/fusion/pipeline.py").read_text()
    tree = ast.parse(src)
    helper_names = {"_fusion_search_params", "_make_fusion_params", "_build_fusion_params"}
    found = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in helper_names
        for node in ast.walk(tree)
    )
    assert found, \
        "fusion/pipeline.py must have a helper function (_fusion_search_params or similar) " \
        "that encapsulates the inflated_top_k / threshold override logic"
