"""Tests for Task 15: sync asyncio.run() wrapper methods must be removed from pipelines."""
import ast
import pathlib


def _sync_asyncio_run_wrappers(path: str, class_name: str, expected_removed: set[str]) -> list[str]:
    """Find methods in class_name that call asyncio.run() and are in expected_removed."""
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in ast.walk(node):
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name not in expected_removed:
                continue
            # check if body contains asyncio.run(...)
            for child in ast.walk(item):
                if isinstance(child, ast.Call):
                    f = child.func
                    if isinstance(f, ast.Attribute) and f.attr == "run":
                        if isinstance(f.value, ast.Name) and f.value.id == "asyncio":
                            found.append(f"{class_name}.{item.name}")
    return found


def test_hybrid_pipeline_no_sync_wrappers():
    """HybridRAGPipeline must not have sync asyncio.run() wrappers for run/search/generate_answer."""
    remaining = _sync_asyncio_run_wrappers(
        "rag/hybrid/pipeline.py",
        "HybridRAGPipeline",
        {"run", "search", "generate_answer"},
    )
    assert not remaining, (
        f"HybridRAGPipeline still has sync asyncio.run() wrappers: {remaining}. "
        "Remove them and call asyncio.run() directly at the call site."
    )


def test_fusion_pipeline_no_sync_wrappers():
    """FusionRAGPipeline must not have sync asyncio.run() wrappers for search/generate_answer."""
    remaining = _sync_asyncio_run_wrappers(
        "rag/fusion/pipeline.py",
        "FusionRAGPipeline",
        {"search", "generate_answer"},
    )
    assert not remaining, (
        f"FusionRAGPipeline still has sync asyncio.run() wrappers: {remaining}. "
        "Remove them and call asyncio.run() directly at the call site."
    )
