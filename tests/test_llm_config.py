"""Tests for Task 6: LLM model must be configurable via LLM_MODEL env var."""
import ast
import pathlib


def _direct_model_literals(filepath: str, fn_names: list[str]) -> list[str]:
    """Return model string literals passed directly to fn_names calls (not via os.getenv)."""
    src = pathlib.Path(filepath).read_text()
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match ChatOpenAI(model=...) or similar
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name not in fn_names:
            continue
        for kw in node.keywords:
            if kw.arg == "model" and isinstance(kw.value, ast.Constant):
                found.append(kw.value.value)
    return found


def test_app_py_no_hardcoded_model():
    """ChatOpenAI in app.py must not receive a literal model string."""
    direct = _direct_model_literals("backend/api/app.py", ["ChatOpenAI"])
    assert direct == [], \
        f"backend/api/app.py passes literal model to ChatOpenAI: {direct}. Use os.getenv('LLM_MODEL', 'gpt-4o')."


def test_pipeline_no_hardcoded_default_arg():
    """HybridRAGPipeline.__init__ must not default llm_model to a string literal."""
    src = pathlib.Path("rag/hybrid/pipeline.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for arg, default in zip(
                reversed(node.args.args),
                reversed(node.args.defaults),
            ):
                if arg.arg == "llm_model" and isinstance(default, ast.Constant) and isinstance(default.value, str):
                    assert False, \
                        f"HybridRAGPipeline.__init__ defaults llm_model to {default.value!r}. Use None and resolve via os.getenv."


def test_env_example_documents_llm_model():
    """LLM_MODEL must appear in .env.example."""
    env_example = pathlib.Path(".env.example").read_text()
    assert "LLM_MODEL" in env_example, \
        ".env.example must document LLM_MODEL= so operators know it's configurable."
