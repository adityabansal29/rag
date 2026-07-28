"""Tests for Task 13: DynamoDB scan must use Limit to avoid full-table scans."""
import ast
import pathlib


def test_dynamo_scan_has_limit():
    """list_jobs() must pass Limit= to dynamo.scan() to cap unbounded table scans."""
    src = pathlib.Path("backend/api/routes.py").read_text()
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # look for .scan(...) calls
        if not (isinstance(func, ast.Attribute) and func.attr == "scan"):
            continue
        kwarg_keys = {kw.arg for kw in node.keywords}
        assert "Limit" in kwarg_keys, (
            "dynamo.scan() in routes.py must include Limit= to prevent full-table scans. "
            "Without it, every /jobs request scans the entire DynamoDB table."
        )
