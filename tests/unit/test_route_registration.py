"""The `/smart_segmentation` route in `src/main.py` must reach the tested code.

`src/main.py` builds an `sly.Api()` and instantiates the model at import time, so
it can not be imported offline. Its route registration is therefore pinned
statically: `test_smart_segmentation_route.py` exercises the handler body for
real, and these checks make sure the served route is that body and that the
initial figure is normalized by the adapter instead of the bitmap-only SDK
decoder.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"


def module(name):
    return ast.parse((SRC / name).read_text(), filename=name)


def routes(tree):
    """Handlers registered with `@server.post(<path>)`, by path."""
    registered = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "post"
                and len(decorator.args) == 1
                and isinstance(decorator.args[0], ast.Constant)
            ):
                registered[decorator.args[0].value] = node
    return registered


def calls(tree):
    """Every `<name>.<attr>(...)` call in the module, as "name.attr" strings."""
    return {
        f"{node.func.value.id}.{node.func.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
    }


def test_main_serves_the_tested_smart_segmentation_handler():
    handler = routes(module("main.py")).get("/smart_segmentation")

    assert handler is not None, "src/main.py must keep serving POST /smart_segmentation"
    assert [argument.arg for argument in handler.args.args] == ["response", "request"]
    # The whole handler is the delegation exercised by the route tests.
    assert len(handler.body) == 1
    statement = handler.body[0]
    assert isinstance(statement, ast.Return)
    assert isinstance(statement.value, ast.Call)
    assert ast.dump(statement.value.func) == ast.dump(
        ast.Attribute(value=ast.Name(id="smart_tool", ctx=ast.Load()), attr="smart_segmentation", ctx=ast.Load())
    )
    assert [argument.id for argument in statement.value.args] == ["self", "response", "request"]


def test_main_keeps_its_other_routes():
    assert set(routes(module("main.py"))) == {
        "/smart_segmentation",
        "/is_online",
        "/smart_segmentation_batched",
    }


@pytest.mark.parametrize("name", ["main.py", "smart_tool.py"])
def test_the_initial_figure_is_not_read_by_the_bitmap_only_sdk_decoder(name):
    module_calls = calls(module(name))

    assert "functional.download_init_mask" not in module_calls
    if name == "smart_tool.py":
        assert "geometry_adapter.download_init_mask" in module_calls
        # the full-image expansion of the normalized bitmap is still the SDK one
        assert "functional.bitmap_to_mask" in module_calls
