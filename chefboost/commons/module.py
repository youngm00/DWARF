import sys
from types import ModuleType

# pylint: disable=no-else-return


def load_module(module_name: str) -> ModuleType:
    """
    Load python module with its name.

    Rule modules (e.g. the GA's shared ``model_test``) are rewritten on every
    fit, often several times within the same second. Both ``imp.load_module``
    and importlib's default loader may then serve a *stale* cached bytecode
    (.pyc mtime resolution is 1s), returning a previous fit's tree and causing
    str/float type mismatches at predict time. To be robust we resolve the
    source path, then read and ``exec`` the current source directly — no
    bytecode cache is consulted or written.
    """
    import os

    # chefboost passes a path-style name like "outputs/rules/model_test"
    # (see Training.buildDecisionTree / restoreTree docstring). Resolve it to the
    # .py file directly; fall back to importlib for dotted module names.
    source_path = module_name + ".py"
    if not os.path.exists(source_path):
        import importlib.util

        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise ImportError(f"Module '{module_name}' not found")
        source_path = spec.origin

    from chefboost.commons import rulebuffer

    rulebuffer.flush(source_path)

    with open(source_path, "r", encoding="UTF-8") as f:
        source = f.read()

    module = ModuleType(os.path.basename(module_name))
    module.__file__ = source_path
    exec(compile(source, source_path, "exec"), module.__dict__)  # pylint: disable=exec-used
    return module
