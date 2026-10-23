"""Pre-deploy check: all LWD imports resolve. Run: python3 tests/lwd_import_check.py"""
import ast, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
OUR_FILES = [
    "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_scheduler.py",
    "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py",
    "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py",
]
def collect_names(f):
    if not f.exists(): return set()
    tree = ast.parse(f.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)): names.add(node.name)
        if isinstance(node, ast.Assign):
            for a in node.targets:
                if isinstance(a, ast.Name): names.add(a.id)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name): names.add(node.target.id)
        if isinstance(node, ast.ImportFrom):
            for a in node.names: names.add(a.name)
        if isinstance(node, ast.Import):
            for a in node.names: names.add(a.asname or a.name.split(".")[0])
    return names
def resolve_module(mod):
    parts = mod.split(".")
    base = REPO / Path(*parts[:-1]); leaf = parts[-1]
    candidates = [base / (leaf + ".py"), base / leaf / "__init__.py", base / "__init__.py"]
    init = base / leaf / "__init__.py"
    if init.exists():
        tree = ast.parse(init.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                sub = (base / leaf / Path(*node.module.lstrip(".").split("."))).with_suffix(".py")
                if sub.exists(): candidates.append(sub)
    return [c for c in candidates if c.exists()]
failures = []
for rel in OUR_FILES:
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom): continue
        if not node.module or not node.module.startswith("vllm.v1.lwd"): continue
        targets = resolve_module(node.module)
        if not targets: continue
        available = set()
        for t in targets: available |= collect_names(t)
        for alias in node.names:
            if alias.name in ("*", "annotations"): continue
            if alias.name not in available:
                failures.append(f"{rel}:{node.lineno}: from {node.module} import {alias.name}")
if failures:
    print(f"IMPORT FAILURES: {len(failures)}")
    for f in failures: print(f"  {f}")
    sys.exit(1)
print("ALL LWD IMPORTS RESOLVE")
