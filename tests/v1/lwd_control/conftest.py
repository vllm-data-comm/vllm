"""lwd_control 单测装载垫片。

真实环境(Python >=3.10 且 torch 在)一切原生 import;无 torch 的
轻量环境(如本机 3.9)以 stub 顶替 vllm/__init__(其 env_override
依赖 torch),使纯逻辑模块(picker/ledger/inbox、包入口、公共面)
仍可测试。依赖 vllm.logger/envs 完整链的用例(test_lwd_config)在
此类环境模块级跳过,见各文件头部。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_PKG_DIR = REPO_ROOT / "vllm"

if importlib.util.find_spec("torch") is None and "vllm" not in sys.modules:
    stub = types.ModuleType("vllm")
    stub.__path__ = [str(VLLM_PKG_DIR)]
    sys.modules["vllm"] = stub
