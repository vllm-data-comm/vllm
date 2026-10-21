# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""lwd_aivec_analyzer 表驱动测试(2026-10-20 aivec 排查配套工具)。

两轨可跑:真 pytest(torch 环境)走 parametrize;轻量环境(仓级 conftest
被 import torch 挡住时)手工跑器直接调 _check_* 与模块级 CASES 表。
覆盖:四态判定矩阵、编码自适应硬性要求(带/无 BOM 的 UTF-16、GBK、
CRLF 混行尾、--encoding 强制)、乱码永不崩溃、JSON 输出。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "log_analyze_tools" / "lwd_aivec_analyzer.py"

# py3.9 的 dataclass 字段处理查 sys.modules,spec 装载必须先注册
_spec = importlib.util.spec_from_file_location("lwd_aivec_analyzer", TOOL)
analyzer = importlib.util.module_from_spec(_spec)
sys.modules["lwd_aivec_analyzer"] = analyzer
_spec.loader.exec_module(analyzer)


def write_log(lines: list[str], encoding: str, newline: str = "\n") -> Path:
    tmp = tempfile.NamedTemporaryFile(
        "wb", suffix=".log", delete=False, dir=str(Path(tempfile.gettempdir()))
    )
    try:
        payload = newline.join(lines) + newline
        tmp.write(payload.encode(encoding))
    finally:
        tmp.close()
    return Path(tmp.name)


# --------------------------------------------------------------------- #
# 编码自适应矩阵(硬性要求:BOM 判定、无 BOM NUL 启发式、GBK、混行尾)    #
# --------------------------------------------------------------------- #
@dataclass
class EncCase:
    name: str
    encoding: str
    newline: str = "\n"

ENCODING_CASES = [
    EncCase("utf8", "utf-8"),
    EncCase("utf8_crlf_mixed", "utf-8", newline="\r\n"),
    EncCase("utf16_bom", "utf-16"),            # PowerShell > 重定向默认形态
    EncCase("utf16le_no_bom", "utf-16-le"),    # NUL 启发式路径
    EncCase("gbk", "gbk"),
    EncCase("utf8_sig", "utf-8-sig"),
]

PROBE_LINES = [
    "[lwd][LWD][sampler-probe] bincount launch shape=(16, 2048) "
    "stride=(2048, 1) ptr=0x7f1200 vocab=151936 num_seqs=16",
    "[lwd][LWD][spec-probe] POISON RESIDENT req=abc-1 idx=0 prompt_-1=2 "
    "output_-1=0 prompt_len=1561 real=1561 start=0 num_spec=3",
    "普通噪声行:中文内容应当无损往返(GBK 场景)",
]


def _check_encoding(case: EncCase) -> None:
    path = write_log(PROBE_LINES, case.encoding, case.newline)
    try:
        rep = analyzer.analyze([path])
        assert rep.bincount_launches.count == 1, case.name
        assert len(rep.poison_events) == 1, case.name
        assert rep.poison_events[0]["prompt_-1"] == 2, case.name
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("case", ENCODING_CASES, ids=[c.name for c in ENCODING_CASES])
def test_encoding_matrix(case: EncCase) -> None:
    _check_encoding(case)


def test_forced_encoding_overrides_detection() -> None:
    """--encoding 强制指定优先于启发式(硬性要求第 5 条)。"""
    path = write_log(PROBE_LINES, "utf-16-le")
    try:
        assert analyzer.iter_lines(path, "utf-16-le")[1].startswith("[lwd]")
        rep = analyzer.analyze([path], "utf-16-le")
        assert rep.bincount_launches.count == 1
    finally:
        path.unlink(missing_ok=True)


def test_never_crash_on_garbage() -> None:
    """乱码/截断 BOM 不崩:解码问题只降级。"""
    with tempfile.NamedTemporaryFile("wb", suffix=".log", delete=False) as fh:
        fh.write(b"\xff\xfe\x00\x12\x34garbage\x00\xff\xfe\xff")
        path = Path(fh.name)
    try:
        rep = analyzer.analyze([path])  # 不得 raise
        assert rep.bincount_launches.count == 0
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------- #
# 四态判定矩阵(每案:日志行 -> 期望判定 + 关键计数)                      #
# --------------------------------------------------------------------- #
@dataclass
class VerdictCase:
    name: str
    lines: list[str] = field(default_factory=list)
    verdict: str = ""
    checks: dict = field(default_factory=dict)

ACTIVE = "[lwd][LWD][sampler-probe] penalties ACTIVE logits=(16, 151936) prompt_ids=(16, 1561)"
LAUNCH = ("[lwd][LWD][sampler-probe] bincount launch shape=(16, 2048) "
          "stride=(2048, 1) ptr=0x7f1200 vocab=151936 num_seqs=16")
POISON_PROMPT = ("[lwd][LWD][spec-probe] POISON RESIDENT req=abc-1 idx=0 prompt_-1=2 "
                 "output_-1=0 prompt_len=1561 real=1561 start=0 num_spec=3")
WRITE_PROMPT = ("[lwd][LWD][spec-probe] WRITE INTO PROMPT REGION req=abc-1 "
                "start=0 num_spec=3 num_prompt=1561 spec_ids=[-1, -1, 240]")
INJECT = ("[Lwd][cloud-runner] INJECT window mismatch req=abc-3 seqno=4: "
          "chunk rows=1561 != scheduled=1536 (computed=0 prompt=4096)")

VERDICT_CASES = [
    VerdictCase(
        "空日志-无效", [], "实验无效",
        {"penalties_active": 0},
    ),
    VerdictCase(
        "wire全默认-无效且点名请求侧", [
            "[lwd][LWD][sampler-probe] wire penalties req=a rep=1.0 pres=0.0 freq=0.0",
        ], "实验无效",
        {"wire_penalties_total": 1, "wire_penalties_nonzero": 0},
    ),
    VerdictCase(
        "有效且干净-毒源枯竭", [ACTIVE, LAUNCH], "毒源枯竭",
        {"penalties_active": 1, "launch": 1, "poison": 0},
    ),
    VerdictCase(
        "旧版launch带min-历史负值", [
            ACTIVE,
            LAUNCH + " min=-1 max=151000",
            "[lwd][LWD][sampler-probe] bincount OUT-OF-RANGE shape=(16, 2048) "
            "stride=(2048, 1) ptr=0x7f3400 vocab=151936 min=-1 max=151000 sample=[-1]",
        ], "毒源枯竭",
        {"launch": 1, "neg_min_seen": 1, "out_of_range": 1, "observed_min": -1},
    ),
    VerdictCase(
        "账本毒-async设计常态-不算毒", [ACTIVE, LAUNCH, POISON_PROMPT],
        "毒源枯竭",
        {"poison": 1, "penalty_input_poison": 0},
    ),
    VerdictCase(
        "惩罚输入毒-修复链失效-静默污染", [
            ACTIVE + " poison_rows=2",
            LAUNCH,
            "[lwd][LWD][sampler-probe] PENALTY INPUT POISON rows=2/16",
        ], "毒源仍在-静默污染",
        # 双行只计一份:毒行数以 POISON 行为准
        {"penalty_input_poison": 2, "penalty_input_rows_seen": 1},
    ),
    VerdictCase(
        "毒在-无故障行-静默污染", [ACTIVE, LAUNCH, POISON_PROMPT, WRITE_PROMPT],
        "毒源仍在-静默污染",
        {"poison_prompt_route": 2, "fault": False},
    ),
    VerdictCase(
        "毒在-有故障行-已崩溃", [
            ACTIVE, LAUNCH,
            "[lwd][LWD][sampler-probe] PENALTY INPUT POISON rows=1/16",
            "EE1001 : 2026-10-20 10:33:00 device fault aivec kernel Illegal memory access",
        ], "毒源仍在-已崩溃",
        {"fault": True, "penalty_input_poison": 1},
    ),
    VerdictCase(
        "注入窗口失配字段解析", [ACTIVE, LAUNCH, INJECT], "毒源枯竭",
        {"inject_rows": 1561, "inject_sched": 1536},
    ),
]


def _check_verdict(case: VerdictCase) -> None:
    path = write_log(case.lines, "utf-8")
    try:
        rep = analyzer.analyze([path])
        got, _notes = analyzer.verdict(rep)
        assert got == case.verdict, f"{case.name}: {got} != {case.verdict}"
        c = case.checks
        assert rep.penalties_active == c.get("penalties_active", rep.penalties_active)
        assert rep.bincount_launches.count == c.get("launch", rep.bincount_launches.count)
        assert len(rep.poison_events) == c.get("poison", len(rep.poison_events))
        assert rep.wire_penalties_total == c.get("wire_penalties_total", rep.wire_penalties_total)
        assert rep.wire_penalties_nonzero == c.get("wire_penalties_nonzero", rep.wire_penalties_nonzero)
        assert rep.poison_prompt_route == c.get("poison_prompt_route", rep.poison_prompt_route)
        assert rep.bincount_launches.neg_min_seen == c.get("neg_min_seen", rep.bincount_launches.neg_min_seen)
        assert rep.bincount_launches.out_of_range == c.get("out_of_range", rep.bincount_launches.out_of_range)
        assert rep.bincount_launches.observed_min == c.get("observed_min", rep.bincount_launches.observed_min)
        assert rep.fault_observed == c.get("fault", rep.fault_observed)
        assert rep.penalty_input_poison == c.get("penalty_input_poison", rep.penalty_input_poison)
        assert rep.penalty_input_rows_seen == c.get("penalty_input_rows_seen", rep.penalty_input_rows_seen)
        if "inject_rows" in c:
            assert rep.inject_mismatch[0]["chunk_rows"] == c["inject_rows"]
            assert rep.inject_mismatch[0]["scheduled"] == c["inject_sched"]
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("case", VERDICT_CASES, ids=[c.name for c in VERDICT_CASES])
def test_verdict_matrix(case: VerdictCase) -> None:
    _check_verdict(case)


def test_json_output() -> None:
    path = write_log([
        ACTIVE, LAUNCH,
        "[lwd][LWD][sampler-probe] PENALTY INPUT POISON rows=1/16",
    ], "utf-8")
    out = Path(tempfile.gettempdir()) / "lwd_analyzer_test_out.json"
    try:
        rc = analyzer.main([str(path), "--json", str(out)])
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["verdict"] == "毒源仍在-静默污染"
        assert payload["penalty_input_poison"] == 1
    finally:
        path.unlink(missing_ok=True)
        out.unlink(missing_ok=True)


def test_adoption_line_routed() -> None:
    """无探针标记的 [cloud-params] 行也必须被解析(路由事故回归)。"""
    path = write_log(
        ["[lwd][cloud-params] chunk align profile adopted: block=128 hash=128 "
         "mamba_split=True eagle_drop=True lookahead=3"], "utf-8")
    try:
        rep = analyzer.analyze([path])
        assert rep.align_profile_adopted is not None
        assert "block=128" in rep.align_profile_adopted
    finally:
        path.unlink(missing_ok=True)
