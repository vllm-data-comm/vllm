# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""外部接入面守卫测试 + 包内依赖方向守卫。

core.py 内联读 vllm_config.lwd_config 选引擎类;serve.py 不再接触
lwd_control;重量级模块(需 torch 的引擎/调度器)以 find_spec 断言
存在性,不触发执行。
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

try:
    from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
except Exception:  # pragma: no cover
    pytest.skip("vllm 包入口不可导入", allow_module_level=True)

assert callable(LwdControlDiag.token_passthrough_enabled)
assert callable(LwdControlDiag.disable_down_enabled)
assert callable(LwdControlDiag.bypass_unembed_rows)

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"


def _spec(name: str):
    return importlib.util.find_spec(name)


def test_module_inventory():
    """核心模块存在性(不触发 torch 依赖的执行)。"""
    for name in [
        "vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker",
    ]:
        assert _spec(name) is not None, name


def test_deleted_modules_gone():
    """PD 分离相位机、旧配置/台账模块已删,不可再 import。"""
    for name in [
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefill_notify_queue",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_awaiting_tracker",
        "vllm.v1.lwd_control.lwd_config",
    ]:
        assert _spec(name) is None, name


def test_external_entry_sites():
    """core.py 内联选择块在;serve.py 零 lwd_control 引用。"""
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "vllm_config.lwd_config" in core
    assert "LwdEdgeEngineCore" in core and "LwdCloudEngineCore" in core
    serve = (VLLM_DIR / "entrypoints" / "cli" / "serve.py").read_text()
    assert "lwd_control" not in serve


def test_cloud_scheduler_keeps_debug_static_method():
    """lwd_debug 引用的静态方法名仍在(文本守卫,免 torch);core.py 无
    LWD 模型输出钩子(c2e 收编调度器 update_from_output + 引擎 post_step)。"""
    source = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_scheduler.py"
    ).read_text()
    assert "def _lwd_c2e_finish_reasons(" in source
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "lwd_process_model_output" not in core
    assert "lwd_handle_model_output" not in core


def test_cloud_package_never_imports_edge_package():
    """依赖方向守卫:云侧包禁止 import 边侧包。"""
    cloud_dir = VLLM_DIR / "v1" / "lwd_control" / "control_cloud_scheduler"
    offenders = [
        py.name
        for py in cloud_dir.glob("*.py")
        if "control_edge_scheduler" in py.read_text()
    ]
    assert offenders == []


def test_diag_package_purity():
    """诊断包纯度:零功能依赖(不 import 调度/引擎/通信类),不持
    执行适配器(future/伪输出归引擎);违者即诊断与功能再次糅合。"""
    diag_dir = VLLM_DIR / "v1" / "lwd_control" / "lwd_control_diag"
    files = list(diag_dir.glob("*.py"))
    assert files, "lwd_control_diag 包缺失"
    banned = (
        "control_edge_scheduler",
        "control_cloud_scheduler",
        "lwd_comm",
        "control_scheduler",
        "LwdBypassFuture",
        "LwdStubModelOutput",
        "LWD_EMPTY_MODEL_OUTPUT",
    )
    for py in files:
        text = py.read_text()
        for marker in banned:
            assert marker not in text, (py.name, marker)


def test_no_scheduler_mode_strings_left():
    """PD 分离判据字符串在包内清零(唯一形态 PD_mix)。"""
    for py in (VLLM_DIR / "v1" / "lwd_control").rglob("*.py"):
        text = py.read_text()
        assert "prefill_first" not in text, py
        assert "decode_first" not in text, py


# ------------------------------------------------------------------ #
# 评审闭环守卫(PR #3 review 轮落定的结构与接口约束)                  #
# ------------------------------------------------------------------ #
def test_lwd_control_uses_lwd_logger_only():
    """日志接口唯一化:lwd_control 包内零 vllm.logger 直连,全部走
    vllm.lwd_logger(消息自动带 [lwd] 前缀,VLLM_LWD_LOG_LEVEL 独立控级);
    本 PR 在原生文件新加的两处日志(sched dbg / has_work)同口径。"""
    for py in (VLLM_DIR / "v1" / "lwd_control").rglob("*.py"):
        text = py.read_text()
        assert "from vllm.logger import" not in text, py
        assert "init_logger" not in text, py
    sched = (VLLM_DIR / "v1" / "core" / "sched" / "scheduler.py").read_text()
    assert "lwd_logger.info" in sched
    base_engine = (
        VLLM_DIR / "v1" / "lwd_control" / "control_scheduler" / "lwd_base_engine.py"
    ).read_text()
    assert "lwd_logger.info" in base_engine


def test_has_work_override_lives_in_base_engine():
    """has_work 状态转换日志收编基类:core.py 原生判定不动,覆写在
    LwdBaseEngineCore(评审:通用逻辑放 lwd_base_engine)。"""
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "_last_has_work_state" not in core
    base_engine = (
        VLLM_DIR / "v1" / "lwd_control" / "control_scheduler" / "lwd_base_engine.py"
    ).read_text()
    assert "def has_work" in base_engine


def test_message_dispatch_skeleton_in_base_engine():
    """接收线程消息入口统一在基类:入口 _lwd_on_payload + 未知帧告警
    丢弃住基类;两侧引擎只实现 _lwd_dispatch_message 钩子(评审:与
    cloud 的分派一起提取到基类)。"""
    base_engine = (
        VLLM_DIR / "v1" / "lwd_control" / "control_scheduler" / "lwd_base_engine.py"
    ).read_text()
    assert "def _lwd_on_payload" in base_engine
    assert "drop unexpected frame" in base_engine
    for sub in (
        "control_edge_scheduler/lwd_edge_engine.py",
        "control_cloud_scheduler/lwd_cloud_engine.py",
    ):
        text = (VLLM_DIR / "v1" / "lwd_control" / sub).read_text()
        assert "def _lwd_dispatch_message" in text, sub
        assert "def _lwd_on_payload" not in text, sub


def test_review_deleted_code_stays_deleted():
    """评审裁剪的代码不再回流(后续功能落地时随功能整体回归):
    - 零接受活行剔除(_filter_deliver_rows,4 卡问题修复,后续再带上);
    - 边调度器请求入口编排 lwd_edge_add_request(收编进引擎 add_request);
    - 云侧 wire-aware 哈希链统一在 _lwd_build_request,不回流旧构造helper。"""
    cloud_sched = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_scheduler.py"
    ).read_text()
    assert "_filter_deliver_rows" not in cloud_sched
    edge_sched = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_scheduler.py"
    ).read_text()
    assert "def lwd_edge_add_request" not in edge_sched
    cloud_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_engine.py"
    ).read_text()
    assert "_lwd_build_hashed_request" not in cloud_engine
    assert "block_hasher=block_hasher" in cloud_engine
    assert "LwdAbortNotify" not in edge_sched
    edge_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_engine.py"
    ).read_text()
    assert "LwdAbortNotify" in edge_engine
    assert "self.scheduler.lwd_edge_abort(request_ids)" in edge_engine


def test_cloud_abort_uses_native_abort_path():
    source = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_engine.py"
    ).read_text()
    assert (
        "isinstance(msg, (LwdRangeNotify, LwdRequestNotify, LwdAbortNotify))" in source
    )
    assert "def _lwd_handle_abort_notify" in source
    assert "self.aborts_queue.put_nowait([request_id])" in source
    assert "EngineCoreRequestType.ABORT" in source


def test_edge_add_request_communication_in_engine():
    """请求入口分工(评审:通信最好在 engine 里进行):引擎 add_request
    编排 准入+路由(lwd_admit_request 返回 dp_idx)-> 按路由云预告
    (_lwd_notify_request(dp_idx, ...))-> 原生 super().add_request
    (abort_immediately 走原生 abort_requests)。"""
    edge_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_engine.py"
    ).read_text()
    assert "def add_request" in edge_engine
    assert "dp_idx = self.scheduler.lwd_admit_request(request)" in edge_engine
    assert "self._lwd_notify_request(dp_idx, request)" in edge_engine
    assert "super().add_request(request)" in edge_engine
    edge_sched = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_scheduler.py"
    ).read_text()
    assert "def lwd_admit_request" in edge_sched
    # 预告发布(通信)不再住调度器
    assert "def lwd_edge_notify_request" not in edge_sched


def test_edge_batch_queue_depth_is_two():
    """批队列深度 = 2(评审定值;性能验证在真机环境补)。"""
    import regex as re

    edge_engine = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_edge_scheduler"
        / "lwd_edge_engine.py"
    ).read_text()
    match = re.search(r"^LWD_EDGE_BATCH_QUEUE_DEPTH\s*=\s*(\d+)", edge_engine, re.M)
    assert match is not None
    assert int(match.group(1)) == 2
