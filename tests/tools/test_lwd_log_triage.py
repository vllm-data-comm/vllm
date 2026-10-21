# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-driven tests for tools/lwd_log_triage.py.

Encoding cases mirror the production-log reality (UTF-16 with/without
BOM, GBK, CRLF); verdict cases are one synthetic log per broken hop of
the 2026-10 bring-up chain, each asserting the hop the tool must name.
Log-line formats are taken from the emitting code (lwd_cloud_worker /
lwd_cloud_scheduler), not invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lwd_log_triage as triage  # noqa: E402


# 完整链路段落(带新饱和打点),各用例按断点截取
SEG_HINT = ("[Lwd][pre-recv] HINT seqno=0 num_tokens=11 ts=100.4")
SEG_SUBMIT = ("[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=0 num_tokens=11 "
              "aux_rows=0 inflight=1 queued=0 ts=100.5")
SEG_DONE = "[Lwd][pre-recv] DONE channel=lwd_up seqno=0 ts=100.9"
SEG_REPORT = "[Lwd][pre-recv] REPORT channel=lwd_up seqno=0 ts=100.95"
SEG_DRAIN = "[Lwd][pre-recv] DRAIN n=1 seqnos=[0] advanced=[0] ts=100.96"
SEG_GATE_OPEN = "[cloud-sched] notify seqno=0 gate OPEN watermark=0"
SEG_GATE_HOLD = "[cloud-sched] notify seqno=0 gate HOLD watermark=-1 (UP 未收齐;后续静默重试)"
SEG_NOTIFY = "[cloud] ingest msg=LwdRangeNotify seqno=0"


CLOUD_STARTUP_OK = [
    "(EngineCore pid=1) INFO [lwd_cloud_engine.py:60] [pre-recv] hint MQ "
    "created by engine, done MQ attached: OK",
    "(Worker pid=2) INFO [lwd_cloud_worker.py:360] [Lwd][pre-recv] comm "
    "thread starting rank=0 hint MQ=attached",
]
EMBED = ("(EngineCore pid=3) INFO [lwd_edge_worker.py:76] [lwd] [LWD]"
         "[edge-worker] EMBED seqno=0 reqs=1 tokens=11 has_mrope=[False]")


def _run(tmp_path: Path, cloud_lines: list[str], edge_lines: list[str] | None = None):
    reports = []
    cloud = tmp_path / "cloud.log"
    cloud.write_text("\n".join(cloud_lines) + "\n", encoding="utf-8")
    reports.append(triage.analyze(cloud, None))
    if edge_lines is not None:
        edge = tmp_path / "edge.log"
        edge.write_text("\n".join(edge_lines) + "\n", encoding="utf-8")
        reports.append(triage.analyze(edge, None))
    return triage.verdicts(reports)


# ---------------- encoding ----------------
@pytest.mark.parametrize(
    "payload, encoding",
    [
        ("EMBED seqno=0 正常", "utf-8"),
        ("EMBED seqno=0 bom", "utf-8"),
        ("EMBED seqno=0 utf16", "utf-16"),
    ],
)
def test_detect_encoding_bom_variants(tmp_path: Path, payload: str, encoding: str) -> None:
    raw = {"utf-8": payload.encode("utf-8"),
           "utf-8-sig": b"\xef\xbb\xbf" + payload.encode("utf-8"),
           "utf-16": b"\xff\xfe" + payload.encode("utf-16-le")}.get(encoding)
    if encoding == "utf-8":
        raw = payload.encode("utf-8")
    path = tmp_path / "enc.log"
    path.write_bytes(raw)
    detected = triage.detect_encoding(path)
    assert detected.startswith(encoding.replace("-sig", ""))


def test_detect_encoding_bare_utf16le_and_gbk(tmp_path: Path) -> None:
    le = tmp_path / "le.log"
    le.write_bytes("EMBED seqno=0 中文内容拉长样本".encode("utf-16-le"))
    assert triage.detect_encoding(le) == "utf-16-le"
    gbk = tmp_path / "gbk.log"
    gbk.write_bytes("EMBED seqno=0 中文内容".encode("gbk"))
    assert triage.detect_encoding(gbk) == "gbk"


def test_iter_lines_never_crashes_and_strips_crlf(tmp_path: Path) -> None:
    path = tmp_path / "crlf.log"
    path.write_bytes("line one\r\nline two\r\n".encode("utf-16"))
    lines = triage.iter_lines(path, None)
    assert lines == ["line one", "line two"]


# ---------------- role + verdict table ----------------
def test_role_detection(tmp_path: Path) -> None:
    cloud = tmp_path / "c.log"
    cloud.write_text("[cloud-sched] notify seqno=0 deferred: x\n", encoding="utf-8")
    assert triage.analyze(cloud, None).role == "cloud"
    edge = tmp_path / "e.log"
    edge.write_text(EMBED + "\n", encoding="utf-8")
    assert triage.analyze(edge, None).role == "edge"


VERDICT_CASES = [
    (
        "engine_code_absent",
        [
            "(Worker pid=9) INFO [lwd_cloud_worker.py:166] [lwd] [Lwd][perf] "
            "cloud-exec total=0.13ms",
        ],
        None,
        "vllm 仓未同步",
    ),
    (
        "attached_none_connection",
        [
            "[pre-recv] hint MQ created by engine, done MQ attached: NONE",
            "[pre-recv] done MQ acquire from tcp://1.2.3.4:29600 failed "
            "(ConnectionRefusedError: [Errno 111] refused)",
        ],
        None,
        "wire store 不可达",
    ),
    (
        "attached_none_no_publish",
        [
            "[pre-recv] hint MQ created by engine, done MQ attached: NONE",
        ],
        None,
        "worker 发布未执行",
    ),
    (
        "hint_mq_none",
        [
            "[pre-recv] hint MQ created by engine, done MQ attached: OK",
            "[Lwd][pre-recv] comm thread starting rank=0 hint MQ=none",
        ],
        None,
        "环境变量句柄",
    ),
    (
        "notify_never_reached_cloud",
        CLOUD_STARTUP_OK,
        [EMBED],
        "控制面预告未达",
    ),
    (
        "notify_without_hint",
        CLOUD_STARTUP_OK + [SEG_NOTIFY],
        None,
        "hint 投递断",
    ),
    (
        "submit_missing",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT],
        None,
        "通信线程未消费",
    ),
    (
        "done_missing",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT],
        None,
        "UP irecv 未收齐",
    ),
    (
        "report_missing",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT, SEG_DONE],
        None,
        "完成上报未入",
    ),
    (
        "drain_missing",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT, SEG_DONE, SEG_REPORT],
        None,
        "排水没跑",
    ),
    (
        "gate_hold_after_drain",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT, SEG_DONE, SEG_REPORT,
                            SEG_DRAIN, SEG_GATE_HOLD],
        None,
        "水位未推进",
    ),
    (
        "gate_open_no_post",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT, SEG_DONE, SEG_REPORT,
                            SEG_DRAIN, SEG_GATE_OPEN],
        None,
        "批未下发",
    ),
    (
        "add_not_landed",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT, SEG_DONE, SEG_REPORT,
                            SEG_DRAIN, SEG_GATE_OPEN,
                            "[cloud-sched] notify seqno=0 deferred: request(s) not added yet"],
        None,
        "ADD 未落地",
    ),
    (
        "preflight_rejected",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT, SEG_DONE, SEG_REPORT,
                            SEG_DRAIN, SEG_GATE_OPEN,
                            "[cloud-sched] notify seqno=0 deferred: preflight rejected "
                            "reqs=1 tokens=11"],
        None,
        "preflight",
    ),
    (
        "chain_complete",
        CLOUD_STARTUP_OK + [SEG_NOTIFY, SEG_HINT, SEG_SUBMIT, SEG_DONE, SEG_REPORT,
                            SEG_DRAIN, SEG_GATE_OPEN,
                            "[Lwd][pre-recv] POST seqno=0 reqs=1 tokens=11 "
                            "endpoint=True ts=101.2 mem_alloc=1.00GiB"],
        None,
        "链路打点完整",
    ),
]


@pytest.mark.parametrize("name,cloud,edge,expected", VERDICT_CASES,
                         ids=[c[0] for c in VERDICT_CASES])
def test_verdict_names_the_broken_hop(
    tmp_path: Path, name: str, cloud: list[str], edge: list[str] | None, expected: str
) -> None:
    verdicts = _run(tmp_path, cloud, edge)
    assert any(expected in v for v in verdicts), f"{name}: {verdicts}"


def test_chain_hop_deltas_rendered(tmp_path: Path) -> None:
    verdicts = _run(tmp_path, VERDICT_CASES[-1][1], None)
    assert any("sub→done=0.400s" in v for v in verdicts), verdicts


def test_edge_ladder_no_http_request(tmp_path: Path) -> None:
    edge = [
        "[edge] engine assembled: cloud LwdPeer at tcp://1.2.3.4:5550, "
        "native batch-queue stepping depth=2",
    ]
    verdicts = _run(tmp_path, CLOUD_STARTUP_OK, edge)
    assert any("请求未打到边" in v for v in verdicts), verdicts


def test_edge_ladder_http_but_engine_idle(tmp_path: Path) -> None:
    edge = [
        'INFO:     76.76.26.17:38222 - "POST /v1/chat/completions HTTP/1.1" 200',
        "[has_work] engines_running=False has_requests=False batch_queue=False",
    ]
    verdicts = _run(tmp_path, CLOUD_STARTUP_OK, edge)
    assert any("引擎未消费" in v for v in verdicts), verdicts


def test_edge_ladder_pending_but_no_embed(tmp_path: Path) -> None:
    edge = [
        'INFO:     76.76.26.17:38222 - "POST /v1/chat/completions HTTP/1.1" 200',
        "[has_work] engines_running=False has_requests=True batch_queue=False",
    ]
    verdicts = _run(tmp_path, CLOUD_STARTUP_OK, edge)
    assert any("未出 EMBED 批" in v for v in verdicts), verdicts


def test_render_no_placeholder_when_startup_present(tmp_path: Path) -> None:
    cloud = tmp_path / "c.log"
    cloud.write_text(
        "[pre-recv] hint MQ created by engine, done MQ attached: OK\n",
        encoding="utf-8",
    )
    text = triage.render([triage.analyze(cloud, None)], as_json=False)
    assert "无启动段标记" not in text


def test_request_hit_cloud_idle_frontend(tmp_path: Path) -> None:
    """请求误打云侧闲置 API server:云有 POST 行、边没有,且无任何 seqno 事件。"""
    cloud = CLOUD_STARTUP_OK + [
        'INFO:     76.76.26.17:38222 - "POST /v1/chat/completions HTTP/1.1" 200',
    ]
    edge = ["[edge] engine assembled: cloud LwdPeer at tcp://1.2.3.4:5550, depth=2"]
    verdicts = _run(tmp_path, cloud, edge)
    assert any("闲置 API server" in v for v in verdicts), verdicts


def test_mixed_encoding_file_fully_parsed(tmp_path: Path) -> None:
    """复现真机形态:UTF-16 BOM 头(启动段) + 原生 UTF-8 追加段(请求期)。
    单遍按 BOM 解码会把追加段变乱码、EMBED 全部丢失。"""
    head = "\n".join([
        "INFO data-plane warmup done: channel=UP my_rank=0 peer=1",
        "INFO [cloud-params] scheduling limits verified",
    ])
    tail = "\n".join([
        "INFO [lwd] [has_work] engines_running=False has_requests=True batch_queue=False",
        "INFO [lwd] [LWD][edge-worker] EMBED seqno=0 reqs=1 tokens=11 has_mrope=[False]",
        "[END] 2026/10/19 19:58:48",
    ])
    path = tmp_path / "mixed.log"
    path.write_bytes(b"\xff\xfe" + head.encode("utf-16-le") + b"\n\0" + tail.encode("utf-8"))
    rep = triage.analyze(path, None)
    assert rep.role == "edge"
    assert 0 in rep.chains and rep.chains[0].embed is not None
    assert rep.has_pending_requests is True


def test_nfkc_folds_lookalikes(tmp_path: Path) -> None:
    """全角数字/宽空格等"长得像"字符经 NFKC 折叠后必须能匹配。"""
    path = tmp_path / "lookalike.log"
    line = ("INFO [lwd] [LWD][edge-worker] EMBED seqno=\uff10 reqs=\uff11 "
            "tokens=\uff11\u3000end").replace("seqno=\uff10", "seqno=0")
    path.write_text(line + "\n", encoding="utf-8")
    rep = triage.analyze(path, None)
    assert 0 in rep.chains and rep.chains[0].embed is not None
