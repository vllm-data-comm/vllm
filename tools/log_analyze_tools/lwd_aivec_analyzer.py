#!/usr/bin/env python3
"""LWD aivec 探针日志分析器(配套 2026-10-20 MTE/bincount 负偏移排查)。

消费 lwd-debug-1019 分支各探针打出的日志行,回答一个问题:
**毒值(-1 哨兵)现在还在不在?**——崩溃本身是三重彩票(毒值在批内
首行 + bin_counts 恰落段边界才 fault),"没崩"没有判别力,所以本工具
以条件证据(POISON RESIDENT / WRITE INTO PROMPT REGION / 历史负值)
为准,并自动给出四态判定。

识别的日志行(匹配不变子串,容忍任意前缀与 NFKC 全角变体):
  sampler-probe(云 worker,ascend 仓):
    "bincount launch shape=.. stride=.. ptr=0x.. vocab=N num_seqs=N[ min=A max=B]"
    "bincount OUT-OF-RANGE .. min=A max=B sample=[..]"        (旧版探针)
    "penalties ACTIVE logits=.. [rep=.. pres=.. freq=..] prompt_ids=.."
    "wire penalties req=R rep=F pres=F freq=F"                 (边->云参数)
  spec-probe(云,vllm 仓):
    "WRITE INTO PROMPT REGION req=R start=N num_spec=N num_prompt=N spec_ids=[..]"
    "POISON RESIDENT req=R idx=N prompt_-1=A output_-1=B prompt_len=N real=N start=N num_spec=N"
  契约/绊线:
    "chunk align profile adopted: block=.. hash=.. mamba_split=.. eagle_drop=.. lookahead=.."
    "INJECT window mismatch req=R seqno=N: chunk rows=A != scheduled=B (computed=C prompt=P)"
    "prompt chunk offset mismatch for R: start=.."
  疑似故障行: EZ9999 / aivec / illegal GM / AI Core / InnerError(仅计数+定位)

用法:
    python3 tools/log_analyze_tools/lwd_aivec_analyzer.py cloud.log [more.log ...] \
        [--json out.json] [--encoding utf-16]

编码自适应: BOM 判定 -> 无 BOM 时 NUL 启发式(UTF-16) -> utf-8/gbk/big5
回退链,读取 errors="replace" 永不因解码失败崩溃;--encoding 强制指定。
跨机器拷贝的日志可能混合行尾,逐行 rstrip("\\n\\r")。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

_BOMS = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16"),  # 带 BOM 用 utf-16 让 BOM 被消费,防首行残留 \ufeff
    (b"\xfe\xff", "utf-16"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
)


def detect_encoding(path: Path, sample_bytes: int = 64 * 1024) -> str:
    """BOM 判定 -> 无 BOM 时 NUL 占比+奇偶启发式 -> 多编码回退链。"""
    with path.open("rb") as fh:
        head = fh.read(sample_bytes)
    for bom, encoding in _BOMS:
        if head.startswith(bom):
            return encoding
    sample = head[: sample_bytes // 2]
    if sample and b"\x00" in sample:
        nul = float(sample.count(b"\x00")) / max(len(sample), 1)
        odd = sum(1 for i in range(1, len(sample), 2) if sample[i:i + 1] == b"\x00")
        even = sum(1 for i in range(0, len(sample), 2) if sample[i:i + 1] == b"\x00")
        # 注意先做 NUL 检查再试 utf-8:含 NUL 的字节串在 utf-8 下"合法"
        if nul > 0.125 and (odd == 0) != (even == 0):
            return "utf-16-le" if odd > even else "utf-16-be"
    for encoding in ("utf-8", "gbk", "big5"):
        try:
            head.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"


def _normalize(line: str) -> str:
    """NFKC 折叠全角变体(转码跳板产生的形近字符不至于断匹配),去混合行尾。"""
    return unicodedata.normalize("NFKC", line).rstrip("\n\r")


def iter_lines(path: Path, encoding: str | None = None) -> list[str]:
    encoding = encoding or detect_encoding(path)
    text = path.read_text(encoding=encoding, errors="replace")
    return [_normalize(line) for line in text.splitlines()]


# --------------------------------------------------------------------- #
# 行文法(正则只锚不变子串,不锚行首前缀)                                #
# --------------------------------------------------------------------- #
@dataclass
class LaunchStat:
    count: int = 0
    max_num_seqs: int = 0
    distinct_ptrs: set[str] = field(default_factory=set)
    neg_min_seen: int = 0          # 旧版探针 launch 行带 min= 且为负
    out_of_range: int = 0          # 旧版 OUT-OF-RANGE 行
    observed_min: int | None = None
    observed_max: int | None = None


@dataclass
class Report:
    files: list[str] = field(default_factory=list)
    encodings: dict[str, str] = field(default_factory=dict)
    penalties_active: int = 0
    penalty_input_poison: int = 0        # 修复后惩罚输入仍含 -1 的行数(真判据)
    penalty_input_rows_seen: int = 0
    bincount_launches: LaunchStat = field(default_factory=LaunchStat)
    wire_penalties_total: int = 0
    wire_penalties_nonzero: int = 0
    probe_installed: int = 0
    align_profile_adopted: str | None = None
    poison_events: list[dict] = field(default_factory=list)
    write_into_prompt: list[dict] = field(default_factory=list)
    inject_mismatch: list[dict] = field(default_factory=list)
    offset_mismatch: int = 0
    suspected_fault_lines: list[tuple[str, int, str]] = field(default_factory=list)

    @property
    def poison_prompt_route(self) -> int:
        return sum(1 for e in self.poison_events if e["prompt_-1"] > 0) + len(
            self.write_into_prompt
        )

    @property
    def poison_output_route(self) -> int:
        return sum(1 for e in self.poison_events if e["output_-1"] > 0)

    @property
    def fault_observed(self) -> bool:
        return bool(self.suspected_fault_lines)


_RE_LAUNCH = re.compile(
    r"bincount launch shape=\((?P<shape>[^)]*)\).*?ptr=(?P<ptr>0x[0-9a-fA-F]+) "
    r"vocab=(?P<vocab>\d+) num_seqs=(?P<seqs>None|\d+)(?: min=(?P<min>-?\d+) "
    r"max=(?P<max>-?\d+))?"
)
_RE_OUT_OF_RANGE = re.compile(r"bincount OUT-OF-RANGE.*?min=(?P<min>-?\d+) max=(?P<max>-?\d+)")
_RE_ACTIVE = re.compile(r"penalties ACTIVE(.*?poison_rows=(?P<poison>-?\d+))?")
_RE_PENALTY_POISON = re.compile(r"PENALTY INPUT POISON rows=(?P<hit>\d+)/(?P<total>\d+)")
_RE_WIRE = re.compile(
    r"wire penalties req=(?P<req>\S+) rep=(?P<rep>-?[\d.]+) "
    r"pres=(?P<pres>-?[\d.]+) freq=(?P<freq>-?[\d.]+)"
)
_RE_INSTALLED = re.compile(r"sampler-probe\] installed on worker \(budget=(\d+)\)")
_RE_ADOPTED = re.compile(
    r"chunk align profile adopted: block=(\d+) hash=(\d+) "
    r"mamba_split=(\w+) eagle_drop=(\w+) lookahead=(\d+)"
)
_RE_POISON = re.compile(
    r"POISON RESIDENT req=(?P<req>\S+) idx=(?P<idx>\d+) "
    r"prompt_-1=(?P<pn>\d+) output_-1=(?P<on>\d+) prompt_len=(?P<pl>\d+) "
    r"real=(?P<real>\d+) start=(?P<start>-?\d+) num_spec=(?P<ns>\d+)"
)
_RE_WRITE_PROMPT = re.compile(
    r"WRITE INTO PROMPT REGION req=(?P<req>\S+) start=(?P<start>-?\d+) "
    r"num_spec=(?P<ns>\d+) num_prompt=(?P<pl>\d+)"
)
_RE_INJECT = re.compile(
    r"INJECT window mismatch req=(?P<req>\S+) seqno=(?P<seqno>\d+): "
    r"chunk rows=(?P<rows>\d+) != scheduled=(?P<sched>\d+) "
    r"\(computed=(?P<computed>\d+) prompt=(?P<prompt>\d+)\)"
)
_RE_OFFSET = re.compile(r"prompt chunk offset mismatch for \S+:")

# 疑似 NPU 故障行(aivec/非法访存在 CANN 侧的常见形态;只计数定位,不解读)
_FAULT_PATTERNS = ("EZ9999", "aivec", "illegal gm", "ai core", "innererror", "device fault")


def analyze_line(line: str, rep: Report) -> None:
    """只解析探针行;疑似故障行由 analyze() 持真实行号负责。"""
    if (m := _RE_ACTIVE.search(line)):
        rep.penalties_active += 1
        if m.group("poison") is not None:
            # 覆盖计数只记一次;毒行数以 PENALTY INPUT POISON 行为准
            # (同一事件两行都打,此处累计会双份)
            rep.penalty_input_rows_seen += 1
    if (m := _RE_PENALTY_POISON.search(line)):
        rep.penalty_input_poison += int(m.group("hit"))
    if (m := _RE_LAUNCH.search(line)):
        s = rep.bincount_launches
        s.count += 1
        s.distinct_ptrs.add(m.group("ptr"))
        if m.group("seqs") != "None":
            s.max_num_seqs = max(s.max_num_seqs, int(m.group("seqs")))
        if m.group("min") is not None:
            mn, mx = int(m.group("min")), int(m.group("max"))
            if mn < 0:
                s.neg_min_seen += 1
            s.observed_min = mn if s.observed_min is None else min(s.observed_min, mn)
            s.observed_max = mx if s.observed_max is None else max(s.observed_max, mx)
    if (m := _RE_OUT_OF_RANGE.search(line)):
        rep.bincount_launches.out_of_range += 1
        mn = int(m.group("min"))
        st = rep.bincount_launches
        st.observed_min = mn if st.observed_min is None else min(st.observed_min, mn)
    if (m := _RE_WIRE.search(line)):
        rep.wire_penalties_total += 1
        if (float(m.group("rep")) != 1.0 or float(m.group("pres")) != 0.0
                or float(m.group("freq")) != 0.0):
            rep.wire_penalties_nonzero += 1
    if _RE_INSTALLED.search(line):
        rep.probe_installed += 1
    if (m := _RE_POISON.search(line)):
        rep.poison_events.append({
            "req": m.group("req"), "idx": int(m.group("idx")),
            "prompt_-1": int(m.group("pn")), "output_-1": int(m.group("on")),
            "prompt_len": int(m.group("pl")), "real": int(m.group("real")),
            "start": int(m.group("start")), "num_spec": int(m.group("ns")),
        })
    if (m := _RE_WRITE_PROMPT.search(line)):
        rep.write_into_prompt.append({
            "req": m.group("req"), "start": int(m.group("start")),
            "num_spec": int(m.group("ns")), "num_prompt": int(m.group("pl")),
        })
    if (m := _RE_INJECT.search(line)):
        rep.inject_mismatch.append({
            "req": m.group("req"), "seqno": int(m.group("seqno")),
            "chunk_rows": int(m.group("rows")), "scheduled": int(m.group("sched")),
            "computed": int(m.group("computed")), "prompt": int(m.group("prompt")),
        })
    if _RE_OFFSET.search(line):
        rep.offset_mismatch += 1
    if (m := _RE_ADOPTED.search(line)):
        rep.align_profile_adopted = (
            f"block={m.group(1)} hash={m.group(2)} mamba_split={m.group(3)} "
            f"eagle_drop={m.group(4)} lookahead={m.group(5)}"
        )


def analyze(paths: list[Path], encoding: str | None = None) -> Report:
    rep = Report()
    for path in paths:
        rep.files.append(str(path))
        rep.encodings[str(path)] = encoding or detect_encoding(path)
        # 全量行进解析器(正文法已足够特异);疑似故障行带真实行号
        lines = iter_lines(path, encoding)
        for no, line in enumerate(lines, 1):
            analyze_line(line, rep)
            low = line.lower()
            for pat in _FAULT_PATTERNS:
                if pat in low:
                    rep.suspected_fault_lines.append((str(path), no, line.strip()[:160]))
                    break
    return rep


VALID, INVALID, SILENT, CRASHED, GONE = (
    "实验有效", "实验无效", "毒源仍在-静默污染", "毒源仍在-已崩溃", "毒源枯竭"
)


def verdict(rep: Report) -> tuple[str, list[str]]:
    """四态判定 + 依据列表。有效性 = 惩罚路径确实执行过。"""
    notes: list[str] = []
    valid = rep.penalties_active > 0 or rep.bincount_launches.count > 0
    if not valid:
        notes.append(
            "惩罚路径未执行(penalties ACTIVE=0 且 bincount launch=0):"
            "本次日志无法证明任何事,先确认请求带非零 penalty"
        )
        if rep.wire_penalties_total > 0 and rep.wire_penalties_nonzero == 0:
            notes.append("wire penalties 全部为默认值(rep=1.0/pres=0/freq=0)——请求侧就没带惩罚")
        return INVALID, notes
    if rep.penalties_active == 0:
        notes.append("有 bincount launch 但无 penalties ACTIVE 行(budget 耗尽或旧版云):launch 数即证据")
    # 毒判据(2026-10-20 修正):惩罚输入(修复后)含 -1,或 spec 写入 prompt 区。
    # token_ids_cpu 账本含 -1 是 async 调度设计常态(上游 :3815 亲手写 -1),
    # 仅作信息项,不构成毒证据。
    kernel_poison = rep.penalty_input_poison + len(rep.write_into_prompt)
    if kernel_poison:
        if rep.penalty_input_poison:
            notes.append(
                f"PENALTY INPUT POISON:修复后惩罚输入含 -1 共 "
                f"{rep.penalty_input_poison} 行(观测 {rep.penalty_input_rows_seen} 次)"
                f"——占位修复链(update_async_output_token_ids)在云侧失效"
            )
        for e in rep.write_into_prompt[:5]:
            notes.append(f"WRITE INTO PROMPT REGION req={e['req']} start={e['start']} num_prompt={e['num_prompt']}")
        if len(rep.write_into_prompt) > 5:
            notes.append(f"...共越界写 {len(rep.write_into_prompt)} 条")
        return (CRASHED if rep.fault_observed else SILENT), notes
    notes.append("修复后惩罚输入无 -1(占位修复链工作正常)")
    if rep.poison_events:
        notes.append(
            f"注:token_ids_cpu 账本含 -1 共 {len(rep.poison_events)} 条——"
            "async 设计常态,不进内核"
        )
    if rep.bincount_launches.neg_min_seen or rep.bincount_launches.out_of_range:
        notes.append(
            f"但历史 launch 观测到负值(min={rep.bincount_launches.observed_min},"
            f"OUT-OF-RANGE {rep.bincount_launches.out_of_range} 条):旧日志期毒值确实存在"
        )
    return GONE, notes


def render(rep: Report) -> str:
    v, notes = verdict(rep)
    s = rep.bincount_launches
    out = [
        "=" * 64,
        f"LWD aivec 探针日志分析  files={rep.files}",
        f"编码判定: {rep.encodings}",
        "-" * 64,
        f"探针安装行          : {rep.probe_installed}",
        f"对齐 profile 采纳    : {rep.align_profile_adopted or '(未见——边侧未到启动校验?)'}",
        f"wire penalties       : 总 {rep.wire_penalties_total} / 非默认 {rep.wire_penalties_nonzero}",
        f"penalties ACTIVE     : {rep.penalties_active}"
        + (f" (含 poison_rows 标注 {rep.penalty_input_rows_seen} 次)" if rep.penalty_input_rows_seen else ""),
        f"惩罚输入毒(真判据)   : {rep.penalty_input_poison} 行 / 观测 {rep.penalty_input_rows_seen} 次"
        f"  [PENALTY INPUT POISON 命中即修复链失效]",
        f"bincount launch      : {s.count} 次, num_seqs<= {s.max_num_seqs}, "
        f"去重 ptr {len(s.distinct_ptrs)} 个",
        f"  (三重彩票参考: 仅批内第 0 行且 ptr 段边界才 fault; num_seqs 越大 lottery 越难中)",
        f"历史负值观测         : launch 行 min<0 {s.neg_min_seen} 条 / OUT-OF-RANGE {s.out_of_range} 条"
        + (f" / 历史最小值 {s.observed_min}" if s.observed_min is not None else ""),
        f"POISON RESIDENT      : {len(rep.poison_events)} 条 (prompt 路线 {rep.poison_prompt_route} = "
        f"POISON prompt 命中 + 越界写事件 / output 路线 {rep.poison_output_route})",
        f"WRITE INTO PROMPT    : {len(rep.write_into_prompt)} 条",
        f"INJECT window 失配   : {len(rep.inject_mismatch)} 条" + (
            f" 首条: {rep.inject_mismatch[0]}" if rep.inject_mismatch else ""),
        f"offset mismatch      : {rep.offset_mismatch} 条",
        f"疑似故障行           : {len(rep.suspected_fault_lines)} 条",
    ]
    for f, no, text in rep.suspected_fault_lines[:5]:
        out.append(f"  {f}:{no} {text}")
    out += ["-" * 64, f"判定: {v}"]
    out += [f"  * {n}" for n in notes]
    out.append("=" * 64)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LWD aivec 探针日志分析器")
    parser.add_argument("logs", nargs="+", type=Path, help="日志文件(可多个,任意编码)")
    parser.add_argument("--json", type=Path, default=None, help="附加输出 JSON 到此路径")
    parser.add_argument("--encoding", default=None,
                        help="强制指定输入编码(默认 BOM/NUL 启发式自适应)")
    args = parser.parse_args(argv)
    for path in args.logs:
        if not path.is_file():
            print(f"[analyzer] 文件不存在: {path}", file=sys.stderr)
            return 2
    rep = analyze(args.logs, args.encoding)
    print(render(rep))
    if args.json:
        v, _ = verdict(rep)
        payload = {
            "files": rep.files, "encodings": rep.encodings, "verdict": v,
            "penalties_active": rep.penalties_active,
            "penalty_input_poison": rep.penalty_input_poison,
            "penalty_input_rows_seen": rep.penalty_input_rows_seen,
            "bincount_launches": {
                "count": rep.bincount_launches.count,
                "max_num_seqs": rep.bincount_launches.max_num_seqs,
                "distinct_ptrs": len(rep.bincount_launches.distinct_ptrs),
                "neg_min_seen": rep.bincount_launches.neg_min_seen,
                "out_of_range": rep.bincount_launches.out_of_range,
                "observed_min": rep.bincount_launches.observed_min,
            },
            "wire_penalties_total": rep.wire_penalties_total,
            "wire_penalties_nonzero": rep.wire_penalties_nonzero,
            "poison_events": rep.poison_events,
            "write_into_prompt": rep.write_into_prompt,
            "inject_mismatch": rep.inject_mismatch,
            "offset_mismatch": rep.offset_mismatch,
            "suspected_fault_lines": rep.suspected_fault_lines,
            "align_profile_adopted": rep.align_profile_adopted,
        }
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[analyzer] JSON 已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
