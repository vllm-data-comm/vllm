# log_analyze_tools

LWD 调试期的日志分析工具集(配套分支 `lwd-debug-1019` 的探针体系)。

## lwd_aivec_analyzer.py — aivec/bincount 负偏移排查分析器

配套 2026-10-20 MTE 排查的探针(`sampler-probe` / `spec-probe` / 驻留扫描),
回答一个问题:**毒值(-1 哨兵)现在还在不在**。

背景:崩溃本身是三重彩票(毒值在批内首行 + `bin_counts` 恰落分配器段
边界才 fault;非首行/段中的负偏移写是**静默污染**),所以"没崩"没有
判别力,本工具以条件证据为准。

```bash
python3 tools/log_analyze_tools/lwd_aivec_analyzer.py cloud.log edge.log \
    [--json out.json] [--encoding utf-16]
```

四态判定:

| 判定 | 含义 |
|---|---|
| 实验无效 | 惩罚路径未执行(penalties ACTIVE=0 且 launch=0)——先确认请求带非零 penalty |
| 毒源枯竭 | 有效实验下三条探针全阴:token_ids_cpu 无 -1 残留 |
| 毒源仍在-静默污染 | POISON RESIDENT 命中但无故障行——**比崩更糟**,需重新立项 |
| 毒源仍在-已崩溃 | POISON 命中且捕获疑似故障行(EZ9999/aivec/illegal GM 等) |

识别的日志行(匹配不变子串,容忍任意前缀与全角变体):新版/旧版
`sampler-probe`(bincount launch / OUT-OF-RANGE / penalties ACTIVE /
wire penalties)、`spec-probe`(WRITE INTO PROMPT REGION / POISON
RESIDENT)、`chunk align profile adopted`、INJECT window mismatch、
prompt chunk offset mismatch。

编码自适应(硬性要求):BOM 判定 → 无 BOM 时 NUL 奇偶启发式(UTF-16)
→ utf-8/gbk/big5 回退链;`errors="replace"` 永不因解码失败崩溃;
逐行 `rstrip("\n\r")` 容忍混合行尾;`--encoding` 强制指定。

配套测试:`tests/v1/lwd_control/test_lwd_aivec_analyzer.py`
(编码矩阵 6 例 + 判定矩阵 7 例 + JSON/乱码/路由回归 3 例)。

## 相关工具

- `tools/lwd_log_triage.py`:拉取链路四点探针(T0/E0/E1/T1)全链路配对分析。
