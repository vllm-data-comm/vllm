# LWD 边云控制面原生化重构 · 设计文档（v4 定稿）

| 项 | 值 |
| --- | --- |
| 基线 | `86d12a0c5`（2026-09-29，3497 行 / 12 文件） |
| 落地 | `5b607068d`（v2 删相位机）→ `c00cac309`（v3 原生化，27 文件 +1410/−1995） |
| 现状 | 2490 行 / 15 文件，净减 1007 行（−29%）；测试 29 用例全绿 |
| 分支 | `prefill_only_ww_1009_refact`（vllm + vllm-ascend 双仓同名，已推 origin） |
| 配套 | 流程差异 `lwd_control_flow_diff.md` · 变更台账 v3 A–K `lwd_control_deepening_design.md` |

---

## 1 · 背景与动机

LWD prefill_only 把一次推理拆到边云两个进程：边侧持有真实 prompt 只做 embedding，云侧拿占位缓冲跑 decode，控制面走 ZMQ（PRE_OUT 边→云，POST_OUT 云→边），数据面张量走 vllm-ascend 的 UP/DOWN 通道按 seqno 配对。

重构前的问题可以收在一个词里：偏离原生。边侧引擎覆写 step/has_work 自建批队列流水线，请求在嵌入完结时被 finish 出调度器、由三本手工台账代管生命周期；云侧 c2e 发布挂在 core.py 的模型输出钩子链上，每步带调试日志。原生已有的机制（批队列、收割、入账、生命周期），我们几乎都维护着一份平行实现——09-22 一天四场真机事故（超批 2×cap、608≠604、280 步静默串行、全程串行）全部出在这些平行实现的接缝上。

重构目标：批产出与收割统一收口原生主路径；请求生命周期回归原生调度器；删防御性编程与调试残留；PD 分离相位机按拍板整链删除。一句话，原生有的用原生，自己写的每一行都要有原生给不了的理由。

## 2 · 两条硬约束（一切设计的出发点）

理解本方案只需记住两条：

约束一：张量在预告后面飞。边侧发布 RangeNotify 的同一拍就发出 embedding 张量，worker 按 seqno 预挂接收。云侧排程量必须逐请求等于预告量，错一行，张力就注入错窗口，静默错算。这决定了云侧不能让原生自由排 prefill，必须精确复现边侧的 chunk 划分。

约束二：一步只产一个 SchedulerOutput。没有先填再回读的窗口。这决定了 decode 的精确消耗必须预先推导、预告必须当预算用。

## 3 · 云侧方案：动了手脚的原生调度

### 3.1 每步决策流程（`schedule`:431 → `_schedule_impl`:322）

```
收集 decode 候选(判据:prompt 已算完且未终结)
        ↓
peek 队首预告 ──无──→ decode-only:可见集隔离调用(不钳制)
        ↓ 有
条目对账(账本有此请求且未算完?)  ──ADD 未落地──→ 整批 defer(留队首)
        ↓ ready
提前收闸门(水位 ≥ seqno?)      ──没收完──────→ defer
        ↓ 收完
preflight 三检(token/名额/KV)   ──不过────────→ defer
        ↓ 通过
混排步 _schedule_mixed
```

### 3.2 混排步三件事（`_schedule_mixed`:322）

1. decode_exact 逐字复刻原生 num_new 推导（:339-359），含 max_tokens 守卫与节拍守卫的跳过分支。多算 4 个 token 余量，末位截断块就被超排（真机实录 604→608）。
2. 一次隔离调用（:360 → 基类 `_lwd_schedule_for_visible_reqs`）：把 decode_ids + 预告条目请求做成可见集，带双钳制——`chunk_threshold = max(items)`（复现非末位块边界）、`token_budget_cap = decode_exact + Σitems`（预算耗尽复现末位截断块）——藏起其余请求、按隐藏数扣减名额，然后调 `super().schedule()`。批内每请求排多少 token 是原生代码自己算的，我们只是把它能看到的世界改小、读的配置改紧，使它的结论恰好落在预告的数字上。
3. 消费与挂批（:368-377）：按对象身份 pop 预告，挂 `LwdBatch(EMBED, seqno)`，batch_meta 透传 offset 与 mrope 字节给 worker 注入。

### 3.3 请求处理的两段式

`LwdRequestNotify`（元数据）→ IO 线程字段表建 Request（占位零值 embeds + 边侧哈希链）直投 `input_queue(ADD)` 落地 waiting，此刻存在但不可算；`LwdRangeNotify`（范围）→ 通知队列，四道门放行后驱动该请求的 prefill 排程，与 UP 张量按 `(request_id, offset, seqno)` 配对。此后请求是普通原生 decode 公民。

### 3.4 c2e 回流：记账与发布分离

调度器 `update_from_output` 在原生入账后解 worker 的 pinned carrier，组 `LwdC2eNotify`（完成码取自 engine_core_outputs，finished 缺口 ABORT 兜底）入 `pending_c2e`，纯记账不做 IO；引擎 `post_step` 原生钩子统一冲刷。core.py 的模型输出钩子链与每步 trace 日志全删，LWD 残留仅剩 10 行引擎选择块。

## 4 · 边侧方案：零覆写原生步进

### 4.1 三项构造期适配（引擎 `__init__`）

1. 批队列深度恒 4（size 与 maxlen 同步重建，防 appendleft 在满队静默挤掉在飞批）；
2. `is_pooling_model = True`，该属性全仓仅一个消费点（core.py 采样分支），置掉它即禁掉原生 sample_tokens 补发，边侧采样在 execute 内完成；
3. executor 代理。底层重写后"正常完成返回 None"的默认形态会被原生判为执行失败，代理把 None 适配为空占位（异常原样穿透）；DISABLE_DOWN 诊断的 UNEMBED 批在此拦截为携带旁路 token 的已完成 future，走与正常批一致的原生收割路径。

### 4.2 谓词保证空批不可能产生（`has_requests`:139）

`has_requests` 覆写为"有可派发的批"（通告队列非空 / EMBED 可派）。闸门 HOLD 且无在飞时谓词为 False，引擎经原生 `has_work` 直接睡眠。schedule 被调用即必有真批，这是原生批队列的隐含契约，空批会以 None 结果炸收割路径（即原 None 断言错误的根因）。

### 4.3 schedule 双批产出（`schedule`:149）

EMBED 优先（防被 decode 回程挤出深度配额）→ 两道门（满员 / pending chunk < 3）+ 续传豁免 → picker 纯函数组批 → 预算钳制的隔离调用 → 发布成功才挂批（seqno 与发布绑定）；否则消费 c2e 通告出 UNEMBED 批，与 EMBED 走同一条原生队列。

### 4.4 生命周期回归原生（awaiting 迁移）

嵌入完结不再 finish，请求留 running 转 decode 相位（`num_computed == num_prompt`，与云侧收集谓词同款），原生 `self.requests` 成为唯一事实源：inflight 一处读数 `len(running)`、迟到结果按账本成员判定、终结由云端完成码经 `_LWD_FINISH_STATUS` 映射走原生 `finish_requests`（KV/缓存释放与 finished_req_ids 交接白拿）。pending 闸门两本账（首 token 集 / seqno→批）保留，服务云侧消费水位，与请求住哪无关。

### 4.5 收割统一（`update_from_output`:220）

返回值即步输出：EMBED 批登记进度（完结者转 decode 相位）；UNEMBED 批按 worker 回填的 req_ids × sampled_token_ids 逐请求交付；迟到/已终结幂等丢弃；占位批 no-op。

## 5 · 公共基础设施

| 设施 | 内容 |
| --- | --- |
| `LwdBaseEngineCore`（40 行） | 调度器注入先于 super（一次性消费）、chunked prefill 恢复（0 KV 组模型的上游禁用对边侧 chunk 记账不适用）、两面生命周期 |
| `Subscriber.start(handler)` | 接收线程骨架上提，云侧保留自管循环保 zmq 线程亲和 |
| `LwdPrefillNotifyQueue`（72 行） | 一把锁持有 IO 线程/主循环的线程纪律（append/peek/pop_front/purge），替代散在注释里的契约 |
| `lwd_edge_picker`（161 行） | 组批纯函数（续传优先/FCFS 前缀/末位截断/独占规则），17 用例表驱动 + 20000 组与基线 oracle 对拍零分歧 |
| `LWD_SAMPLING_FIELDS` | 采样参数一张表边组包/云建请求共用 |
| 配置上提 `vllm/config/lwd.py` | 搭 additional_config 解析 + env 覆盖坏值告警；serve 守卫删除，部署校验内联 `_lwd_setup_zmq` 头部 |

## 6 · 重构前后对照

| 维度 | 前 | 后 |
| --- | --- | --- |
| 边侧步进 | 覆写 step/has_work + 自建批队列 | 零覆写 + 三适配 |
| 边侧生命周期 | finish 出调度器 + 三本台账 | 留原生 running，requests 唯一事实源 |
| 边侧批产出 | schedule 只产 EMBED，UNEMBED 引擎直投 | schedule 统一产出，同一原生队列 |
| 云侧 c2e | core.py 钩子链 + 每步日志 | 调度器记账 + post_step 冲刷，core.py 零钩子 |
| 云侧调度器 | mixed + phase 两套，判型四处 | 唯一 LwdCloudScheduler |
| 请求落地 | 门池两跳 | 直投 input_queue |
| 防御核验 | chunk 核验/offset 对账/准入核验/wire 互校 | 全删（拍板），防线移 CI/联调 |
| 日志 | 47 处 INFO + LwdDebug 恒开 | 删 27 处；LwdDebug 默认关且先判开关 |
| 测试 | 0 | 29 用例 |

不变的部分：wire 协议零字节变化（除删 4 个死字段）；chunk 算术（对拍证明）；四道门语义；双钳制机制；HELLO 三元组互裁；部署校验；mrope 全链路；WAKEUP 语义。

## 7 · 功能一致性证明

| 手段 | 内容 |
| --- | --- |
| picker 对拍 | 基线 oracle（逐字转录）× 现实现，20000 组随机（含 8000 阈值两侧/65536 长 prompt/cap 0-16384/free_seq 0-4/deepseek 开关），零分歧 |
| 表驱动边界 | 17 用例锁死截断/名额/独占/续传序 |
| 接口守卫 | 云不 import 边（扫描+测试）、core.py 零 LWD 钩子、已删模块 find_spec 为 None |
| 语义等价迁移 | 步进/收割/生命周期的每条路径在流程差异文档 §2-§5 逐项对账 |
| 有意偏差清单 | ① 防御核验删除（拍板）② wire 删 4 死字段（同版本部署内安全，ascend 零代码引用）③ 空步 finished_req_ids 交付滞后（基线固有） |

## 8 · 风险与冒烟清单

1. 划分漂移静默化（最大风险）：上游原生调度演进时不再当场报错，冒烟必跑 65536 长序列 + 多模态混发，抽日志比对逐请求排程量与预告量。
2. 长驻 decode 相位请求：AsyncScheduler 旁路（preemption 统计等）对其行为需真机确认（参考分支同形态已验证）。
3. finished_req_ids 时序：终结改收割时原生 finish，随下一个 EMBED 批 SO 交接 worker，滞后与基线一致，看 worker 缓存释放日志。
4. 本机 3.9 无 torch：config 用例 1 跳过，真实环境补跑；29 用例在真实环境应全绿。
5. 过程教训（已入台账）：索引式/字面替换的三次静默伤（孤儿 @staticmethod、_is_decode 残留、缩进未命中）均已修复并全包扫描，后续机械改动必须带命中断言。

## 9 · 后续候选（未做）

- pending 闸门两本账去除：c2e 携带云侧已消费 seqno（协议 +1 int），边侧 pending = 已发布 − 已消费。
- 规范整改三拍板项：注释语言、日志前缀 [LWD] 统一、诊断开关初始化期读取（当前现读与规范 7.9 冲突）。
- RFC issue 建立并关联 MR（超 500 LOC 阈值，标准要求）。
