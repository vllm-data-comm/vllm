# lwd_control 架构加深重构设计（v3 检视稿）

| 项 | 值 |
| --- | --- |
| 基线 | `vllm` 仓 `prefill_only_chunked_prefill_mm` 分支 @ `86d12a0c5` |
| 范围 | `vllm/v1/lwd_control/` 包 + `vllm/config/lwd.py` + core.py/serve.py 接入点 |
| v2 约束（用户拍板） | **只保留 PD_mix；PD 分离（prefill_first/decode_first 相位机）整链删除** |
| v3 约束（用户拍板） | **代码越少越好**；借鉴 `prefill_only_refact_scheduler` 分支（PD 分离时代重构）的可适用手法 |
| 状态 | v3 已实现：包内 3497 → 2680 行，测试 35 通过 1 跳过 |

## v3 变更记录（在 v2 之上，五批落地）

- **A 基类改进**：预算钳制与 chunk 阈值折进 `_lwd_schedule_for_visible_reqs`（两侧手搓 save/restore 删除）；隐藏期名额扣减防超发（纵深）；`_lwd_is_decode` 内联为 FINISHED 先判；采样参数字段表（边组包/云建请求共用，加字段只改一处）。
- **B 日志删减**：每步/每批/每请求 INFO 与 perf 探针删除（27 处），保留 fail-fast、HOLD/OPEN、defer 原因、purge、一次性装配日志；`LwdDebug.ENABLED` 默认 False 且热路径先判开关再做 tokenizer 解码/finish 重算。
- **C docstring 契约化**：调度器与引擎的叙事性注释压缩为契约句（45 处），不变量与 fail-fast 条件保留。
- **D 配置上提**：传输层字段并入 `vllm/config/lwd.py` 的 LwdConfig（`additional_config` 段 + legacy 段回退 + `VLLM_ASCEND_LWD_*` env 覆盖、坏值告警）；`lwd_control/lwd_config.py` 删除；serve 守卫删除（部署校验内联云引擎 `_lwd_setup_zmq` 头部，后续已随 `_lwd_cloud_deploy_guard` 删除合并）；core.py 内联 12 行引擎选择；包 `__init__` 只剩诊断 helper。
- **E 引擎基类**：`LwdBaseEngineCore`（调度器注入时序、chunked prefill 恢复、两面生命周期、shutdown）；`Subscriber.start(handler)` 起接收线程；云引擎 PRE_OUT 线程改构造期启动（删 `process_input_sockets` 覆写，socket 仍在 IO 线程内建保 zmq 亲和）；门池 gate_pending 改 seen 集合消息直投（删 `_lwd_promote`）。
- **c2e 收编（2026-10-08 二次评估后落地，沿 refact 分支形态）**：此前缓议的顾虑（update_from_output 拿不到 engine_core_outputs）不成立——原生 `super().update_from_output()` 的返回值即 engine_core_outputs，finish 码来源不变。落地：调度器 `update_from_output` 解 carrier 组 c2e 入 `pending_c2e`（纯记账不做 I/O），引擎 `post_step` 原生钩子统一冲刷发边；core.py 删除 `lwd_process_model_output`/`lwd_handle_model_output` 钩子链、两处 [Lwd][trace] 与 'start send' 调试日志，LWD 残留仅剩引擎选择块（10 行）。云引擎 451→330 行，mixed 373→488（承接搬迁），lwd_debug 改指调度器静态方法。
- **J awaiting 生命周期迁移（2026-10-08，用户拍板）**：嵌入完结不再 finish 出调度器，请求留原生 running 转 decode 相位（与云侧收集谓词同款）；`_lwd_awaiting` 删除，inflight/max_seqs 改读 len(running)；迟到结果判定改原生 requests 成员检查；终结改云端完成码映射（_LWD_FINISH_STATUS）走原生 finish_requests（KV/缓存释放与 finished_req_ids 交接白拿）。本轮原生步进 + 自定义收割 + wire 携带 finish_reasons 三项前提齐备后，历史风险主体（token 走原生入账）不存在。pending 闸门两本账（first_token/batch_requests）不随迁，需协议携带消费 seqno 才能去除。冒烟重点：长驻 decode 相位请求在 AsyncScheduler 旁路（preemption 统计等）的行为。
- **K 变量/接口盘点清理（2026-10-09，用户拍板"改一改"）**：删 6 项（Publisher.endpoint 与 Communicator.endpoint 只写不读属性、LwdHelloNotify.wire_version 字段与 LWD_WIRE_VERSION 常量（唯一读者已随防御清除删）、LwdRangeNotify 顶层 request_id/offset/num_tokens 三字段与边侧填充（wire 上每条通告重复发首条目两遍的死重；连带 purge legacy 分支）、spec _lwd_spec_drafts 计数与 mean_accept_len 指标）；合 4 项（5000 超时双份收进 subscriber 常量、空占位与旁路 DTO 合一为 _LwdStubModelOutput、_lwd_notify_items 内联、lwd_edge_max_num_seqs_check 内联）；_LWD_ENV_PREFIX 双份保留（合并需包 __init__ 引 vllm.config，拉起 py3.9 不兼容链、破坏 picker 纯模块可测性，得不偿失）。wire 变更两项（HELLO 少 1 字段、RangeNotify 少 3 字段）为同版本部署内安全（msgspec 按字段名，ascend 对两结构体零代码引用）。过程修复两处前轮静默伤：_lwd_verify_full_admission 删除时孤儿 @staticmethod 落在 _schedule_impl 头上（真机首步即 TypeError）、classify 调用内联替换缩进未匹配调用已删方法——已修并全包扫描同类（无）。教训：索引式/字面替换必须带命中断言。
- **I 防御性编程清除（2026-10-08，用户拍板：开发阶段保证不报错）**：删除自洽断言与不可能状态守卫——picker 预算断言、unembed 长度断言、边侧 chunk 步后核验、云侧准入核验与 offset/mrope 对账循环、wire_version 互校、HELLO legacy peer 分支、通知 items 旧版单请求回退（两处）、云侧重复元数据守卫、边侧 publisher None 守卫、_lwd_hello_seen 日志去重标志。保留：部署校验（master_addr/pre_out_host/死锁配置/三元组互裁）、协议竞态语义（classify unknown/stale、abort 清扫、迟到结果丢弃）、请求边界校验。风险须知：chunk 契约漂移（上游原生调度演进）从当场报错变为静默错算，回归防线后移到 CI/联调。
- **H awaiting 台账并入调度器（2026-10-08，用户拍板删文件）**：三本账（awaiting 集/首 token 集/seqno→批内请求）内联为 LwdEdgeScheduler 私有字段，转换逻辑为调度器私有段（语义按用户已回退的原语义版；I5a 后于同日修复，见 §8）；`lwd_edge_awaiting_tracker.py` 与其测试删除，生命周期转换的独立单测面随之消失，靠真机冒烟兜底；时间戳账（无人读）一并去除。
- **F 步进原生化（评估后改道，2026-10-08）**：目标是 refact 分支的"边引擎步进全走父类 step_with_batch_queue"。读原生收割段后确认三处硬冲突：① EMBED 批会落原生 `sample_tokens` 采样分支（边侧 embed 不采样）；② DISABLE_DOWN 的 `future=None` 直通无法进原生批队列（原生只收真 future，合成假 future 即 refact 分支 total=0 事故形状）；③ awaiting 请求已 finish 出调度器，原生 `update_from_output` 无从入账。refact 分支能走通是因它同时换了生命周期模型（请求留 running + 占位欠条）。**落地改为安全瘦身**：批队列元组去只写不读的时间戳、两个单行体派发方法内联，边引擎 360 → 351。全量原生化与生命周期迁移绑定，列为独立立项。奥林边侧六项改动（pending 闸门/HELLO 互裁/mrope 切片/encoder 绕过/DISABLE_DOWN 直通/chunk 化）均不在本次触碰面或已显式保留。

以下为 v2 正文（行号对应 v2 状态，仅供追溯）。

---

v1 → v2 的方案变化：删除相位机后，云侧只剩一个调度器，v1 里的三块设计随之消失——选型工厂 `lwd_cloud_scheduler_cls()`、`is_mixed` 字段、云侧公共基类 `lwd_cloud_base.py`（它们的存在理由是 mixed/phase 二选一与去重，现在只剩一者）。新增模块从 5 个减为 4 个，删除文件从 1 个增为 2 个。

---

## 1 问题清单（量化）

| 编号 | 证据 | 量化 |
| --- | --- | --- |
| P1 | 云侧 2 文件横穿 import 边侧包取共享配置：`lwd_cloud_engine.py:36`、`lwd_cloud_phase_scheduler.py:56` | `lwd_edge_assemble.py` 157 行中共享配置 100%、边装配 0% |
| P2 | 调度器选型判据 4 种拼写（`__init__.py:62-66`、`lwd_cloud_engine.py:80-90`、`lwd_edge_scheduler.py:139-142`、`lwd_cloud_engine.py:88`） | 同一事实 4 处实现 |
| P3 | chunk 组批 picker 读 8 个 `self` 属性（`lwd_edge_scheduler.py:215-333`），测试须构造完整 AsyncScheduler | 119 行；注释记录 3 起真机事故；`deepseek_v4` 与 `8000` 硬编码 |
| P4 | IO 线程与调度主线程经裸属性互通（`lwd_cloud_engine.py:243、299-327、133-134`；防御性身份弹队 `lwd_cloud_scheduler.py:307-309、431-435`） | 并发契约散在 2 文件 5 处注释 |
| P5 | ~~mixed/phase 两调度器 5 段复写~~ | **v2 已消**：phase 整体删除，去重对象不复存在；仅 `_lwd_spec_factor` 推导三处复写保留为待收敛项 |
| P6 | 边侧 awaiting 生命周期摊在三本字典、五个方法 | 摘除循环手抄两份；闸门读裸表示 |
| P7 | 诊断开关 3 读点 2 helper，违反包自立的 env 纪律 | 同一变量裸读两遍 |
| P8 | `lwd_seqno_registry` 全仓无读者，注释宣称的 `SO.lwd_up_seqnos` 不存在 | 死代码 3 触点 |
| P9（v2 新增） | PD 分离相位机整链：`lwd_cloud_phase_scheduler.py`（235 行）、判型四处、边侧 `_lwd_mixed` 逃生通道分支、`_lwd_early_recv` 开关、`LwdConfig.scheduler_name` 字段 | 用户约束：全部删除 |

## 2 目标与非目标

**目标**：P1–P4、P6–P9（P5 因 phase 删除自动消解，spec factor 公式收进 `lwd_config.lwd_spec_factor()` 单点）。

**非目标**：

- wire 协议零改动（`lwd_notify.py` 与传输层不动）
- 外部 import 面零改动（§3 冻结表）；对照臂 `core/sched/phase_scheduler.py` 只继承 `LwdBaseScheduler`，不动
- 不动 vllm-ascend 仓；诊断开关 env 变量名与 `=="1"` 判定语义不变
- 不抽 SPI/ABC（无第二实现场景）
- 不做性能优化，不改任何 `[Lwd][*]` 日志行字段与格式（phase 分支独有的日志随删除消失）

## 3 上游一致性评估

`lwd_control` 整包下游自有（git 历史全为 `feat(lwd)/refactor(lwd)` 提交），上游 vLLM 无此目录，本设计上游 diff 面增量 0。外部 import 冻结表：

| 外部文件 | import | 守住手段 |
| --- | --- | --- |
| `entrypoints/cli/serve.py:186` | `from vllm.v1.lwd_control import lwd_serve_guard` | 入口函数签名不动 |
| `v1/engine/core.py:1206` | `from vllm.v1.lwd_control import lwd_resolve_engine_cls` | 同上 |
| `v1/core/sched/phase_scheduler.py:24` | `from ...control_scheduler.lwd_base_scheduler import LwdBaseScheduler` | `LwdBaseScheduler` 位置与方法不动 |
| `v1/lwd_debug/lwd_debug.py:83` | `from ...lwd_cloud_engine import LwdCloudEngineCore`（用其静态方法 `_lwd_c2e_finish_reasons`） | 方法名与签名不动 |

对照臂 docstring 提及 `LwdCloudPhaseScheduler` 一句为历史注释，属外部文件，不为本设计改动。

## 4 机制 ↔ 类职责对照表（重构后）

LWD prefill_only 把推理拆到两个进程：边侧持有真实 prompt 只做 embedding，云侧拿占位缓冲跑 decode；控制面承载模式判定、两侧调度语义与通知协议，数据面在 vllm-ascend 经 `SchedulerOutput.lwd_batch` 接缝对接。

| 类/函数 | 机制 | 一句话职责 |
| --- | --- | --- |
| `LwdConfig` + `is_lwd_prefill_only` + `lwd_spec_factor` | 模式判定与公共参数 | env/config 唯一解析点，plain 值出厂 |
| `LwdBaseScheduler` | 可见集调度 | 三队列剔除-调度-拼回原语（两侧调度器与对照臂共用） |
| `LwdCloudScheduler` | 云侧调度（唯一形态） | notify 门控混排 + preflight + 双钳制复现 |
| `LwdPrefillNotifyQueue` | 云侧线程接缝 | IO 线程投递、主线程消费的预告通知队列 |
| `LwdEdgeScheduler` | 边侧调度 | 准入校验、闸门、步后核验、预告发布 |
| `lwd_pick_prefill_batch()` | 边侧组批算术 | 续传优先 + FCFS 前缀 + 末位截断纯函数 |
| `LwdEdgeAwaitingTracker` | 边侧结果生命周期 | awaiting 三账的登记/到账/摘除 |
| `LwdEdgeEngineCore` / `LwdCloudEngineCore` | 引擎装配 | 通信面、步进编排、引擎接口覆写 |

## 5 目标架构

```
lwd_control/
├── __init__.py            入口（serve 守卫 + 引擎类分流；无条件注入 Mixed）
├── lwd_config.py          ★新增：配置 + 模式判定 + spec factor 公式 + 诊断 helper
├── control_scheduler/lwd_base_scheduler.py      不动
├── control_communication/                        全部不动
├── control_edge_scheduler/
│   ├── lwd_edge_scheduler.py  瘦身（picker/ledger 外移，逃生通道删除）
│   ├── lwd_edge_picker.py     ★新增：组批纯函数
│   ├── lwd_edge_awaiting_tracker.py     ★新增：生命周期台账
│   └── lwd_edge_engine.py     诊断开关经 lwd_config 集中 helper
└── control_cloud_scheduler/
    ├── lwd_prefill_notify_queue.py     ★新增：预告通知队列
    ├── lwd_cloud_scheduler.py  唯一云调度器（inbox 接入）
    └── lwd_cloud_engine.py    无条件 Mixed；registry/散读删除
    （lwd_cloud_phase_scheduler.py、lwd_edge_assemble.py 删除）
```

依赖规则：`lwd_config` → 仅 communication 常量与 logger；picker/ledger → 仅 stdlib；**云侧包禁止 import 边侧包**（grep 把关）。

## 6 API 定义

### 6.1 `lwd_config.py`（约 175 行）——P1/P2/P7/P9

- `LwdConfig` 整体搬迁自 `lwd_edge_assemble.py`；**删除 `scheduler_name` 字段**，段内残留 `scheduler` 键告警后忽略（PD 分离已删，PD_mix 唯一形态）
- 新增 `lwd_diag_token_passthrough_enabled()` / `lwd_diag_disable_down_enabled()`（调用点现读 env，与原实现同时机；严格 `== "1"`，定义单点收在 lwd_config）
- 新增 `lwd_spec_factor(vllm_config) -> int`：`1 + num_speculative_tokens` 公式唯一住处（边 cap、云 preflight、HELLO 三处复写收敛）
- `is_lwd_prefill_only` 逐字搬迁

调用点改写：`__init__.py` 与云引擎的判型段删除，无条件注入 `LwdCloudScheduler`；`_lwd_early_recv` 删除（提前收恒开，hint MQ 缺失降为恒告警）；边调度器 `_lwd_mixed` 与逃生通道分支删除；两引擎诊断 helper 收编为 lwd_config 的 lwd_diag_*（仍按调用点现读，语义同时机）。

### 6.2 `lwd_edge_picker.py`（约 160 行）——P3/P9

```python
class LwdPrefillCandidate(NamedTuple):
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int

def lwd_pick_prefill_batch(
    running: Sequence[LwdPrefillCandidate], waiting: Sequence[LwdPrefillCandidate],
    *, cap: int, free_seq: int, chunk_unit: int,
    single_prefill_min_tokens: int = 0,   # >0：总长达到该值的请求独占整批
) -> tuple[list[str], dict[str, int]]
```

命名沿用原实现词汇：候选（原 `candidates` 变量）、`single_prefill`（原独占规则变量名）、`chunk_unit`（原 `_lwd_chunk_unit`）。内部 `_append_chunk / _pick_resumes / _pick_new_requests` 三函数逐句翻译原两个循环（含 single_prefill 标志跨循环传递、末位截断、free_seq 门）；预算断言随行。调度器侧缩为约 40 行适配层（候选组装 + cap/free_seq 计算 + deepseek_v4 → `single_prefill_min_tokens=8000`），逃生通道分支不复存在。

### 6.3 `lwd_edge_awaiting_tracker.py`（约 105 行）——P6

```python
class LwdEdgeAwaitingTracker:
    is_awaiting(rid) / awaiting_count() / pending_chunks()
    register_batch(seqno, rids) / embed_done(rids, now)
    mark_first_token(rid) / finish(rid) / abort(rid)
```

两份手抄扫批循环合一为 `_drop_consumed_batches`；调度器五个触碰点各改一行调用。

### 6.4 `lwd_prefill_notify_queue.py`（约 80 行）——P4/P8

```python
class LwdPrefillNotifyQueue:   # 一把 threading.Lock + deque
    append(notify)            # IO 线程
    purge(request_id) -> int  # IO 线程（abort 清扫，条目级摘除）
    peek() -> notify | None   # 主线程
    pop_front(notify) -> bool # 主线程（按对象身份弹队首；phase 独有的塞回路径随删除消失）
    __len__()
```

类名与属性名沿用原 `prefill_notify_queue`；引擎借 `self.scheduler.prefill_notify_queue` 引用投递/清扫。引擎侧删 `_lwd_purge_prefill_notify`（29 行）与 `_lwd_seqno_registry` 三触点（P8 死代码）；队列由 mixed 调度器自建（IO 线程晚于构造启动，无竞态窗口）。

## 7 重构前后职责映射（无遗漏）

| 旧 | 新 | 性质 |
| --- | --- | --- |
| `lwd_edge_assemble` 全部配置解析 | `lwd_config`（减 `scheduler_name`） | 搬家 |
| 判型四处（guard/云引擎/边调度/early_recv） | 删除；guard 与云引擎无条件注入 Mixed | 删除（P9/P2） |
| `LwdCloudPhaseScheduler` 235 行 | 删除 | 删除（P9） |
| 边侧 `_lwd_mixed` 逃生通道分支 | 删除 | 删除（P9） |
| 两引擎诊断 helper 与裸读 | `lwd_config.lwd_diag_*` 集中 helper（调用点现读） | 合一 |
| mixed 的 `prefill_notify_queue` + 身份弹队 ×2 + 引擎 purge 29 行 | `LwdPrefillNotifyQueue` | 对象化/合一 |
| `lwd_seqno_registry` 三触点 | 删除 | 死代码（P8） |
| `_lwd_spec_factor` 三处复写 | `lwd_config.lwd_spec_factor` | 合一 |
| `_lwd_pick_prefill_batch` 119 行主体 | `lwd_pick_prefill_batch` + 30 行适配层 | 拆分 |
| 边侧三账五触碰点 | `LwdEdgeAwaitingTracker` | 拆分/合一 |
| 其余全部（notify 协议、传输层、可见集机制、preflight、步后核验、引擎步进、HELLO 互校、mrope 计算、spec 观测） | 原地不动 | — |

## 8 行为不变量与验证

| # | 不变量 | 验证 |
| --- | --- | --- |
| I1 | wire 协议字节零变化 | 改动集不含 notify/传输层 |
| I2 | PD_mix 路径行为逐字节不变（选型、组批、闸门、preflight、双钳制、日志行） | 单测 + 真机冒烟比对 |
| I3 | 诊断开关严格 `== "1"`，env 名不变，调用点现读时机不变，vllm-ascend 侧不动 | `test_lwd_config` |
| I4 | picker 输出与原实现逐请求一致（续传序、FCFS、末位截断、独占规则、Σ≤cap） | `test_lwd_edge_picker` 表驱动（含边界：cap<1、c=0 不入批、free_seq=0、single_prefill 跨循环传递） |
| I5 | 台账语义：迟到幂等丢弃、abort 三账齐摘、pending_chunks 定义不变、批全到齐即移除 | `test_lwd_edge_awaiting_tracker` |
| I5a | 已修复（2026-10-08，移植 aolin PR #10）：原语义下首 token 即终结的请求被摘出首 token 集后，其所在批若仍有未到账成员将永久留存——bench 长跑泄漏累积使 pending 水位只涨不降，prefill 闸门永久 HOLD（4 卡 + MTP 实测：云侧空转刷 c2e_meta=None）。修复方式：finish 路径复用 abort 的摘账逻辑（`_lwd_purge_request_ledger`：摘首 token 集 + 各批剔除该请求，批摘空或只剩已见者整条移除），与"finish 视同已消费"等价防漏且与 abort 语义对齐；tracker 内联时该行为曾按用户决定回退为原语义并视作良性，PR #10 证实为死锁，予以修复 | 代码评审 + 真机冒烟（长跑 bench 观察闸门不再永久 HOLD） |
| I6 | inbox 次序语义：FIFO、条目级 purge、purge 与 peek 并发安全 | `test_lwd_prefill_notify_queue`（含双线程烟雾） |
| I7 | 外部 import 面零变化；`lwd_edge_assemble`/`lwd_cloud_phase_scheduler` 全仓零引用 | `test_lwd_public_surface` + grep |
| I8 | registry 删除无行为影响 | 全仓 grep 证据 |
| I9 | 残留 `scheduler` 配置键：告警不拦截（PD_mix 唯一形态） | `test_lwd_config` |
| I10 | 引擎构造次序约束不变（scheduler_cls 先于 super、IO 线程晚于构造） | 代码评审 + 冒烟 |

测试位于 `vllm/tests/v1/lwd_control/`：`test_lwd_config.py`、`test_lwd_public_surface.py`、`test_lwd_edge_picker.py`、`test_lwd_edge_awaiting_tracker.py`、`test_lwd_prefill_notify_queue.py`（纯 CPU；本机 python3.9 无 torch，重量级 import 以 `find_spec` 断言替代）。

## 9 迁移与回滚

| 阶段 | 内容 | 问题 | 回滚 |
| --- | --- | --- | --- |
| A | 删相位机：`lwd_cloud_phase_scheduler.py` 删除、判型四处无条件化、边侧逃生通道删除、`scheduler_name` 字段删除 | P9 | 单 commit revert |
| B | `lwd_config.py` 归位 + 诊断开关 + spec factor 合一，删 `lwd_edge_assemble.py` | P1/P2/P7 | 单 commit revert |
| C | `LwdEdgeAwaitingTracker` 台账 | P6 | 单 commit revert |
| D | picker 抽取 | P3 | 单 commit revert |
| E | `LwdPrefillNotifyQueue` + registry 死代码删除 | P4/P8 | 单 commit revert |

本次检视版为五阶段一次性落盘（未提交 git），检视通过后可按上表拆 commit 合入。

## 10 风险

| 风险 | 对策 |
| --- | --- |
| 现网仍有 `scheduler: prefill_first` 部署 | 残留键告警；拉起前 grep 配置目录 |
| picker 搬运手误 | 表驱动测试覆盖原注释记录的事故形态（截断漏记、超批） |
| inbox 锁时序 | 临界区无 IO、每步 O(1)；purge 仅 abort |
| 对照臂 docstring 提及已删类 | 外部文件不动，历史注释保留 |
