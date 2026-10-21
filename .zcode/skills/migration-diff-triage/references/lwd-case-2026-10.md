# LWD 迁移断点实战案例（2026-10，vllm/vllm-ascend）

背景：`prefill_only_chunked_prefill_mm`（gitcode aolin，能跑）→ `vllm-data-comm` 两仓 main（迁移目标）。审计 issue：vllm-data-comm/vllm#18。修复分支：两仓 `lwd-debug-1019`。

## 断点定性总表（issue #18 七断点 + 实战新发现）

| # | 现象 | 定性 | 对照证据（old → new） | 修复 |
|---|------|------|----------------------|------|
| 1 | `UnboundLocalError: hardware_profile`（配置期） | **B 重写出错** | old `platform.py:660` 平铺互斥链、默认分支写死类名；new 按 hardware_profile 架构重写且从未被 LWD 路径执行 | xlite/默认块收进非 LWD else（vllm-ascend `6acf01f07`，AST 守卫） |
| 2 | `torch world size 1/8 != hccl_world_size` | **D 有意分期** | old executor 1325 行（rank 偏移+MQ）vs new 1127 行；world 覆盖在 old `parallel.py:836` | 精华移植未合入 PR #14 的会合机制：tp_rank→YAML 全局 rank + wire store（vllm `0ea4c94908`） |
| 3 | 云侧首前向 `AttributeError: flash_comm_v1_enabled` | **A 漏件（适配层）** | 字段由 old `ascend_forward_context.py:127` 动态挂（仅 SP 时 True），未随迁 | 两处 `getattr(..., False)`——无 SP 部署下与 old 取值严格等价（vllm-ascend `7e6785a77`，AST 守卫） |
| 4 | 首请求 `ModuleNotFoundError: vllm.v1.lwd_debug` | **A 漏件（整包）** | old 7 文件 ~300 行；new 无包但收口仍 import | 收口处 try/except 降级 no-op，真包落地自动恢复（vllm `208e77a6af`） |
| 5 | 云侧闸门永不放行 prefill | **D 有意分期**（与 2 同块） | old executor `:311-316` 建 hint/done MQ；new 无 | 未修，与 2 同属 RFC 范围 |
| 6 | `ImportError: LwdC2eMeta` | **A 漏件（上游文件增改）** | old `outputs.py:232` 定义；new 无而云调度器仍 import | 按 old 原定义补回（vllm `208e77a6af`，源码守卫） |
| 7 | V4 hash 路由死代码 | A 漏件（白名单） | `_EXTRA_CTX` 缺字段 | 不涉当前部署，未修 |
| 8 | 云侧 MTP proposer `AttributeError: _ensure_query_start_loc_arange_capacity` | **C 合并事故** | 同文件两个同名 `AscendSpecDecodeBaseProposer`（114 残段被 198 完整版遮蔽，三方法静默消失）；全包 AST 扫描仅此一处 | 合并两类体（vllm-ascend `9f083a58f`，"同名顶层类"通用守卫） |
| 9 | 边侧 `AssertionError: Prefix caching requires at least one cacheable KV cache group` | **A 变种（上游同型机制未覆盖新场景）** | 上游 `vllm/config/vllm.py:1990` 对纯 encoder 实例配置期关 APC；LWD 边同情形（零 KV 组）未被覆盖 | 引擎期关 APC（`LwdEdgeEngineCore.__init__`，vllm `7d2f27d329`→`613f909baf`；先配置期后按非侵入原则挪到自有包，守卫同时钉住"上游文件不得出现 LWD 分支"） |
| 10 | 首请求双侧 `TypeError: schedule() takes 1 positional argument` | **A 变种（上游签名演进未随迁）** | 原生 `schedule(self, throttle_prefills=False)`（`scheduler.py:562`）；LWD 两调度器仍 mm 无参签名 | 基类模板方法收口签名一次（`LwdBaseScheduler.schedule` → `_lwd_schedule_step` 钩子，vllm `6b0da8b10f`→`fe5a9394e5`，签名兼容守卫） |

## 环境/语义类（勿与迁移断点混淆）

- `--headless` 语义误用：headless=引擎等**外部 LB 监督者**回应启动握手（`run_headless` 无进程 bind 握手地址），非"不开前端"→ 固定 5 分钟超时。修=普通 serve + 闲置 API 端口（生成器 `8aadfce5d9`）。
- 混合 GDN 模型 + 手动 `--no-enable-prefix-caching` + 显式 `--mamba-block-size`：触发 main 新增校验（mm 时代无）。修=脚本删参，APC 关闭交给引擎期（与断点 9 的引擎期方案互补）。
- 云物理卡 RoCE 网口 down（`EI0009 network port is down`）：hccn_tool 三步查光模块/IP/lldp。
- 端口占用（Errno 98）、手改脚本丢续行符致参数截断（traceback 无 LWD 帧即证据）。

## 方法要点回顾

1. 每个断点都是"报错栈 file:line → 两分支并排读同段 → 定性 → 按定性修法"；定性决定修法（漏件=移植/降级，重写=复刻 old 语义于新结构，合并事故=合并+通用守卫，分期=找官方 PR 精华移植）。
2. 断点 2 的机制推演（流程→代码块→数据结构三元组）确认：去 PP 只改"组切割"，"世界组建"三件套（world 覆盖/rank 偏移/共享 store）缺一必炸于 world size 断言，且无自愈路径。
3. 修复顺序跟着启动时间轴走：配置期(1)→世界组建(2)→前端握手(headless)→worker 初始化(8)→调度器构造(9)→首请求调度(10)→首前向(3)→prefill 派发(5)。每修一个预告下一个卡点，用户复测日志验证预判——修复-验证闭环。
4. 非侵入优先：断点 9 首版落配置期（上游文件），用户指出后挪进自有包引擎期——守卫把"上游文件无 LWD 分支"固化成 CI 规则。
5. 每修复配守卫：可复现（stash 红绿）+ 拦"事故类"（同名类遮蔽、签名漂移、import 面可解析、危险属性读法）而非仅盯本例。
