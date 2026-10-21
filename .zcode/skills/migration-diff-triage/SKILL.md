---
name: migration-diff-triage
description: 跨分支迁移对比排障法：当代码是从旧分支迁移/重构到新底座（或 issue 列出迁移断点），用"能跑的旧分支 vs 迁移目标分支"逐点对照定位断裂并最小修复。Use whenever 用户报告迁移后跑不起来、旧分支没这个问题、是不是迁移遗漏/漏迁、断点核对、迁移现状审计、为什么去掉了 X 还是报错，或提到 mm 分支、prefill_only、迁移断点、旧分支对照、PR 未合入等。涵盖四类断裂定性（漏件/重写出错/合并事故/有意分期）、grep 对照命令模板、与未合入 PR 同构的最小修复、AST 守卫测试。
---

# 跨分支迁移对比排障（migration-diff-triage）

方法核心：**不做全量 diff，从报错点出发，沿启动/请求时间轴逐断点在两个分支上对照同一段代码**。重构式迁移必然产生四类断裂，先定性再修，定性决定修法。

## 第 0 步：定基线对

- `old` = 已知能跑的分支（精确提交），`new` = 报错的分支（精确提交）。从 issue/PR 描述、仓内 AGENTS.md、报错元数据里的版本指纹（commit hash 如 `g869b32b7a`、依赖 pin 如 `transformers==5.5.3`）锁定——指纹对不上哪条线，就说明报错来自哪个源。
- 两边 fetch 后直接查远端跟踪分支，不必 checkout：`git grep <sym> <remote>/<branch> -- <path>`、`git show <remote>/<branch>:<file>`。
- gitcode.com 的 git 操作必须绕代理：`git -c http.proxy= -c https.proxy= fetch ...`（本机 Clash 会掐断 gitcode 的 TLS）。

## 四类断裂的识别与验证

### A 漏件（最常见）

- **整包未迁**：`git ls-tree --name-only <old> -- <dir>/` 有而 new 无；消费方 import 残留在 new（`git grep "import <pkg>" <new> -- vllm/`）→ 首次触达即 ModuleNotFoundError。修法：整包移植，或在唯一收口处 no-op 降级（真包落地自动恢复）。
- **上游文件身上的增改未随迁（最阴险）**：增量不是长在自有包里，而是加在**上游文件**上（outputs.py 里加的类、forward context 动态挂的字段、envs 白名单加的名字、`__init__.py` 的 re-export）。迁移"搬自有包 + 小改上游"时这类全靠人工 diff，漏掉不报错直到运行时。验证：符号在 old 的上游文件里 `git grep <sym> <old> -- <上游文件>` 命中、new 同文件不命中，而 new 的消费方仍在 import/读取。

### B 重写出错

- 特征：old 的同函数里**不存在** new 出问题的中间变量/结构（说明这段是按新架构重写的），且该路径在 new 上从未被执行过——一执行就 `UnboundLocalError`/语义覆盖。典型形态：old 是平铺互斥链（if/elif/else），new 改成前置 if/else 但后续默认块没收进 else，既炸作用域又会覆盖前面选好的值。
- 验证：并排读两版同函数（`git show <old>:<file> | sed -n` vs 工作区），确认差异是"重写"而非"漏搬"。

### C 合并事故

- 特征：同模块**同名顶层类/函数定义两次**（Python 后者遮蔽前者，前者的方法全部静默消失）；或两版代码粘进同一个文件。
- 通用扫描（一条命令扫全包，修前必跑）：

```bash
python3 - <<'EOF'
import ast, pathlib
for p in pathlib.Path("<目标包目录>").rglob("*.py"):
    t = ast.parse(p.read_text(encoding="utf-8")); seen = {}
    for n in t.body:
        if isinstance(n, ast.ClassDef): seen.setdefault(n.name, []).append(n.lineno)
    for k, v in seen.items():
        if len(v) > 1: print(p, k, v)
EOF
```

### D 有意分期

- issue/PR 明确排后的大块（如 executor 拉起层走 RFC）。**不算遗漏，不要自己发明修法**——去找那个未合入的官方 PR：`git fetch <remote> refs/pull/N/head && git log --oneline FETCH_HEAD`，读它的机制。

## grep 对照工具箱

| 目的 | 命令 |
|------|------|
| 符号双向存在性 | `git grep -n <sym> <old> -- <path>` vs `git grep -n <sym> <new> -- <path>`（消费方与定义处分开查） |
| 包完整性 | `git ls-tree -r --name-only <old> -- <pkg>/` 对照 new |
| 改造量指纹 | `git show <old>:<file> \| wc -l` vs 工作区 `wc -l`（差值即"新迁入的改造块"） |
| 线身份指纹 | 报错元数据里的版本号/依赖 pin 与各分支 requirements 对表 |
| 未合入 PR | `git fetch <remote> refs/pull/N/head:pr-N`；`git merge-base --is-ancestor pr-N <main>` 判断是否已合 |

## 机制推演定根因（用户硬性规范，见 ~/.zcode/AGENTS.md）

定性之后用时间轴三元组把根因钉死：每步【流程/事件 X】具体名（禁"某请求"式抽象）→【代码块 X】`file:line` + 关键行摘录 →【数据结构 X】状态快照；收尾必答两点：**事故点**在哪一行、**为什么无法自愈**（能纠正该状态的代码为何从此不可达）。推演同时预告时间轴上下一个必然炸点，让用户带着预期复测。

## 修复原则

1. **与官方将合入的 PR 同构优先**：把未合入 PR 的机制做精华移植（记录总行数与挂接点，PR 合入后删掉即可对齐）；绝不自创相反方向的分叉（如官方在 vllm 侧做、你却在 ascend 侧覆写）。
2. **修在 seam/收口处**：诊断 funnel 的 try/except 降级、`getattr(context, "field", False)` 默认值——默认值必须论证与 old 在当前部署下取值等价（例：字段只在某特性开启时才被 old 设置，当前部署该特性关闭）。
3. fail-fast 校验补在入口（越界/场景不符/数量不匹配直接 raise，别传到深处）。
4. 非侵入：能加在自有包/新函数里，就不 patch 上游文件；上游文件只允许"加一段收口钩子"级别的改动。
5. 每个修复独立 commit（英文 conventional + `fix(lwd):` 风格 + signoff），提交信息里报告测试结果。

## 守卫测试（每修复必配，同 commit）

1. **可复现**：测试对修复前源码必须为红——用 `git stash` 往返验证同一测试文件的红绿。
2. **AST/源码扫描类守卫**（stdlib-only，任何环境可跑）拦"事故类"而非仅本例：import 面可解析性（`find_spec`/源码扫描）、同名类遮蔽扫描、危险属性读法扫描（bare `ctx.field` → 必须经 getattr）。
3. **回归面**：修后跑周边测试目录；既有失败先 stash 验明是预置失败（与本次改动无关）再继续，并在提交信息注明。
4. 重依赖环境跑不了的用例：源码断言替代或 `pytest.skip` 注明补跑条件；测试按用户规范列清单供人工审查。

## 语境纪律（用户硬性规范）

- 默认用户环境/分支/构建/操作正确；"可能测错分支"只有结构指纹实锤时提**一次**，以证据形式呈现。
- 报错先分类：环境类（端口占用、依赖 pin、残留进程）与代码类分开，别把环境问题往迁移断点上套。

## 实战案例

完整 worked example（LWD vllm/vllm-ascend 迁移，2026-10，issue #18 七断点 + 实战新发现的签名漂移/零 KV 组/类遮蔽等共 10 类断点、环境类问题清单、修复-验证闭环）见 [references/lwd-case-2026-10.md](references/lwd-case-2026-10.md)。修复全部在两仓 `lwd-debug-1019` 分支：vllm `0ea4c94908`/`208e77a6af`/`8aadfce5d9`/`613f909baf`/`fe5a9394e5`，vllm-ascend `6acf01f07`/`7e6785a77`/`9f083a58f`。
