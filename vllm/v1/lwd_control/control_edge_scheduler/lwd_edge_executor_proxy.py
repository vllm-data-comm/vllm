"""边侧 executor 边界代理:统一承接 LWD 对执行器输出的三项契约。

  ① None 适配:底层"正常完成返回 None"(无输出批的默认形态)会被
     原生收割判为执行失败,经 NoneSafeFuture 替换为空占位;
  ② DISABLE_DOWN 诊断直通:旁路批不派 worker,BypassFuture 携带
     c2e 旁路 token 即结果,走与正常批一致的收割/交付路径
     (直通判定与行集归 lwd_control_diag,只出数据;future 包装归
     本代理,执行适配器不进诊断);
  ③ 透传:其余属性/调用经 __getattr__ 直达底层 executor。

stub/future 家族全部封为内部类,不向模块外泄漏形态——引擎与诊断
只认识本代理。
"""

from __future__ import annotations

from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag


class LwdEdgeExecutorProxy:
    """executor 边界代理(边引擎装配期包裹 model_executor)。"""

    class StubModelOutput:
        """伪 ModelRunnerOutput(仅 req_ids x sampled_token_ids):空占位
        与诊断旁路 token 共用的最小输出形态。"""

        __slots__ = ("req_ids", "sampled_token_ids")

        def __init__(self, req_ids, sampled_token_ids) -> None:
            self.req_ids = req_ids
            self.sampled_token_ids = sampled_token_ids

    class NoneSafeFuture:
        """future 适配:仅当底层正常完成且结果为 None 时替换为空占位;
        异常原样穿透,不吞执行失败。"""

        __slots__ = ("_fut",)

        def __init__(self, fut) -> None:
            self._fut = fut

        def result(self, timeout=None):
            result = self._fut.result(timeout)
            return (
                result
                if result is not None
                else LwdEdgeExecutorProxy.EMPTY_MODEL_OUTPUT
            )

        def __getattr__(self, name):
            return getattr(self._fut, name)

    class BypassFuture:
        """已完成的 future:诊断直通批不派 worker,旁路 token 即结果。"""

        __slots__ = ("_value",)

        def __init__(self, value) -> None:
            self._value = value

        def result(self, timeout=None):
            return self._value

    # 空占位(无输出批的 None 兜底);类级单例,收割路径按值判空即可
    EMPTY_MODEL_OUTPUT = StubModelOutput((), ())

    def __init__(self, inner) -> None:
        self._inner = inner

    def execute_model(self, scheduler_output, *args, **kwargs):
        bypass = LwdControlDiag.bypass_unembed_rows(
            getattr(scheduler_output, "lwd_batch", None)
        )
        if bypass is not None:
            req_ids, sampled = bypass
            return self.BypassFuture(self.StubModelOutput(req_ids, sampled))
        return self.NoneSafeFuture(
            self._inner.execute_model(scheduler_output, *args, **kwargs)
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)
