# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefill chunk 对齐 profile 的构建与采纳(边云 chunk 划分复现契约的
补全件)。

背景(issue #18 后续):边侧是 chunk 划分权威,云侧靠钳制参数复现批内
边界;但云侧原生 schedule() 还有一组边侧无法本地推导的截断分支——
GDN 混合模型的 mamba 块对齐(`_mamba_block_aligned_split`,边侧 kv
cache 无 mamba 组)与多模组 MTP 的 prefill lookahead 预留(边侧无
speculative 配置)。两侧 profile 不一致时云侧会静默改切 chunk,边侧
按原尺寸发 embeds,云侧 INJECT 窗口校验崩溃(实录:chunk rows=1561
!= scheduled=1536)。

修法:云侧把调度器实况属性经 LwdCloudParamsNotify 下发,边侧原样
采纳到自己的调度器上——边侧原生 schedule() 跑的就是上游同一段
split 代码,零逻辑复制,上游改动天然跟随。

本模块保持纯逻辑(不 import 消息结构体,鸭子类型读属性),任何解释器
可测;结构体字段与本模块属性表的一致性由测试 AST 守卫冻结。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.config import CacheConfig

try:  # 轻量环境可测:logger 链可用则打点,不可用则静默
    from vllm.lwd_logger import lwd_logger
except Exception:  # pragma: no cover
    lwd_logger = None  # type: ignore[assignment]


class LwdChunkAlignSource(Protocol):
    """profile 读侧协议:消息结构体与测试 stub 共同满足。"""

    @property
    def block_size(self) -> int: ...
    @property
    def hash_block_size(self) -> int: ...
    @property
    def need_mamba_block_aligned_split(self) -> bool: ...
    @property
    def mamba_prefill_checkpoint_alignment(self) -> int | None: ...
    @property
    def mamba_has_prefill_checkpoint_blocks(self) -> bool: ...
    @property
    def mamba_partial_cache_hit(self) -> bool: ...
    @property
    def mamba_fine_grained_prefix_cache(self) -> bool: ...
    @property
    def use_eagle_block_drop(self) -> bool: ...
    @property
    def num_prefill_lookahead(self) -> int: ...


# 边侧无合法本地来源、直接采纳的调度器属性(与 `_mamba_block_aligned_split`
# 和 `_reserve_prefill_lookahead` 运行期读取的 self 属性一一对应)。
LWD_CHUNK_ALIGN_ADOPT_ATTRS: tuple[str, ...] = (
    "need_mamba_block_aligned_split",
    "mamba_prefill_checkpoint_alignment",
    "mamba_has_prefill_checkpoint_blocks",
    "mamba_partial_cache_hit",
    "mamba_fine_grained_prefix_cache",
    "use_eagle_block_drop",
    "num_prefill_lookahead",
)


def lwd_build_chunk_align_profile(scheduler, cache_config: CacheConfig):
    """云侧填充:从调度器实况属性构建 profile(单一事实源,不做二次推导)。

    block_size 取 cache_config(与 `_mamba_block_aligned_split` 的读取
    口径一致),其余取调度器 init 派生属性。
    """
    from vllm.v1.lwd_control.lwd_comm.lwd_messages import LwdChunkAlignProfile

    return LwdChunkAlignProfile(
        block_size=cache_config.block_size,
        hash_block_size=scheduler.hash_block_size,
        **{name: getattr(scheduler, name) for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS},
    )


def lwd_apply_chunk_align_profile(
    scheduler, cache_config: CacheConfig, profile: LwdChunkAlignSource
) -> None:
    """边侧采纳:校验两侧块尺寸一致后,把 profile 属性盖到调度器上。

    采纳发生在引擎启动期(首个 prefill 批派出前),此后边侧原生
    schedule() 对 chunk 的截断与云侧逐步一致。块尺寸校验 fail-fast:
    尺寸有本地来源(两侧同为平台默认推导),不一致即部署错位,静默
    采纳只会让每一次复现都错。
    """
    if profile.block_size != cache_config.block_size:
        raise RuntimeError(
            f"[LWD] edge/cloud block size mismatch: edge="
            f"{cache_config.block_size} cloud={profile.block_size}; the "
            "chunk alignment profile assumes equal block sizes, start both "
            "sides with the same value"
        )
    if profile.hash_block_size != scheduler.hash_block_size:
        raise RuntimeError(
            f"[LWD] edge/cloud hash block size mismatch: edge="
            f"{scheduler.hash_block_size} cloud={profile.hash_block_size}; "
            "start both sides with the same value"
        )
    for name in LWD_CHUNK_ALIGN_ADOPT_ATTRS:
        setattr(scheduler, name, getattr(profile, name))
    if lwd_logger is not None:
        lwd_logger.info(
            "[cloud-params] chunk align profile adopted: block=%d hash=%d "
            "mamba_split=%s eagle_drop=%s lookahead=%d",
            profile.block_size,
            profile.hash_block_size,
            profile.need_mamba_block_aligned_split,
            profile.use_eagle_block_drop,
            profile.num_prefill_lookahead,
        )
