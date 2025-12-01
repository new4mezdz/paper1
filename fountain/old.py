# -*- coding: utf-8 -*-
"""
极简 LT 编解码（RSD + Systematic + 剥皮译码）
- 每个编码包携带：{k, block_size, msg_len, seed 或 sys_idx, payload}
- 非系统包的邻居集合由 seed + RSD 在译码端可复现，无需显式传输度和邻居列表。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------- Robust Soliton（稳健小 k 处理） ----------------
def robust_soliton_cdf(k: int, c: float = 0.1, delta: float = 0.5) -> np.ndarray:
    """
    返回度分布 μ(d) 的累计分布（索引 0..k，其中 cdf[d] 表示 1..d 的累计）。
    对小 k 做了健壮处理：
      - k == 1 时，总是采样 d=1；
      - 对 t 做截断，避免越界。
    """
    if k == 1:
        cdf = np.zeros(2, dtype=np.float64)  # 0..1
        cdf[1] = 1.0
        return cdf

    assert k >= 2, "k must be >= 2"
    R = c * math.log(max(k / delta, 2.0)) * math.sqrt(k)

    # Ideal soliton ρ(d)
    rho = np.zeros(k + 1, dtype=np.float64)  # 0..k
    rho[1] = 1.0 / k
    for d in range(2, k + 1):
        rho[d] = 1.0 / (d * (d - 1))

    # τ(d)
    tau = np.zeros(k + 1, dtype=np.float64)
    if R > 0:
        t = int(math.floor(k / R))
        # 安全截断到 [1, k]
        if t > k:
            t = k
        if t >= 1:
            for d in range(1, t):
                tau[d] = R / (d * k)
            tau[t] = (R * math.log(max(R, 2.0) / delta)) / k

    mu = rho + tau
    Z = mu[1:].sum()
    mu /= Z
    cdf = np.cumsum(mu)
    return cdf


def sample_degree(rng: random.Random, cdf: np.ndarray) -> int:
    """根据累计分布 cdf 采样度 d ∈ [1..k]。"""
    u = rng.random()
    lo, hi = 1, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] >= u:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------- 数据结构 ----------------
@dataclass
class LTPacket:
    k: int
    block_size: int
    msg_len: int
    seed: Optional[int]       # 非系统包用 seed
    sys_idx: Optional[int]    # 系统包用索引
    payload: bytes            # 长度 == block_size


class LTEncoder:
    def __init__(
        self,
        message: bytes,
        block_size: int = 512,
        base_seed: int = 1234,
        c: float = 0.1,
        delta: float = 0.5,
        systematic: bool = True,
    ):
        self.msg = message
        self.block_size = int(block_size)
        self.base_seed = int(base_seed)
        self.c = float(c)
        self.delta = float(delta)
        self.systematic = systematic

        # 切分源块；空消息也至少 1 块
        self.k = max(1, math.ceil(len(message) / self.block_size))
        padded_len = self.k * self.block_size
        buf = bytearray(padded_len)
        buf[: len(message)] = message
        self.blocks = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(self.k, self.block_size)

        self.cdf = robust_soliton_cdf(self.k, self.c, self.delta)
        self.next_id = 0  # 递增的包序

    def _neighbors_from_seed(self, seed: int) -> List[int]:
        rng = random.Random(seed)
        d = sample_degree(rng, self.cdf)
        # 采样 d 个不同的索引
        return rng.sample(range(self.k), d)

    def next_packet(self) -> LTPacket:
        """生成下一个 LT 包（systematic 在前 k 个包送原始块）。"""
        if self.systematic and self.next_id < self.k:
            idx = self.next_id
            payload = bytes(self.blocks[idx])
            pkt = LTPacket(
                k=self.k,
                block_size=self.block_size,
                msg_len=len(self.msg),
                seed=None,
                sys_idx=idx,
                payload=payload,
            )
        else:
            seed = (self.base_seed + self.next_id) & 0xFFFFFFFF
            neigh = self._neighbors_from_seed(seed)
            acc = np.zeros(self.block_size, dtype=np.uint8)
            for i in neigh:
                acc ^= self.blocks[i]
            pkt = LTPacket(
                k=self.k,
                block_size=self.block_size,
                msg_len=len(self.msg),
                seed=seed,
                sys_idx=None,
                payload=bytes(acc),
            )
        self.next_id += 1
        return pkt


class LTDecoder:
    def __init__(self, c: float = 0.1, delta: float = 0.5):
        self.c = float(c)
        self.delta = float(delta)
        self.initialized = False
        self.k = 0
        self.block_size = 0
        self.msg_len = 0
        self.cdf: Optional[np.ndarray] = None
        self.known: Optional[np.ndarray] = None      # (k, block_size) uint8
        self.known_mask: Optional[np.ndarray] = None # (k,) bool
        self.eqs: list = []                          # list[(set(indices), payload np.uint8[block_size])]

    def _ensure_init(self, k: int, block_size: int, msg_len: int):
        if self.initialized:
            return
        self.initialized = True
        self.k = k
        self.block_size = block_size
        self.msg_len = msg_len
        self.cdf = robust_soliton_cdf(k, self.c, self.delta)
        self.known = np.zeros((k, block_size), dtype=np.uint8)
        self.known_mask = np.zeros(k, dtype=bool)

    def _neighbors_from_seed(self, seed: int) -> List[int]:
        rng = random.Random(seed)
        d = sample_degree(rng, self.cdf)  # type: ignore[arg-type]
        return rng.sample(range(self.k), d)

    def add_packet(self, pkt: LTPacket) -> None:
        self._ensure_init(pkt.k, pkt.block_size, pkt.msg_len)
        assert self.known is not None and self.known_mask is not None

        if pkt.sys_idx is not None:
            # 系统块，直接揭示该块
            idx = pkt.sys_idx
            if not self.known_mask[idx]:
                self.known[idx] = np.frombuffer(pkt.payload, dtype=np.uint8)
                self.known_mask[idx] = True
            self._peel()
            return

        # 非系统包：构造方程
        assert pkt.seed is not None
        neigh = set(self._neighbors_from_seed(pkt.seed))
        payload = np.frombuffer(pkt.payload, dtype=np.uint8).copy()

        # 扣掉已知块贡献
        rem = []
        for i in neigh:
            if self.known_mask[i]:
                payload ^= self.known[i]
                rem.append(i)
        for i in rem:
            neigh.discard(i)

        if len(neigh) == 0:
            # 全部扣完 -> 冗余
            return

        self.eqs.append((neigh, payload))
        self._peel()

    def _peel(self):
        """标准剥皮：不断寻找度为1的方程，推出未知块。"""
        assert self.known is not None and self.known_mask is not None
        progressed = True
        while progressed:
            progressed = False
            singles_idx = [ei for ei, (S, _) in enumerate(self.eqs) if len(S) == 1]
            if not singles_idx:
                break
            singles_idx.sort(reverse=True)
            for ei in singles_idx:
                S, payload = self.eqs.pop(ei)
                j = next(iter(S))
                if not self.known_mask[j]:
                    self.known[j] = payload
                    self.known_mask[j] = True
                    progressed = True
            if progressed and self.eqs:
                new_eqs = []
                for S, payload in self.eqs:
                    rem = [j for j in S if self.known_mask[j]]
                    for j in rem:
                        payload ^= self.known[j]
                        S.discard(j)
                    if len(S) > 0:
                        new_eqs.append((S, payload))
                self.eqs = new_eqs

    def is_decoded(self) -> bool:
        return self.initialized and self.known_mask is not None and bool(self.known_mask.all())

    def reconstruct(self) -> bytes:
        assert self.is_decoded(), "not enough packets to decode"
        assert self.known is not None
        buf = self.known.reshape(-1)[: self.msg_len]
        return bytes(buf)
