# -*- coding: utf-8 -*-
"""
极简 LT 编解码（RSD + Systematic + 剥皮译码）
[CRC 校验版 + Meta 包集成]
- LTPacket 携带: seed (4B) + crc32 (4B) + payload
- seed 编码 sys_idx: seed = BASE_SEED ^ sys_idx (系统包)
- 每个包都有 CRC32 校验，解码时自动丢弃损坏包
- 集成 Meta 包生成和解析
"""
from __future__ import annotations

import math
import random
import struct
import zlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ================ Robust Soliton 分布 ================
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


# ================ Meta 包（心跳包）================
@dataclass
class MetaPacket:
    """
    心跳包/Meta 包，携带全局参数
    总计 24 字节
    """
    k: int  # 4 字节: 源块总数
    block_size: int  # 4 字节: 每块大小
    msg_len: int  # 4 字节: 原始消息长度
    base_seed: int  # 4 字节: 基础种子
    msg_crc: int  # 4 字节: 消息全局 CRC32
    meta_crc: int  # 4 字节: Meta 包自身的 CRC32（校验前 5 个字段）


def create_meta_packet(
        k: int,
        block_size: int,
        msg_len: int,
        base_seed: int,
        msg_crc: int
) -> MetaPacket:
    """创建 Meta 包并计算其 CRC"""
    # 前 5 个字段打包
    header = struct.pack('>5I', k, block_size, msg_len, base_seed, msg_crc)
    # 计算 Meta 包自身的 CRC
    meta_crc = zlib.crc32(header) & 0xFFFFFFFF

    return MetaPacket(
        k=k,
        block_size=block_size,
        msg_len=msg_len,
        base_seed=base_seed,
        msg_crc=msg_crc,
        meta_crc=meta_crc
    )


def serialize_meta_packet(meta: MetaPacket) -> bytes:
    """序列化 Meta 包为 24 字节"""
    return struct.pack(
        '>6I',  # 6 个 unsigned int (big-endian)
        meta.k,
        meta.block_size,
        meta.msg_len,
        meta.base_seed,
        meta.msg_crc,
        meta.meta_crc
    )


def deserialize_meta_packet(data: bytes) -> MetaPacket:
    """反序列化 Meta 包，并验证 CRC"""
    if len(data) != 24:
        raise ValueError(f"Meta 包长度错误: 期望 24 字节, 实际 {len(data)} 字节")

    k, block_size, msg_len, base_seed, msg_crc, meta_crc = struct.unpack('>6I', data)

    # 验证 Meta 包的 CRC
    header = struct.pack('>5I', k, block_size, msg_len, base_seed, msg_crc)
    expected_crc = zlib.crc32(header) & 0xFFFFFFFF

    if expected_crc != meta_crc:
        raise ValueError(f"Meta 包 CRC 校验失败: 期望 {expected_crc:08x}, 实际 {meta_crc:08x}")

    return MetaPacket(
        k=k,
        block_size=block_size,
        msg_len=msg_len,
        base_seed=base_seed,
        msg_crc=msg_crc,
        meta_crc=meta_crc
    )


# ================ 数据包 ================
@dataclass
class LTPacket:
    seed: int  # 4 字节: 系统包 = BASE_SEED ^ sys_idx；冗余包 = base_seed + pkt_id
    crc32: int  # 4 字节: payload 的 CRC32 校验值
    payload: bytes  # block_size 字节


def serialize_lt_packet(pkt: LTPacket) -> bytes:
    """序列化 LT 包为字节流: seed(4B) + crc32(4B) + payload"""
    header = struct.pack('>2I', pkt.seed, pkt.crc32)  # 8 字节
    return header + pkt.payload


def deserialize_lt_packet(data: bytes, block_size: int) -> LTPacket:
    """反序列化 LT 包"""
    if len(data) < 8:
        raise ValueError(f"数据包太短: {len(data)} 字节")

    seed, crc32 = struct.unpack('>2I', data[:8])
    payload = data[8:8 + block_size]

    if len(payload) != block_size:
        raise ValueError(f"Payload 长度错误: 期望 {block_size}, 实际 {len(payload)}")

    return LTPacket(seed=seed, crc32=crc32, payload=payload)


# ================ 编码器 ================
class LTEncoder:
    def __init__(
            self,
            message: bytes,
            block_size: int = 512,
            base_seed: int = 0x12345678,  # 基础种子，要远离 [0, k) 范围
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

        # 用于解码端验证最终消息
        self.msg_crc = zlib.crc32(message) & 0xFFFFFFFF

    def _neighbors_from_seed(self, seed: int) -> List[int]:
        rng = random.Random(seed)
        d = sample_degree(rng, self.cdf)
        # 采样 d 个不同的索引
        return rng.sample(range(self.k), d)

    def next_packet(self) -> LTPacket:
        """生成下一个 LT 包（systematic 在前 k 个包送原始块）。"""
        if self.systematic and self.next_id < self.k:
            # 系统包：用 seed 编码 sys_idx
            idx = self.next_id
            payload = bytes(self.blocks[idx])
            seed = self.base_seed ^ idx  # XOR 编码
            crc = zlib.crc32(payload) & 0xFFFFFFFF

            pkt = LTPacket(
                seed=seed,
                crc32=crc,
                payload=payload,
            )
        else:
            # 冗余包：用 seed 生成邻居
            seed = (self.base_seed + self.next_id) & 0xFFFFFFFF
            neigh = self._neighbors_from_seed(seed)
            acc = np.zeros(self.block_size, dtype=np.uint8)
            for i in neigh:
                acc ^= self.blocks[i]
            payload = bytes(acc)
            crc = zlib.crc32(payload) & 0xFFFFFFFF

            pkt = LTPacket(
                seed=seed,
                crc32=crc,
                payload=payload,
            )
        self.next_id += 1
        return pkt

    def get_meta_packet(self) -> MetaPacket:
        """生成 Meta 包（应在传输开始前发送）"""
        return create_meta_packet(
            k=self.k,
            block_size=self.block_size,
            msg_len=len(self.msg),
            base_seed=self.base_seed,
            msg_crc=self.msg_crc
        )


# ================ 解码器 ================
class LTDecoder:
    def __init__(self, c: float = 0.1, delta: float = 0.5):
        self.c = float(c)
        self.delta = float(delta)

        # 状态标记
        self.initialized = False

        # 这些参数需要通过 set_params 显式设置
        self.k = 0
        self.block_size = 0
        self.msg_len = 0
        self.base_seed = 0
        self.msg_crc = 0  # 可选的全局消息 CRC

        self.cdf: Optional[np.ndarray] = None
        self.known: Optional[np.ndarray] = None  # (k, block_size) uint8
        self.known_mask: Optional[np.ndarray] = None  # (k,) bool
        self.eqs: list = []  # list[(set(indices), payload np.uint8[block_size])]

        # 统计信息
        self.packets_received = 0
        self.packets_crc_failed = 0
        self.packets_duplicate = 0

    def set_params(
            self,
            k: int,
            block_size: int,
            msg_len: int,
            base_seed: int = 0x12345678,
            msg_crc: Optional[int] = None
    ):
        """
        显式初始化解码器参数。
        通常在收到 Meta 包后调用此方法。
        """
        if self.initialized:
            # 如果参数没变，直接返回；如果参数变了，重置
            if (self.k == k and self.block_size == block_size and
                    self.msg_len == msg_len and self.base_seed == base_seed):
                return
            else:
                self.reset()

        self.initialized = True
        self.k = k
        self.block_size = block_size
        self.msg_len = msg_len
        self.base_seed = base_seed
        self.msg_crc = msg_crc or 0

        self.cdf = robust_soliton_cdf(k, self.c, self.delta)
        self.known = np.zeros((k, block_size), dtype=np.uint8)
        self.known_mask = np.zeros(k, dtype=bool)
        self.eqs = []

    def set_params_from_meta(self, meta: MetaPacket) -> None:
        """从 Meta 包初始化参数"""
        self.set_params(
            k=meta.k,
            block_size=meta.block_size,
            msg_len=meta.msg_len,
            base_seed=meta.base_seed,
            msg_crc=meta.msg_crc
        )

    def reset(self):
        """重置解码器状态"""
        self.initialized = False
        self.known = None
        self.known_mask = None
        self.eqs = []
        self.packets_received = 0
        self.packets_crc_failed = 0
        self.packets_duplicate = 0

    def _neighbors_from_seed(self, seed: int) -> List[int]:
        rng = random.Random(seed)
        d = sample_degree(rng, self.cdf)  # type: ignore[arg-type]
        return rng.sample(range(self.k), d)

    def add_packet(self, pkt: LTPacket) -> None:
        """
        添加一个数据包。
        注意：必须先调用 set_params() 或 set_params_from_meta() 初始化！
        """
        # 如果还没有初始化（没收到心跳包），直接忽略数据包
        if not self.initialized:
            return

        assert self.known is not None and self.known_mask is not None

        # 1. CRC 校验
        actual_crc = zlib.crc32(pkt.payload) & 0xFFFFFFFF
        if actual_crc != pkt.crc32:
            self.packets_crc_failed += 1
            return  # CRC 校验失败，丢弃此包

        # 2. 检查 Payload 长度
        if len(pkt.payload) != self.block_size:
            return

        self.packets_received += 1

        # 3. 判断是系统包还是冗余包
        possible_sys_idx = pkt.seed ^ self.base_seed

        if 0 <= possible_sys_idx < self.k:
            # 系统包：直接揭示该块
            idx = possible_sys_idx
            if not self.known_mask[idx]:
                self.known[idx] = np.frombuffer(pkt.payload, dtype=np.uint8)
                self.known_mask[idx] = True
                self._peel()
            else:
                self.packets_duplicate += 1
            return

        # 4. 冗余包：构造方程
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
            self.packets_duplicate += 1
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

    def reconstruct(self, verify_crc: bool = True) -> bytes:
        """
        重建消息。
        verify_crc: 是否验证全局消息 CRC（需要在 set_params 时提供 msg_crc）
        """
        assert self.is_decoded(), "not enough packets to decode"
        assert self.known is not None
        buf = self.known.reshape(-1)[: self.msg_len]
        result = bytes(buf)

        # 可选的全局 CRC 验证
        if verify_crc and self.msg_crc != 0:
            actual_crc = zlib.crc32(result) & 0xFFFFFFFF
            if actual_crc != self.msg_crc:
                raise ValueError(f"消息 CRC 校验失败: 期望 {self.msg_crc:08x}, 实际 {actual_crc:08x}")

        return result

    def get_stats(self) -> dict:
        """返回统计信息"""
        return {
            "packets_received": self.packets_received,
            "packets_crc_failed": self.packets_crc_failed,
            "packets_duplicate": self.packets_duplicate,
            "blocks_decoded": int(self.known_mask.sum()) if self.known_mask is not None else 0,
            "total_blocks": self.k,
            "progress": f"{int(self.known_mask.sum())}/{self.k}" if self.known_mask is not None else "0/0",
        }


# ================ 使用示例 ================
if __name__ == "__main__":
    print("=" * 60)
    print("喷泉码编解码测试（带 CRC 校验 + Meta 包）")
    print("=" * 60)

    # ============ 编码端 ============
    message = b"Hello, Fountain Code! " * 50  # 约 1.1 KB
    print(f"\n原始消息长度: {len(message)} 字节")

    encoder = LTEncoder(message, block_size=512, base_seed=0x12345678)
    print(f"源块数 k = {encoder.k}")

    # 1. 生成并序列化 Meta 包
    meta = encoder.get_meta_packet()
    meta_bytes = serialize_meta_packet(meta)
    print(f"\nMeta 包: {len(meta_bytes)} 字节")
    print(f"  k={meta.k}, block_size={meta.block_size}, msg_len={meta.msg_len}")
    print(f"  base_seed=0x{meta.base_seed:08x}, msg_crc=0x{meta.msg_crc:08x}")

    # 2. 生成并序列化数据包
    redundancy = 1.2  # 冗余率
    num_packets = int(encoder.k * redundancy)
    packets_bytes = []

    print(f"\n生成 {num_packets} 个数据包（冗余率 {redundancy}）...")
    for i in range(num_packets):
        pkt = encoder.next_packet()
        pkt_bytes = serialize_lt_packet(pkt)
        packets_bytes.append(pkt_bytes)

    print(f"已生成 {len(packets_bytes)} 个数据包")
    print(f"每个数据包大小: {len(packets_bytes[0])} 字节")

    # ============ 解码端 ============
    decoder = LTDecoder()

    # 1. 接收并解析 Meta 包
    print(f"\n解码端接收 Meta 包...")
    meta_recv = deserialize_meta_packet(meta_bytes)
    decoder.set_params_from_meta(meta_recv)
    print(f"解码器已初始化: k={decoder.k}, block_size={decoder.block_size}")

    # 2. 接收数据包
    print(f"\n开始接收数据包...")
    for i, pkt_bytes in enumerate(packets_bytes):
        pkt = deserialize_lt_packet(pkt_bytes, decoder.block_size)
        decoder.add_packet(pkt)

        # 每 10 个包打印一次进度
        if (i + 1) % 10 == 0 or decoder.is_decoded():
            stats = decoder.get_stats()
            print(f"  已接收 {i + 1}/{len(packets_bytes)} 包 | "
                  f"已解码块: {stats['progress']} | "
                  f"CRC失败: {stats['packets_crc_failed']} | "
                  f"重复: {stats['packets_duplicate']}")

        if decoder.is_decoded():
            print(f"\n✓ 解码完成！共接收 {i + 1} 个包")
            break

    # 3. 重建并验证
    if decoder.is_decoded():
        result = decoder.reconstruct(verify_crc=True)
        print(f"\n解码成功！")
        print(f"  重建消息长度: {len(result)} 字节")
        print(f"  消息匹配: {result == message}")

        stats = decoder.get_stats()
        print(f"\n统计信息:")
        print(f"  总共接收: {stats['packets_received']} 个有效包")
        print(f"  CRC 失败: {stats['packets_crc_failed']} 个")
        print(f"  重复包: {stats['packets_duplicate']} 个")
        print(f"  解码进度: {stats['progress']}")
    else:
        print("\n✗ 解码失败，包数不足")

    print("\n" + "=" * 60)