import torch
import triton
import triton.language as tl

from torch_mlu_ops.triton.utils import get_total_core_num
from torch_mlu_ops.triton.fla.index import prepare_chunk_indices
from torch_mlu_ops.triton.fla.utils import FLA_GDN_FIX_BT

# BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
BKV_LIST = [128]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        # for BK in BKV_LIST
        # for BV in BKV_LIST
        for num_warps in [1]
        for num_stages in [4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def tmo_chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT,
    chunk_num,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    id = tl.program_id(0)
    dim = tl.num_programs(0)
    rem = chunk_num % dim
    block = chunk_num // dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)
    ones_vec = tl.full((1, BT), 1, tl.float32)
    global_T = T
    for offset in range(block_start,block_start + block):
        i_b = offset % B
        i_h = offset // B % H
        i_t = offset // B // H % NT
    # i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_tg = i_t
            i_n, i_t = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
            )
            bos, eos = (
                tl.load(cu_seqlens + i_n).to(tl.int32),
                tl.load(cu_seqlens + i_n + 1).to(tl.int32),
            )
            T = eos - bos
        else:
            i_tg = i_b * NT + i_t
            bos, eos = i_b * T, i_b * T + T

    # if IS_VARLEN:
    #     i_tg = i_t
    #     i_n, i_t = (
    #         tl.load(chunk_indices + i_t * 2).to(tl.int32),
    #         tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
    #     )
    #     bos, eos = (
    #         tl.load(cu_seqlens + i_n).to(tl.int32),
    #         tl.load(cu_seqlens + i_n + 1).to(tl.int32),
    #     )
    #     T = eos - bos
    #     NT = tl.cdiv(T, BT)
    # else:
    #     NT = tl.cdiv(T, BT)
    #     i_tg = i_b * NT + i_t
    #     bos, eos = i_b * T, i_b * T + T

        # offset calculation
        q_offset = (bos * Hg + i_h // (H // Hg)) * K
        k_offset = (bos * Hg + i_h // (H // Hg)) * K
        v_offset = (bos * H + i_h) * V
        o_offset = (bos * H + i_h) * V
        h_offset = (i_tg * H + i_h).to(tl.int64) * K * V

        # b_o = tl.zeros([BT, BV], dtype=tl.float32)
        # b_A = tl.zeros([BT, BT], dtype=tl.float32)

        # for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + q_offset, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        p_k = tl.make_block_ptr(
            k + k_offset, (K, T), (1, Hg * K), (0, i_t * BT), (BK, BT), (0, 1)
        )
        if STATE_V_FIRST:
            p_h = tl.make_block_ptr(
                h + h_offset, (V, K), (K, 1), (0, 0), (BV, BK), (1, 0)
            )
        else:
            p_h = tl.make_block_ptr(
                h + h_offset, (K, V), (V, 1), (0, 0), (BK, BV), (1, 0)
            )
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        if STATE_V_FIRST:
            b_o = tl.dot(b_q, tl.trans(b_h), allow_tf32=ALLOW_TF32)
        else:
            b_o = tl.dot(b_q, b_h, allow_tf32=ALLOW_TF32)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A = tl.dot(b_q, b_k, allow_tf32=ALLOW_TF32)

        if USE_G:
            # g += bos * H + i_h
            p_g = tl.make_block_ptr(g + bos + i_h * B * global_T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_o = b_o * tl.exp(b_g)[:, None]
            # b_A = b_A * tl.exp(b_g[:, None] - b_g[None, :])
            b_A = b_A * tl.exp(tl.dot(b_g[:, None], ones_vec, allow_tf32=False) - b_g[None, :])
        # o_t = i_t * BT + tl.arange(0, BT)
        # m_t = o_t < T
        # m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        # b_A = tl.where(m_A, b_A, 0)

        p_v = tl.make_block_ptr(
            v + v_offset, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0)
        )
        p_o = tl.make_block_ptr(
            o + o_offset, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        b_A = b_A.to(b_v.dtype)
        b_A = tl.where(o_t[:, None] >= o_t[None, :], b_A, tl.cast(0, b_v.dtype))
        b_A = tl.where(m_t[:, None] & m_t, b_A, tl.cast(0, b_v.dtype))

        # to fix mma -> mma layout conversion
        # already solved by triton v3.2 or higher
        b_o = (b_o + tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=ALLOW_TF32)) * scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,  # cumsum of log decay
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    allow_tf32: bool = True,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = 64 if FLA_GDN_FIX_BT else min(chunk_size, max(16, triton.next_power_of_2(T)))
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = triton.cdiv(K,64) * 64
    BV = triton.cdiv(V,64) * 64
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)

    # def grid(meta):
    #     return (triton.cdiv(V, meta["BV"]), NT, B * H)
    # grid = (NT, B * H)
    chunk_num = B * H * NT
    grid_num = min(get_total_core_num(), chunk_num)
    grid = lambda meta: (grid_num, )

    tmo_chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        B=B,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
        chunk_num=chunk_num,
        ALLOW_TF32=allow_tf32,
        bottleneck="simd",
        STATE_V_FIRST=state_v_first,
    )
    return o

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_DW': lambda args: args['dw'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1]
        for num_stages in [1,2,3,4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW'],
)
@triton.jit(do_not_specialize=['T'])
def tmo_chunk_bwd_kernel_dqkwg(
    q,
    k,
    v,
    g,
    g_gamma,
    h,
    dout,
    dh,
    dq,
    dk,
    dw,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    B,
    T,
    NT,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    chunk_num,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_DW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    id = tl.program_id(0)
    dim = tl.num_programs(0)
    rem = chunk_num % dim
    block = chunk_num // dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)

    BT_ones_vec = tl.full((1, BT), 1, tl.float32)
    BK_ones_vec = tl.full((1, BK), 1, tl.float32)
    BV_ones_vec = tl.full((1, BV), 1, tl.float32)
    global_T = T
    for offset in range(block_start,block_start + block):
        # i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        # i_b, i_h = i_bh // H, i_bh % H
        # grid = (NK, NT, B, H)
        i_k = offset // (NT * B * H)
        i_t = offset // (B * H) % NT
        i_b = offset // H % B
        i_h = offset % H

        all = B * T
        if IS_VARLEN:
            i_tg = i_t
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
            # NT = tl.cdiv(T, BT)
        else:
            # NT = tl.cdiv(T, BT)
            i_tg = i_b * NT + i_t
            bos, eos = i_b * T, i_b * T + T

        # offset calculation
        q_offset  = (bos * H + i_h) * K
        k_offset  = (bos * H + i_h) * K
        v_offset  = (bos * H + i_h) * V
        do_offset = (bos * H + i_h) * V
        dq_offset = (bos * H + i_h) * K
        dk_offset = (bos * H + i_h) * K

        h_offset  = (i_tg * H + i_h).to(tl.int64) * K * V
        dh_offset = (i_tg * H + i_h).to(tl.int64) * K * V

        # for delta rule only
        if USE_DW:
            dw_offset = (bos * H + i_h) * K
            dv_offset = (bos * H + i_h) * V

        if USE_G:
            dg_offset = i_k * all * H
            b_dg_last = tl.zeros([1], dtype=tl.float32) if USE_G else None
        if USE_G_GAMMA:
            b_gamma = tl.load(g_gamma + i_h)
            b_g = b_gamma * (tl.arange(0, BT) + 1)
            b_g_last = b_gamma * min(BT, T - i_t * BT)


        if V == BV:
            p_v = tl.make_block_ptr(v + v_offset, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(dout + do_offset, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h + h_offset, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh + dh_offset, (V, K), (1, V), (0, 0), (BV, BK), (0, 1))

            # [BT, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))

            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            if USE_G:
                b_dg_last += (tl.sum(b_h * b_dh))

            # [BT, BV] @ [BV, BT] -> [BT, BT]
            b_ds = tl.dot(b_do, tl.trans(b_v), allow_tf32=ALLOW_TF32)
            # [BT, BV] @ [BV, BK] -> [BT, BK]
            b_dq = tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=ALLOW_TF32) * scale
            # [BT, BV] @ [BV, BK] -> [BT, BK]
            b_dk = tl.dot(b_v, b_dh.to(b_v.dtype), allow_tf32=ALLOW_TF32)
            if USE_DW:
                p_dv = tl.make_block_ptr(dv + dv_offset, (T, V), (H*V, 1), (i_t * BT, 0), (BT, BV), (1, 0))
                b_dv = tl.load(p_dv, boundary_check=(0, 1))
                b_dw = tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=ALLOW_TF32)
                p_dw = tl.make_block_ptr(dw + dw_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
                tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

            # tl.debug_barrier()
            p_q = tl.make_block_ptr(q + q_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + k_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            p_dq = tl.make_block_ptr(dq + dq_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + dk_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            # m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
            if USE_G:
                b_dg = tl.zeros([BT], dtype=tl.float32)
                # g += bos * H + i_h
                # dg += bos * H + i_h
                p_g = tl.make_block_ptr(g + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_last = tl.load(g + (bos + i_h * B * global_T) + (min(i_t * BT + BT, T) - 1))
                b_dg_last *= tl.exp(b_g_last)

                # b_dq = b_dq * (tl.exp(b_g)[:, None])
                b_dq = b_dq * tl.dot(tl.exp(b_g)[:, None], BK_ones_vec, allow_tf32=False)
                # b_dg += tl.sum(b_dq * b_q, axis=1)
                b_dg += tl.dot(BK_ones_vec, tl.trans(b_q * b_dq), allow_tf32=False)[0,:]

                # b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
                b_g_e_eff = tl.where(m_t, tl.exp(-b_g + b_g_last), 0)
                b_dk = b_dk * tl.dot(b_g_e_eff[:, None], BK_ones_vec, allow_tf32=False)
                b_dg -= tl.dot(BK_ones_vec, tl.trans(b_k * b_dk), allow_tf32=False)[0,:]
                # b_dg -= tl.sum(b_k * b_dk, axis=1)
                b_dg_last += tl.sum(b_dk * b_k)

                b_ds = tl.exp((tl.dot(b_g[:, None], BT_ones_vec, allow_tf32=False) - b_g[None, :])) * b_ds
                # b_ds = tl.where(m_A, b_ds, 0) * scale
                b_ds = tl.where(o_t[:, None] >= o_t[None, :], b_ds, 0)
                b_ds = tl.where(m_t[:, None] & m_t, b_ds, 0) * scale
                b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k), allow_tf32=ALLOW_TF32)
                # b_dg += tl.sum(b_ds2, axis=1)
                b_dg -= tl.sum(b_ds2, axis=0)
                b_dg += tl.dot(BT_ones_vec, tl.trans(b_ds2), allow_tf32=False)[0,:]

                b_ds = b_ds.to(b_k.dtype)
                # [BT, BK]
                b_dq += tl.dot(b_ds, b_k, allow_tf32=ALLOW_TF32)
                b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=ALLOW_TF32)
                p_dg = tl.make_block_ptr(dg + dg_offset + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
                # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
                # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last
                b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

            elif USE_G_GAMMA:
                b_dq = b_dq * tl.exp(b_g)[:, None] * scale
                b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
                b_ds = tl.where(m_A, b_ds * tl.exp(b_g[:, None] - b_g[None, :]), 0) * scale
                b_ds = b_ds.to(b_k.dtype)
                # [BT, BK]
                b_dq += tl.dot(b_ds, b_k, allow_tf32=ALLOW_TF32)
                b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=ALLOW_TF32)
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            else:
                b_ds = tl.where(m_A, b_ds, 0)
                b_ds = b_ds.to(b_k.dtype)
                b_dq += tl.dot(b_ds, b_k, allow_tf32=ALLOW_TF32)
                b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=ALLOW_TF32) * scale
                b_dq *= scale
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        else:
            b_dq = tl.zeros([BT, BK], dtype=tl.float32)
            b_dk = tl.zeros([BT, BK], dtype=tl.float32)
            b_ds = tl.zeros([BT, BT], dtype=tl.float32)
            b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None

            for i_v in range(tl.cdiv(V, BV)):
                p_v = tl.make_block_ptr(v + v_offset, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_do = tl.make_block_ptr(dout + do_offset, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_h = tl.make_block_ptr(h + h_offset, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh + dh_offset, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                # [BT, BV]
                b_v = tl.load(p_v, boundary_check=(0, 1))
                b_do = tl.load(p_do, boundary_check=(0, 1))
                # [BV, BK]
                b_h = tl.load(p_h, boundary_check=(0, 1))
                b_dh = tl.load(p_dh, boundary_check=(0, 1))
                if USE_G:
                    b_dg_last += (tl.sum(b_h * b_dh))
                # [BT, BV] @ [BV, BT] -> [BT, BT]
                b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=ALLOW_TF32)
                # [BT, BV] @ [BV, BK] -> [BT, BK]
                b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=ALLOW_TF32)
                # [BT, BV] @ [BV, BK] -> [BT, BK]
                b_dk += tl.dot(b_v, b_dh.to(b_v.dtype), allow_tf32=ALLOW_TF32)
                if USE_DW:
                    p_dv = tl.make_block_ptr(dv + dv_offset, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    b_dv = tl.load(p_dv, boundary_check=(0, 1))
                    b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=ALLOW_TF32)

            if USE_DW:
                p_dw = tl.make_block_ptr(dw + dw_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
                tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))

            # tl.debug_barrier()
            p_q = tl.make_block_ptr(q + q_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(k + k_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))

            p_dq = tl.make_block_ptr(dq + dq_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + dk_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            # m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
            if USE_G:
                b_dg = tl.zeros([BT], dtype=tl.float32)
                # g += bos * H + i_h
                # dg += bos * H + i_h
                p_g = tl.make_block_ptr(g + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
                b_g_last = tl.load(g + (bos + i_h * B * global_T) + (min(i_t * BT + BT, T) - 1))
                b_dg_last *= tl.exp(b_g_last)

                b_dq = b_dq * (tl.exp(b_g)[:, None] * scale)
                # b_dg += tl.sum(b_dq * b_q, axis=1)
                b_dg += tl.dot(BK_ones_vec, tl.trans(b_q * b_dq), allow_tf32=False)[0,:]

                b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
                # b_dg -= tl.dot(BV_ones_vec, tl.trans(b_k * b_dk), allow_tf32=False)[0,:]
                b_dg -= tl.sum(b_k * b_dk, axis=1)
                b_dg_last += tl.sum(b_dk * b_k)

                b_ds = tl.exp((tl.dot(b_g[:, None], BT_ones_vec, allow_tf32=False) - b_g[None, :])) * b_ds
                # b_ds = tl.where(m_A, b_ds, 0) * scale
                b_ds = tl.where(o_t[:, None] >= o_t[None, :], b_ds, 0)
                b_ds = tl.where(m_t[:, None] & m_t, b_ds, 0) * scale
                b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k), allow_tf32=ALLOW_TF32)
                # b_dg += tl.sum(b_ds2, axis=1)
                b_dg -= tl.sum(b_ds2, axis=0)
                b_dg += tl.dot(BT_ones_vec, tl.trans(b_ds2), allow_tf32=False)[0,:]

                b_ds = b_ds.to(b_k.dtype)
                # [BT, BK]
                b_dq += tl.dot(b_ds, b_k, allow_tf32=ALLOW_TF32)
                b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=ALLOW_TF32)
                p_dg = tl.make_block_ptr(dg + dg_offset + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
                # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
                # b_dg = tl.dot(tl.where(o_t[:, None] <= o_t[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last
                b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last)
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

            elif USE_G_GAMMA:
                b_dq = b_dq * tl.exp(b_g)[:, None] * scale
                b_dk = b_dk * tl.where(m_t, tl.exp(-b_g + b_g_last), 0)[:, None]
                b_ds = tl.where(m_A, b_ds * tl.exp(b_g[:, None] - b_g[None, :]), 0) * scale
                b_ds = b_ds.to(b_k.dtype)
                # [BT, BK]
                b_dq += tl.dot(b_ds, b_k, allow_tf32=ALLOW_TF32)
                b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=ALLOW_TF32)
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

            else:
                b_ds = tl.where(m_A, b_ds, 0)
                b_ds = b_ds.to(b_k.dtype)
                b_dq += tl.dot(b_ds, b_k, allow_tf32=ALLOW_TF32)
                b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=ALLOW_TF32) * scale
                b_dq *= scale
                tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

def chunk_bwd_dqkwg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    allow_tf32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    CONST_TILING = 256
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w) if w is not None else None
    # dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device) if g is not None else None
    dg = torch.empty((H, B, T), dtype=torch.float32, device=g.device) if g is not None else None

    # grid = (NK, NT, B * H)
    chunk_num = NK * NT * B * H
    grid_num = min(get_total_core_num(), chunk_num)
    grid = lambda meta: (grid_num, )
    tmo_chunk_bwd_kernel_dqkwg[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        h=h,
        dout=dout,
        dh=dh,
        dw=dw,
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
        chunk_num=chunk_num,
        ALLOW_TF32=allow_tf32,
    )

    # if dg is not None:
    #     dg = dg.sum(0)
    return dq, dk, dw, dg.permute(1,2,0)

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'USE_A': lambda args: args['A'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1]
        for num_stages in [3]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G'],
)
@triton.jit(do_not_specialize=['T'])
def tmo_chunk_bwd_kernel_dv_local(
    q,
    k,
    g,
    g_gamma,
    A,
    dout,
    dv,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT,
    chunk_num,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    USE_A: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    id = tl.program_id(0)
    dim = tl.num_programs(0)
    rem = chunk_num % dim
    block = chunk_num // dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)
    ones_vec = tl.full((1, BT), -1, tl.float32)
    global_T = T
    for offset in range(block_start,block_start + block):
        # grid = (NT, B, H)
        i_t = offset // (B * H)
        i_b = offset // H % B
        i_h = offset % H
        # i_t, i_bh = tl.program_id(0), tl.program_id(1)
        # i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        # offset calculation
        q_offset  = (bos * H + i_h) * K
        k_offset  = (bos * H + i_h) * K
        do_offset = (bos * H + i_h) * V
        dv_offset = (bos * H + i_h) * V

        if USE_A:
            p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
            b_A = tl.load(p_A, boundary_check=(0, 1))
        else:
            if USE_G:
                # g += bos * H + i_h
                # p_g = tl.make_block_ptr(g + (bos * H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
                p_g = tl.make_block_ptr(g + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,))
            if USE_G_GAMMA:
                b_gamma = tl.load(g_gamma + i_h)
                b_g = b_gamma * (tl.arange(0, BT) + 1)

            b_A = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(k + k_offset, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
                p_q = tl.make_block_ptr(q + q_offset, (K, T), (1, H*K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))

                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_A += tl.dot(b_k, b_q, allow_tf32=ALLOW_TF32) * scale
            if USE_G or USE_G_GAMMA:
                # b_A *= tl.exp(b_g[None, :] - b_g[:, None])
                b_A = b_A * tl.exp(tl.dot(b_g[:, None], ones_vec, allow_tf32=False) + b_g[None, :])

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        b_A = b_A.to(dout.dtype.element_ty)
        b_A = tl.where(o_t[:, None] <= o_t[None, :], b_A, tl.cast(0, dout.dtype.element_ty))
        b_A = tl.where(m_t[:, None] & m_t, b_A, tl.cast(0, dout.dtype.element_ty))
        # m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
        # b_A = tl.where(m_A, b_A, 0).to(dout.dtype.element_ty)

        for i_v in range(tl.cdiv(V, BV)):
            p_do = tl.make_block_ptr(dout + do_offset, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + dv_offset, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
            b_dv = tl.dot(b_A, b_do, allow_tf32=ALLOW_TF32)
            tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

def chunk_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    dout: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    allow_tf32: bool = True,
) -> torch.Tensor:
    B, T, H, K, V = *k.shape, dout.shape[-1]
    BT = chunk_size * 1
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    # 用于标记最大能处理的D
    CONST_TILING = 256
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(dout)
    # grid = (NT, B * H)
    chunk_num = B * H * NT
    grid_num = min(get_total_core_num(), chunk_num)
    grid = lambda meta: (grid_num, )
    tmo_chunk_bwd_kernel_dv_local[grid](
        q=q,
        k=k,
        g=g,
        g_gamma=g_gamma,
        A=A,
        dout=dout,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT=NT,
        chunk_num=chunk_num,
        ALLOW_TF32=allow_tf32,
        bottleneck="simd",
    )
    return dv
