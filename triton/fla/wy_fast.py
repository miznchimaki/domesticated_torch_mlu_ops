import torch
import triton
import triton.language as tl

from torch_mlu_ops.triton.fla.index import prepare_chunk_indices
from torch_mlu_ops.triton.utils import get_total_core_num

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1]
        for num_stages in [4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def tmo_recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
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
    ALLOW_TF32: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    id = tl.program_id(0)
    dim = tl.num_programs(0)
    rem = chunk_num % dim
    block = chunk_num // dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)
    global_T = T
    for offset in range(block_start,block_start + block):
        i_b = offset % B
        i_h = offset // B % H
        i_t = offset // B // H % NT
    # i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
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
            bos, eos = i_b * T, i_b * T + T
        p_beta = tl.make_block_ptr(
            beta + bos + i_h * B * global_T, (T,), (1,), (i_t * BT,), (BT,), (0,)
        )
        p_g = tl.make_block_ptr(g + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(
            A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_g = tl.exp(tl.load(p_g, boundary_check=(0,)))

        # for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, 0),
            (BT, BV),
            (1, 0),
        )
        p_u = tl.make_block_ptr(
            u + (bos * H + i_h) * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, 0),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_Ab = (b_A * b_beta[None, :]).to(b_v.dtype)
        b_u = tl.dot(b_Ab, b_v, allow_tf32=False)
        tl.store(p_u, b_u.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        # for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, 0),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, 0),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_Abg = (b_A * b_beta[None, :] * b_g[None, :]).to(b_k.dtype)
        b_w = tl.dot(b_Abg, b_k, allow_tf32=ALLOW_TF32)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.Tensor | None = None,
    allow_tf32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = triton.cdiv(K,64) * 64
    BV = triton.cdiv(V,64) * 64
    u = torch.empty_like(v, device="mlu")
    w = k.new_empty(B, T, H, K, device="mlu")
    # grid = (NT, B * H)
    chunk_num = B * H * NT
    grid_num = min(get_total_core_num(), chunk_num)
    grid = lambda meta: (grid_num, )
    tmo_recompute_w_u_fwd_kernel[grid](
        k=k,
        v=v,
        beta=beta.permute(2,0,1).contiguous(),
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
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
        # bottleneck="simd"
    )
    return w, u

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1]
        for num_stages in [1,2,3,4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
)
@triton.jit(do_not_specialize=['T'])
def tmo_prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    g,
    A,
    dw,
    du,
    dk,
    dv,
    db,
    dg,
    cu_seqlens,
    chunk_indices,
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
        i_b = offset % B
        i_h = offset // B % H
        i_t = offset // B // H % NT
    # i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        p_b = tl.make_block_ptr(beta + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_db = tl.make_block_ptr(db + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

        b_b = tl.load(p_b, boundary_check=(0,))
        b_db = tl.zeros([BT], dtype=tl.float32)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)

        if USE_G:
            p_g = tl.make_block_ptr(g + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = tl.exp(b_g)
            b_dg = tl.zeros([BT], dtype=tl.float32)

        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            # [BT, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # 下面这个高维cycle替换为conv会有性能下降
            if USE_G:
                b_kbg = b_k * (b_b * b_g_exp)[:, None]
            else:
                b_kbg = b_k * b_b[:, None]
            b_dw = tl.load(p_dw, boundary_check=(0, 1))

            b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype), allow_tf32=ALLOW_TF32)
            b_dkbg = tl.dot(b_A, b_dw, allow_tf32=ALLOW_TF32)
            if USE_G:
                b_dk = b_dkbg * (b_g_exp * b_b)[:, None]
                b_db_part = tl.dot(b_g_exp[:, None], BK_ones_vec, allow_tf32=False) * (b_dkbg * b_k)
                b_db += tl.sum(b_db_part, 1)
                b_dg += tl.sum(b_dkbg * b_kbg, 1)
                if K > 64:
                  tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

        for i_v in range(tl.cdiv(V, BV)):
            p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
            b_vb = (tl.dot(b_b[:, None], BV_ones_vec.to(b_b.dtype), allow_tf32=False) * b_v).to(b_v.dtype)
            b_du = tl.load(p_du, boundary_check=(0, 1))
            b_dA += tl.dot(b_du, tl.trans(b_vb), allow_tf32=ALLOW_TF32)
            b_dvb = tl.dot(b_A, b_du, allow_tf32=ALLOW_TF32)
            # b_dv = b_dvb * b_b[:, None]
            b_dv = (tl.dot(b_b[:, None], BV_ones_vec.to(b_b.dtype), allow_tf32=False) * b_dvb).to(p_dv.dtype.element_ty)
            # b_db += tl.sum(b_dvb * b_v, 1)
            b_db += tl.dot(BV_ones_vec, tl.trans(b_dvb * b_v), allow_tf32=False)[0,:]
            tl.store(p_dv, b_dv, boundary_check=(0, 1))

        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
        # b_dA = tl.where(m_A, b_dA, 0)
        b_dA = b_dA.to(b_A.dtype)
        b_dA = tl.where(o_t[:, None] > o_t[None, :], b_dA, tl.cast(0, b_A.dtype))
        b_dA = tl.where(m_t[:, None] & m_t, b_dA, tl.cast(0, b_A.dtype))
        b_dA = tl.dot(b_dA, b_A, allow_tf32=ALLOW_TF32).to(b_A.dtype)
        b_dA = tl.dot(b_A, b_dA, allow_tf32=ALLOW_TF32)

        if USE_G:
            # b_dA *= tl.exp(b_g[:, None] - b_g[None, :])
            b_dA = b_dA * tl.exp(tl.dot(b_g[:, None], BT_ones_vec, allow_tf32=False) - b_g[None, :])

        # b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)
        b_dA = -b_dA.to(b_A.dtype)
        b_dA = tl.where(o_t[:, None] > o_t[None, :], b_dA, tl.cast(0, b_A.dtype))
        b_dA = tl.where(m_t[:, None] & m_t, b_dA, tl.cast(0, b_A.dtype))
        b_A = tl.zeros([BT, BT], dtype=tl.float32)

        # tl.debug_barrier()
        for i_k in range(tl.cdiv(K, BK)):
            p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_kt = tl.trans(b_k)
            b_ktb = b_kt * b_b[None, :]

            b_A += tl.dot(b_k, b_kt, allow_tf32=ALLOW_TF32)
            b_dkb = tl.dot(b_dA, b_k, allow_tf32=ALLOW_TF32)

            # b_db += tl.sum(b_dkb * b_k, 1)
            b_db += tl.dot(BK_ones_vec, tl.trans(b_dkb * b_k), allow_tf32=False)[0,:]

            # b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(b_ktb.to(b_dA.dtype), b_dA))
            # b_dk += tl.load(p_dk, boundary_check=(0, 1))
            if K > 64:
                b_dk = (tl.dot(b_b[:, None], BK_ones_vec.to(b_b.dtype), allow_tf32=False) * b_dkb) + tl.load(p_dk, boundary_check=(0, 1))
            else:
                b_dk += (tl.dot(b_b[:, None], BK_ones_vec.to(b_b.dtype), allow_tf32=False) * b_dkb)
            b_dk += tl.trans(tl.dot(b_ktb.to(b_dA.dtype), b_dA, allow_tf32=ALLOW_TF32))

            tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

        # b_A *= b_b[:, None]
        b_A = tl.dot(b_b[:, None], BT_ones_vec.to(b_b.dtype), allow_tf32=False) * b_A
        if USE_G:
            b_AdA = b_dA * b_A
            p_dg = tl.make_block_ptr(dg + (bos + i_h * B * global_T), (T,), (1,), (i_t * BT,), (BT,), (0,))
            # b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
            b_AdA_sum_low = tl.dot(BT_ones_vec, tl.trans(b_AdA), allow_tf32=False)
            b_dg += b_AdA_sum_low - tl.sum(b_AdA, axis=0)
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty)[0,:], boundary_check=(0,))

def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    g: torch.Tensor = None,
    cu_seqlens: torch.LongTensor | None = None,
    allow_tf32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 256
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    # dg = torch.empty_like((H, B, T), dtype=g.dtype, device=g.device) if g is not None else None
    # db = torch.empty((H, B, T), dtype=beta.dtype, device=beta.device)
    dg = torch.empty_like(g) if g is not None else None
    db = torch.empty_like(beta)

    chunk_num = B * H * NT
    grid_num = min(get_total_core_num(), chunk_num)
    grid = lambda meta: (grid_num, )
    tmo_prepare_wy_repr_bwd_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=du,
        dk=dk,
        dv=dv,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
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
    )
    return dk, dv, db.permute((1,2,0)), dg.permute((1,2,0))
