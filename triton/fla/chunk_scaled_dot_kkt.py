import torch
import triton
import triton.language as tl

from torch_mlu_ops.triton.fla.index import prepare_chunk_indices
from torch_mlu_ops.triton.utils import get_total_core_num

@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        # for BK in [32, 64, 128]
        for BK in [128]
        for num_warps in [1]
        for num_stages in [4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def tmo_chunk_scaled_dot_kkt_fwd_kernel(
    k,
    beta,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    B,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT,
    chunk_num,
    ALLOW_TF32: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    id = tl.program_id(0)
    dim = tl.num_programs(0)
    rem = chunk_num % dim
    block = chunk_num // dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)
    ones_vec = tl.full((1, BT), 1, tl.float32)
    for offset in range(block_start,block_start + block):
        i_b = offset % B
        i_t = offset // B % NT
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
            beta + bos * H, (T, H), (H, 1), (i_t * BT, 0), (BT, H), (1,0)
        )
        p_g = tl.make_block_ptr(
            g + bos * H, (T, H), (H, 1), (i_t * BT, 0), (BT, H), (1,0)
        )

        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_beta = tl.trans(b_beta)
        b_g = tl.trans(b_g)

        ones_vec = tl.full((1, BT), 1, dtype=b_g.dtype)
        ones_vec_1 = tl.full((1, BK), 1, dtype=b_beta.dtype)
        for id_h in range(H):
            p_k = tl.make_block_ptr(
                k + bos * H * K + id_h * K, (T, K), (K * Hg, 1), (i_t * BT, 0), (BT, BK), (1, 0)
            )
            b_k = tl.load(p_k, boundary_check=(0,1))
            beta_temp = b_beta[id_h,:]
            b_kb = b_k * tl.dot(beta_temp[:, None], ones_vec_1, allow_tf32=ALLOW_TF32)
            b_A = tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k), allow_tf32=ALLOW_TF32)
            g_temp = b_g[id_h,:]
            b_A = b_A * tl.exp(tl.dot(g_temp[:, None], ones_vec, allow_tf32=False) - g_temp[None, :])

            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            b_A = b_A.to(A.dtype.element_ty)
            b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, tl.cast(0, A.dtype.element_ty))
            b_A = tl.where(m_t[:, None] & m_t, b_A, tl.cast(0, A.dtype.element_ty))
            p_A = tl.make_block_ptr(
                A + bos * H * BT + id_h * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
            )
            tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.Tensor | None = None,
    allow_tf32: bool = True,
) -> torch.Tensor:
    r"""
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
    # grid = (NT, B * H)
    chunk_num = B * NT
    grid_num = min(get_total_core_num(), chunk_num)
    grid = lambda meta: (grid_num, )
    tmo_chunk_scaled_dot_kkt_fwd_kernel[grid](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        NT=NT,
        chunk_num=chunk_num,
        ALLOW_TF32=allow_tf32,
        bottleneck="simd"
    )
    return A
