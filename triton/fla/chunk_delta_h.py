import torch
import triton
import triton.language as tl

from torch_mlu_ops.triton.fla.index import prepare_chunk_indices, prepare_chunk_offsets

@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1]
        for num_stages in [4]
        # for BV in [32, 64]
        for BV in [64]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def tmo_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    B,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    global_T = T
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([BK, BV], dtype=tl.float32)
        # b_h1_tmp = tl.empty([BK, BV], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # calculate stride
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K

    # load initial state
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = tl.make_block_ptr(h0, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
    # main recurrence
    for i_t in range(NT):
        if STATE_V_FIRST:
            p_h1 = tl.make_block_ptr(
                h + i_t * stride_h, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0)
            )
        else:
            p_h1 = tl.make_block_ptr(
                h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
            )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype), allow_tf32=ALLOW_TF32)
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype), allow_tf32=ALLOW_TF32)
        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        o_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
        b_v = b_v * (-1) + o_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos + last_idx + i_h * B * global_T)
            p_g = tl.make_block_ptr(
                g + bos + i_h * B * global_T, (T,), (1,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            ones_vec_1 = tl.full((1, BV), 1, dtype=b_g.dtype)
            b_v = b_v * tl.dot(tl.where(m_t, tl.exp(b_g_last - b_g), 0)[:, None], ones_vec_1)
            b_h1_tmp = b_h1 * tl.exp(b_g_last)

        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (BK, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_G:
                if STATE_V_FIRST:
                    b_h1_tmp *= tl.exp(b_gk_last1)[None, :]
                else:
                    b_h1_tmp *= tl.dot(tl.exp(b_gk_last1)[:, None], ones_vec_1)
            else:
                if STATE_V_FIRST:
                    b_h1_tmp = b_h1 * tl.exp(b_gk_last1)[None, :]
                else:
                    b_h1_tmp = b_h1 * tl.dot(tl.exp(b_gk_last1)[:, None], ones_vec_1)
            # b_h1 *= tl.exp(b_gk_last1)[:, None]
        if not USE_G and not USE_GK:
            if STATE_V_FIRST:
                b_h1_tmp = b_h1 + tl.trans(tl.dot(b_k, b_v))
            else:
                b_h1_tmp = b_h1 + tl.dot(b_k, b_v)
        else:
            if STATE_V_FIRST:
                b_h1_tmp += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h1_tmp += tl.dot(b_k, b_v)

        # p_h1 = tl.make_block_ptr(
        #     h + i_t * stride_h, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)
        # )
        # tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))

        # b_h1 += tl.dot(b_k, b_v)
        b_h1 = b_h1_tmp
    # epilogue
    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(ht, (V, K), (K, 1), (i_v * BV, 0), (BV, BK), (1, 0))
        else:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    allow_tf32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    BK = triton.cdiv(K,64) * 64
    # BV = triton.cdiv(V,64) * 64

    if state_v_first:
        h = k.new_empty(B, NT, H, V, K)
        final_state = k.new_zeros(N, H, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, H, K, V)
        final_state = (
            k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
        )
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)
    tmo_chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        B=B,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        ALLOW_TF32=allow_tf32,
        bottleneck="simd",
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state

@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1]
        for num_stages in [1,2,3,4]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G', 'USE_EXP2'],
)
@triton.jit(do_not_specialize=['T'])
def tmo_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    dout,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    B,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    i_v, i_nh = 0, tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    global_T = T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    BT_ones_vec = tl.full((1, BT), 1, tl.float32)
    BK_ones_vec = tl.full((1, K), 1, tl.float32)
    BV_ones_vec = tl.full((1, V), 1, tl.float32)

    # [BK, BV]
    # BV = V
    # b_dh1 = tl.zeros([K, V], dtype=tl.float32)
    # if K > 64:
    #     b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    # if K > 128:
    #     b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    # if K > 192:
    #     b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    dout += (bos * H + i_h).to(tl.int64) * V
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * H + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * V), (K, V), (1, 0))
        b_dh1 = tl.load(p_dht1, boundary_check=(0, 1))
        # if K > 64:
        #     p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        #     b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        # if K > 128:
        #     p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        #     b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        # if K > 192:
        #     p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        #     b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (0, i_v * V), (K, V), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        # if K > 64:
        #     p_dh2 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        #     tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        # if K > 128:
        #     p_dh3 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        #     tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        # if K > 192:
        #     p_dh4 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        #     tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) + i_h * B * global_T).to(tl.float32)
            p_g = tl.make_block_ptr(g + bos + i_h * B * global_T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            if USE_EXP2:
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
            else:
                bg_last_exp = tl.exp(bg_last)
                b_g_exp = tl.exp(b_g)

        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * V), (BT, V), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * V), (BT, V), (1, 0))
        p_do = tl.make_block_ptr(dout, (T, V), (H*V, 1), (i_t * BT, i_v * V), (BT, V), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, K), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, K)
            b_gk_last1 = tl.load(gk + last_idx * H*K + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype), allow_tf32=ALLOW_TF32)

        # if K > 64:
        #     p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
        #     b_k = tl.load(p_k, boundary_check=(0, 1))
        #     if USE_GK:
        #         o_k2 = 64 + o_k1
        #         b_gk_last2 = tl.load(gk + last_idx * H*K + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
        #     b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        # if K > 128:
        #     p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
        #     b_k = tl.load(p_k, boundary_check=(0, 1))
        #     if USE_GK:
        #         o_k3 = 128 + o_k1
        #         b_gk_last3 = tl.load(gk + last_idx * H*K + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
        #     b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        # if K > 192:
        #     p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
        #     b_k = tl.load(p_k, boundary_check=(0, 1))
        #     if USE_GK:
        #         o_k4 = 192 + o_k1
        #         b_gk_last4 = tl.load(gk + last_idx * H*K + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
        #     b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            if USE_EXP2:
                # b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
                b_g_eff = tl.where(m_t, exp2(bg_last - b_g), 0)
                b_dv = tl.dot(b_g_eff[:, None], BV_ones_vec, allow_tf32=False) * b_dv
            else:
                # b_dv *= tl.where(m_t, tl.exp(bg_last - b_g), 0)[:, None]
                b_g_eff = tl.where(m_t, tl.exp(bg_last - b_g), 0)
                b_dv = tl.dot(b_g_eff[:, None], BV_ones_vec, allow_tf32=False) * b_dv
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (0, i_t * BT), (K, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (0, i_t * BT), (K, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if USE_EXP2:
                b_dh1 *= exp2(b_gk_last1[:, None])
            else:
                b_dh1 *= tl.exp(b_gk_last1[:, None])
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype), allow_tf32=ALLOW_TF32) * scale - tl.dot(b_w, b_dv.to(b_w.dtype), allow_tf32=ALLOW_TF32)
        # if K > 64:
        #     p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
        #     p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
        #     b_q = tl.load(p_q, boundary_check=(0, 1))
        #     b_w = tl.load(p_w, boundary_check=(0, 1))
        #     if USE_G:
        #         b_dh2 *= bg_last_exp
        #         b_q = b_q * b_g_exp[None, :]
        #     if USE_GK:
        #         if USE_EXP2:
        #             b_dh2 *= exp2(b_gk_last2[:, None])
        #         else:
        #             b_dh2 *= exp(b_gk_last2[:, None])
        #     b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        # if K > 128:
        #     p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
        #     p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
        #     b_q = tl.load(p_q, boundary_check=(0, 1))
        #     b_w = tl.load(p_w, boundary_check=(0, 1))
        #     if USE_G:
        #         b_dh3 *= bg_last_exp
        #         b_q = b_q * b_g_exp[None, :]
        #     if USE_GK:
        #         if USE_EXP2:
        #             b_dh3 *= exp2(b_gk_last3[:, None])
        #         else:
        #             b_dh3 *= exp(b_gk_last3[:, None])
        #     b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        # if K > 192:
        #     p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
        #     p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
        #     b_q = tl.load(p_q, boundary_check=(0, 1))
        #     b_w = tl.load(p_w, boundary_check=(0, 1))
        #     if USE_G:
        #         b_dh4 *= bg_last_exp
        #         b_q = b_q * b_g_exp[None, :]
        #     if USE_GK:
        #         if USE_EXP2:
        #             b_dh4 *= exp2(b_gk_last4[:, None])
        #         else:
        #             b_dh4 *= exp(b_gk_last4[:, None])
        #     b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * V), (K, V), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        # if K > 64:
        #     p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
        #     tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        # if K > 128:
        #     p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
        #     tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        # if K > 192:
        #     p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
        #     tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))

def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
    allow_tf32: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, dout.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (N*H,)
    tmo_chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        dout=dout,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BV=V,
        USE_EXP2=use_exp2,
        ALLOW_TF32=allow_tf32,
        bottleneck="io"
    )
    return dh, dh0, dv2
