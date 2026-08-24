import torch
import triton
import triton.language as tl

NUM_WARPS_AUTOTUNE = [2, 4, 8, 16]
STATIC_WARPS = 32

@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps, num_stages=num_stages)
        for BD in [128, 256]
        for num_warps in [1]
        for num_stages in [4]
    ],
    key=['D', 'W', 'NB'],
)
@triton.jit
def tmo_causal_conv1d_fwd_fla_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    NT,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DTYPE:tl.constexpr,
):
    # i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    id = tl.program_id(0)
    dim = tl.num_programs(0)
    ND = tl.cdiv(D, BD)
    chunk_num = NT * B * ND
    rem = chunk_num % dim
    block = chunk_num // dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)

    # i_d = tl.program_id(1)
    i_d_b = -1
    if HAS_WEIGHT:
        b_w = tl.zeros((BD, BW), dtype=DTYPE)
        b_w_trans = tl.trans(b_w).to(tl.float32)
    for offset in range(block_start, block_start + block):
        idx_t = offset % (B * NT)
        i_d = offset // (B * NT)
        # NT = tl.cdiv(T, BT)
        # for i_t in range(0, NT):
        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, BW) + W - BW
        m_d = o_d < D
        m_w = o_w >= 0
        if HAS_WEIGHT:
            if i_d_b != i_d:
                i_d_b = i_d
                # [BD, BW]
                # b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
                p_w = tl.make_block_ptr(weight, (D, W), (W, 1), (i_d * BD, W - BW), (BD, BW), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_w_trans = tl.trans(b_w).to(tl.float32)

        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + idx_t * 2).to(tl.int32), tl.load(chunk_indices + idx_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_V = eos - bos
            T = tl.cast(T_V, tl.int32)
            p_x = x + bos * stride_x_t
        else:
            i_n = idx_t // NT
            i_t = idx_t % NT
            bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
            p_x = x + i_n * stride_x_n

        b_y = tl.zeros((BT, BD), dtype=tl.float32)
        if not USE_INITIAL_STATE:
            # [BT, BD]
            p_yi = tl.make_block_ptr(p_x, (T, D), (stride_x_t, stride_x_d), (i_t * BT - W + 1, i_d * BD), (BT + W - 1, BD), (1, 0))
            # p_yi = tl.make_block_ptr(base=p_x,
            #                          shape=(T, D),
            #                          strides=(stride_x_t, stride_x_d),
            #                          offsets=(i_t * BT - W + 1, i_d * BD),
            #                          block_shape=(BT + W - 1, BD),
            #                          order=(1, 0))
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)

            for i_w in tl.static_range(0, W):
                # [BT, BD] x [BD]
                if HAS_WEIGHT:
                    b_y += (b_yi[i_w:(i_w + BT), :] * b_w_trans[i_w + BW - W, :])
        elif i_t * BT >= W:
            # to make Triton compiler happy, we need to copy codes
            for i_w in tl.static_range(-W + 1, 1):
                p_yi = tl.make_block_ptr(p_x, (T, D), (stride_x_t, stride_x_d), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                # [BT, BD]
                b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
                if HAS_WEIGHT:
                    b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
                b_y += b_yi
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(-W + 1, 1):
                o_x = o_t + i_w
                m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
                m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

                b_yi = tl.load(
                    p_x + o_x[:, None] * stride_x_t + o_d * stride_x_d,
                    mask=m_x,
                    other=0
                ).to(tl.float32)

                b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)

                if HAS_WEIGHT:
                    b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
                b_y += b_yi

        if HAS_BIAS:
            b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
            b_y = b_y * tl.sigmoid(b_y)

        if HAS_RESIDUAL:
            p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_residual = tl.load(p_residual, boundary_check=(0, 1))
            b_y += b_residual

        p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))

@triton.jit()
def tmo_causal_conv1d_fwd_vllm_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # (batch, n_blocks + padding) The second dimension contains
    # the block indices relevant for each sequence
    # plus potential 0-padding at the beginning and at the end
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,  # (batch,)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    num_computed_tokens,  # (batch,)
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    NT,
    cu_seqlen: tl.int32,  # cu_seqlen
    num_cache_lines,  # added to support vLLM larger cache lines
    # Strides
    stride_x_dim,  # stride to get to next feature-value,
    stride_x_token,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim,  # stride to get to next dim-axis value
    stride_w_width,  # stride to get to next width-axis value
    stride_istate_seq,
    stride_istate_dim,
    stride_istate_token,
    stride_cache_indices,
    stride_o_dim,
    stride_o_token,
    stride_block_m: tl.constexpr,  # Stride block to align divided by BLOCK_M
    # others
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    id = tl.program_id(0)
    task_dim = tl.num_programs(0)
    ND = tl.cdiv(dim, BLOCK_N)
    chunk_num = NT * ND

    rem = chunk_num % task_dim
    block = chunk_num // task_dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)
    for offset in range(block_start, block_start + block):

        # one program handles one chunk in a single sequence
        # rather than mixing sequences - to make updating initial_states across sequences efficiently
        i_t = offset // ND
        i_d = offset % ND
        # single-sequence id
        idx_seq = tl.load(batch_ptr + i_t).to(tl.int64)
        chunk_offset = tl.load(token_chunk_offset_ptr + i_t)

        # BLOCK_N elements along the feature-dimension (channel)
        idx_feats = i_d * BLOCK_N + tl.arange(0, BLOCK_N)
        if idx_seq != pad_slot_id:
            # return
            sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
            sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
            # find the actual sequence length
            seqlen = sequence_end_index - sequence_start_index
            B_size: tl.constexpr = stride_block_m * BLOCK_M

            if IS_APC_ENABLED:
                # Handle the case if prefix caching is enabled.
                # In particular, if prefix caching is enabled, the program write additional cache states to "cache_indices_ptr"

                # Get the length of the completed sequence so far and compute the offset.
                current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
                current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
                sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

                # Compute the offset where the first stride_block_m-aligned first full block is
                # Value in "token-space"
                sequence_completed_offset_token = sequence_completed_index % B_size
                seq_completed_offset = B_size - sequence_completed_offset_token
                seq_end_offset = (seqlen - seq_completed_offset) % B_size
                last_full_block_token_index = sequence_end_index - seq_end_offset
                # If the sequence without the sequence_offset_index is stride_cache_chunk-aligned, then the last full chunk is the second-to-last one
                if seq_end_offset == 0:
                    last_full_block_token_index = last_full_block_token_index - B_size

                # Get the number of blocks to be filled for the current sequence
                # If n_block_to_fill = 0, then only the state at the sequence end is stored
                n_block_to_fill = current_last_index - current_first_index

                # Get the index of the init block
                conv_state_init_index = tl.load(initial_state_idx + idx_seq)
            else:
                n_block_to_fill = 0
                current_last_index = 0
                conv_state_init_index = 0
                current_first_index = 0
                last_full_block_token_index = 0

            token_offset = BLOCK_M * chunk_offset
            segment_len = min(BLOCK_M, seqlen - token_offset)

            # base of the sequence
            x_base = (
                x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
            )  # [BLOCK_N,]

            # cache_idx
            conv_states_input_coord = tl.load(
                conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
            ).to(tl.int64)

            # if HAS_NULL_BLOCK:  # noqa
            #     if conv_states_input_coord == pad_slot_id:
            #         # not processing as this is not the actual sequence
            #         return
            if not HAS_NULL_BLOCK or conv_states_input_coord != null_block_id:
                conv_states_base = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)
                )  # [BLOCK_N,]

                w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

                # Does 2 things:
                # 1. READ prior-block init-state data - [done by every Triton programs]
                # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
                if chunk_offset == 0:
                    # read from conv_states
                    load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
                    if load_init_state:
                        # load from conv_states
                        prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
                        mask_w = idx_feats < dim
                        if KERNEL_WIDTH == 2:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                        if KERNEL_WIDTH == 3:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                        if KERNEL_WIDTH == 4:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                        if KERNEL_WIDTH == 5:
                            conv_states_ptrs = prior_tokens  # [BLOCK_N]
                            col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                            col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                            col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                            conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                            col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
                    else:
                        # prior-tokens are zeros
                        if KERNEL_WIDTH >= 2:  # STRATEGY1
                            # first chunk and does not have prior-token, so just set to 0
                            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                        if KERNEL_WIDTH >= 3:  # STRATEGY1
                            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                        if KERNEL_WIDTH >= 4:  # STRATEGY1
                            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
                        if KERNEL_WIDTH >= 5:  # STRATEGY1
                            col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

                    # STEP 2:
                    # here prepare data for updating conv_state
                    if (
                        state_len <= seqlen
                    ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
                        # just read from 'x'
                        # copy 'x' data to conv_state
                        # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
                        idx_tokens_last = (seqlen - state_len) + tl.arange(
                            0, NP2_STATELEN
                        )  # [BLOCK_M]
                        x_ptrs = (
                            x_ptr
                            + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                            + (idx_feats * stride_x_dim)[None, :]
                        )  # [BLOCK_M,BLOCK_N,]
                        mask_x = (
                            (idx_tokens_last >= 0)[:, None]
                            & (idx_tokens_last < seqlen)[:, None]
                            & (idx_feats < dim)[None, :]
                        )  # token-index  # token-index  # feature-index
                        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                        idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                        # Compute the offset where the last block should be written in the conv_states
                        conv_states_output_coord = tl.load(
                            conv_state_indices_ptr
                            + idx_seq * stride_cache_indices
                            + current_last_index
                        ).to(tl.int64)

                        conv_states_ptrs_target = (
                            conv_states_ptr
                            + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                            + (idx_feats * stride_conv_state_dim)
                        )[None, :] + (  # [BLOCK_N,]
                            idx_tokens_conv * stride_conv_state_tok
                        )[:, None]

                        mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                        tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
                        tl.store(conv_states_ptrs_target, loaded_x, mask)

                    else:
                        if load_init_state:
                            # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                            conv_states_ptrs_source = (
                                conv_states_ptr
                                + (conv_states_input_coord * stride_conv_state_seq)
                                + (idx_feats * stride_conv_state_dim)[None, :]
                                + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                            )  # [BLOCK_M, BLOCK_N]
                            mask = (
                                (conv_states_input_coord < num_cache_lines)
                                & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                                & (idx_feats < dim)[None, :]
                            )
                            conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                            VAL = state_len - seqlen

                            x_ptrs = (
                                x_base[None, :]
                                + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                            )  # [BLOCK_M, BLOCK_N]

                            mask_x = (
                                (idx_tokens_conv - VAL >= 0)[:, None]
                                & (idx_tokens_conv - VAL < seqlen)[:, None]
                                & (idx_feats < dim)[None, :]
                            )  # token-index  # token-index  # feature-index
                            loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                            tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                            new_conv_state = tl.where(
                                mask, conv_state, loaded_x
                            )  # BUG in 'tl.where'  which requires a barrier before this
                            conv_states_ptrs_target = (
                                conv_states_base
                                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                            )  # [BLOCK_M, BLOCK_N]
                            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                                None, :
                            ]
                            tl.store(conv_states_ptrs_target, new_conv_state, mask)
                        else:  # load_init_state == False
                            # update conv_state by shifting left, BUT
                            # set cols prior to 'x' as zeros + cols from 'x'
                            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                            VAL = state_len - seqlen

                            x_ptrs = (
                                x_base[None, :]
                                + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                            )  # [BLOCK_M, BLOCK_N]

                            mask_x = (
                                (idx_tokens_conv - VAL >= 0)[:, None]
                                & (idx_tokens_conv - VAL < seqlen)[:, None]
                                & (idx_feats < dim)[None, :]
                            )  # token-index  # token-index  # feature-index
                            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                            conv_states_ptrs_target = (
                                conv_states_base
                                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                            )  # [BLOCK_M, BLOCK_N]
                            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                                None, :
                            ]
                            tl.store(conv_states_ptrs_target, new_conv_state, mask)

                else:  # chunk_offset > 0
                    # read prior-token data from `x`
                    load_init_state = True
                    prior_tokens = x_base + (token_offset - 1) * stride_x_token
                    mask_w = idx_feats < dim
                    if KERNEL_WIDTH == 2:
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                    if KERNEL_WIDTH == 3:
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                    if KERNEL_WIDTH == 4:
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
                        col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                    if KERNEL_WIDTH == 5:
                        # ruff: noqa: F841
                        conv_states_ptrs = prior_tokens  # [BLOCK_N]
                        col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
                        col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
                        col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
                        conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
                        col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

                    # Store intermediate states aligned with stride_block_m
                    # The additional states are cached starting from the last stride_block_m.
                    # For example:
                    # If n_block_to_fill = 0, then only the state at the sequence end is cached and the process below is not involved.
                    # If n_block_to_fill > 0, then the states at the sequence end and at the n_block_to_fill-last
                    # stride_block_m are cached.
                    # For example chunk_offset = n_block_to_fill stores the state at last_full_block
                    if (chunk_offset - 1) < n_block_to_fill:
                        # Store the states at the chunk boundaries from the start of the sequence
                        idx_tokens_last = (
                            last_full_block_token_index
                            - (n_block_to_fill - chunk_offset) * B_size
                            - state_len
                        ) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
                        x_ptrs = (
                            x_ptr
                            + (idx_tokens_last * stride_x_token)[:, None]
                            + (idx_feats * stride_x_dim)[None, :]
                        )  # [BLOCK_M,BLOCK_N,]

                        mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[
                            None, :
                        ]  # token-index  # token-index  # feature-index
                        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                        idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                        # cache_idx
                        conv_states_output_coord = tl.load(
                            conv_state_indices_ptr
                            + idx_seq * stride_cache_indices
                            + current_first_index
                            + (chunk_offset - 1)
                        ).to(tl.int64)

                        conv_states_ptrs_target = (
                            conv_states_ptr
                            + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                            + (idx_feats * stride_conv_state_dim)
                        )[None, :] + (  # [BLOCK_N,]
                            idx_tokens_conv * stride_conv_state_tok
                        )[:, None]

                        mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                        tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
                        tl.store(conv_states_ptrs_target, loaded_x, mask)

                if HAS_BIAS:
                    bias = bias_ptr + idx_feats
                    mask_bias = idx_feats < dim
                    acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
                        tl.float32
                    )  # [BLOCK_N]
                else:
                    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

                x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

                # PRE-LOAD WEIGHTS
                mask_w = idx_feats < dim
                if KERNEL_WIDTH >= 2:
                    w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
                    w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
                    w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
                    w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
                if KERNEL_WIDTH >= 3:
                    w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
                    w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
                if KERNEL_WIDTH >= 4:
                    w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
                    w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
                mask_x_1d = idx_feats < dim
                for idx_token in range(segment_len):
                    acc = acc_preload

                    matrix_w = w_col0
                    matrix_x = col0
                    for j in tl.static_range(KERNEL_WIDTH):
                        if KERNEL_WIDTH == 2:
                            if j == 1:  # KERNEL_WIDTH-1:
                                matrix_w = w_col1
                                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                        elif KERNEL_WIDTH == 3:
                            if j == 1:
                                matrix_w = w_col1
                                matrix_x = col1
                            elif j == 2:
                                matrix_w = w_col2
                                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
                        elif KERNEL_WIDTH == 4:
                            if j == 1:
                                matrix_w = w_col1
                                matrix_x = col1
                            elif j == 2:
                                matrix_w = w_col2
                                matrix_x = col2
                            elif j == 3:
                                matrix_w = w_col3
                                x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                                matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

                        acc += matrix_x * matrix_w  # [BLOCK_N]

                    if KERNEL_WIDTH == 2:
                        col0 = matrix_x
                    elif KERNEL_WIDTH == 3:
                        col0 = col1
                        col1 = matrix_x
                    elif KERNEL_WIDTH == 4:
                        col0 = col1
                        col1 = col2
                        col2 = matrix_x

                    if SILU_ACTIVATION:
                        acc = acc / (1 + tl.exp(-acc))
                    mask_1d = (idx_token < segment_len) & (
                        idx_feats < dim
                    )  # token-index  # feature-index
                    o_ptrs = (
                        o_ptr
                        + (sequence_start_index + token_offset + idx_token) * stride_o_token
                        + (idx_feats * stride_o_dim)
                    )

                    tl.store(o_ptrs, acc, mask=mask_1d)



@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps, num_stages=num_stages)
        for BD in [128, 256]
        for num_warps in [1]
        for num_stages in [2,4]
    ],
    key=['D', 'W', 'NB'],
)
@triton.jit
def tmo_causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    stride_x_n,   # x batch stride
    stride_x_t,   # x time stride
    stride_x_d,   # x dim stride
    stride_dx_n,  # dx batch stride
    stride_dx_t,  # dx time stride
    stride_dx_d,  # dx dim stride
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    NT,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DTYPE:tl.constexpr,
):
    # i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    id = tl.program_id(0)
    dim = tl.num_programs(0)
    ND = tl.cdiv(D, BD)
    chunk_num = B * NT * ND
    rem = chunk_num % dim
    block = chunk_num // dim + (id < rem)
    block_start = id * block + tl.where(id < rem, 0, rem)
    # i_d = tl.program_id(1)
    i_d_b = -1

    if HAS_WEIGHT:
        b_w = tl.zeros((BD, W), dtype=DTYPE)
        b_w_trans = tl.trans(b_w).to(tl.float32)
    for offset in range(block_start, block_start + block):
        b_dw_trans = tl.zeros((W,BD), dtype=tl.float32)
        if HAS_BIAS:
            b_db = tl.zeros((BD,), dtype=tl.float32)
        idx_t = offset % (B * NT)
        i_d = offset // (B * NT)
        o_d = i_d * BD + tl.arange(0, BD)
        m_d = o_d < D
        o_w = tl.arange(0, BW) + W - BW
        m_w = o_w >= 0
        if HAS_WEIGHT:
            if i_d != i_d_b:
                i_d_b = i_d
                if not USE_FINAL_STATE and not USE_INITIAL_STATE:
                    # [BD, BW]
                    p_w = tl.make_block_ptr(weight, (D, W), (W, 1), (i_d * BD, 0), (BD, W), (1, 0))
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    b_w_trans = tl.trans(b_w).to(tl.float32)
                else:
                    b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)

        if IS_VARLEN:
            # i_tg = i_t
            i_n, i_t = tl.load(chunk_indices + idx_t * 2).to(tl.int32), tl.load(chunk_indices + idx_t * 2 + 1).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_V = eos - bos
            T = tl.cast(T_V, tl.int32)
            p_x = x + bos * stride_x_t
        else:
            # i_tg = i_b * NT + i_t
            # i_n = i_b
            i_n = idx_t // NT
            i_t = idx_t % NT
            bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
            p_x = x + i_n * stride_x_n
        # NT = tl.cdiv(eos - bos, BT)
        # for i_t in range(0, NT):
        if HAS_WEIGHT:
            p_x = tl.make_block_ptr(p_x, (T, D), (stride_x_t, stride_x_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_x = tl.load(p_x, boundary_check=(0, 1))

        # b_dx = tl.zeros((BT, BD), dtype=tl.float32)

        if not USE_FINAL_STATE and not USE_INITIAL_STATE:
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT + W - 1, BD), (1, 0))
            # [BT, BD]
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)

            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT + W - 1, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

            for i_w in tl.static_range(0, W):
                b_wdy = b_dy
                if HAS_WEIGHT:
                    # [BT, BD]
                    if i_w == 0:
                        b_dx = (b_wdy[i_w : i_w + BT, :] * b_w_trans[W - i_w - 1, :])
                    else:
                        b_dx += (b_wdy[i_w : i_w + BT, :] * b_w_trans[W - i_w - 1, :])

                    # [BD]
                    # b_dw_part = tl.sum(b_dy[i_w : i_w + BT, :] * b_x, 0)
                    # tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
                    b_dw_trans[W - i_w - 1,:] += tl.sum(b_dy[i_w : i_w + BT, :] * b_x, 0)
            if HAS_BIAS:
                if T - i_t * BT >= BT:
                    b_db += tl.sum(b_dy[0:BT, :], 0)
                else:
                    o_t = i_t * BT + tl.arange(0, BT)
                    m_t = o_t < T
                    b_db += tl.sum(b_dy[0:BT, :] * m_t[:,None], 0)
        elif i_t * BT >= W:
            # to make Triton compiler happy, we need to copy codes
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                # [BT, BD]
                b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                b_wdy = b_dy
                if HAS_WEIGHT:
                    # [BT, BD]
                    b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                    # [BD]
                    b_dw = tl.sum(b_dy * b_x, 0)
                    tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy, 0)
                b_dx += b_wdy
        else:
            # which may use initial state
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                    b_y_shift = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                    b_ys = tl.sigmoid(b_y_shift)
                    b_dy_shift = b_dy_shift * b_ys * (1 + b_y_shift * (1 - b_ys))
                if HAS_WEIGHT:
                    # gradient comes from x：sum_t dy[t+i_w] * x[t]
                    b_dw = tl.sum(b_dy_shift * b_x, 0)
                    # index of cache：c = W - i_w + t
                    if USE_INITIAL_STATE:
                        mask_head_rows = (o_t < i_w)
                        # dy_head = dy[t]
                        b_dy_head = tl.load(dy + bos * D + o_t[:, None] * D + o_d, mask=(mask_head_rows[:, None] & m_d[None, :]),
                                            other=0.0).to(tl.float32)
                        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                            # use y[t] （not y[t+i_w]）
                            b_y_head = tl.load(y + bos * D + o_t[:, None] * D + o_d,
                                               mask=(mask_head_rows[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                            b_ys_head = tl.sigmoid(b_y_head)
                            b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                        o_c = W - i_w + o_t
                        # index 0 is padding 0
                        mask_c = (mask_head_rows & (o_c >= 1) & (o_c < W))
                        b_xc = tl.load(initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                                       mask=(mask_c[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                        # add the gradient comes from initial_state
                        b_dw += tl.sum(b_dy_head * b_xc, 0)
                    tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_shift, 0)
                b_wdy = b_dy_shift if not HAS_WEIGHT else (b_dy_shift * tl.sum(b_w * (o_w == (W - i_w - 1)), 1))
                b_dx += b_wdy

        if HAS_BIAS:
            b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
            tl.store(db + idx_t * D + o_d, b_db, mask=m_d)

        if USE_FINAL_STATE:
            if i_t * BT + BT >= T-W:
                start_tok = max(0, T - (W - 1))
                offset = i_t * BT + tl.arange(0, BT)
                tok_idx = offset - start_tok
                mask = (offset >= start_tok) & (offset < T)
                w_idx = 1 + tok_idx
                dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
                b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.).to(tl.float32)
                b_dx += b_dht

        if IS_VARLEN:
            p_dx = dx + bos * stride_dx_t
        else:
            p_dx = dx + i_n * stride_dx_n

        p_dx = tl.make_block_ptr(p_dx, (T, D), (stride_dx_t, stride_dx_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))

        tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))
        if HAS_WEIGHT:
           p_dw = tl.make_block_ptr(dw + idx_t * D * W, (D, W), (W, 1), (i_d * BD, 0), (BD, W), (1, 0))
           tl.store(p_dw, tl.trans(b_dw_trans), boundary_check=(0, 1))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['cache'] is not None,
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
})
@triton.jit
def tmo_causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    stride_x_n,  # batch stride
    stride_x_d,  # dim stride
    stride_y_n,  # batch stride
    stride_y_d,  # dim stride
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W

    # [BD]
    b_x = tl.load(x + i_n * stride_x_n + o_d * stride_x_d, mask=m_d, other=0).to(tl.float32)

    b_cache = tl.zeros((BD, BW), dtype=tl.float32)

    if USE_INITIAL_STATE:
        # 2. Shift Cache (Read [1:])
        p_cache_read = tl.make_block_ptr(
            cache + i_n * D*W,
            shape=(D, W),
            strides=(W, 1),
            offsets=(i_d * BD, 1),
            block_shape=(BD, BW),
            order=(1, 0)
        )
        b_cache = tl.load(p_cache_read, boundary_check=(0, 1)).to(tl.float32)

        # 3. Fill x to the last position
        m_update = o_w == (W - 1)
        b_cache = tl.where(m_update[None, :], b_x[:, None], b_cache)

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
        b_y = tl.sum(b_cache * b_w, 1)
    else:
        b_y = tl.sum(b_cache, 1)

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        b_y += tl.load(residual + i_n * D + o_d, mask=m_d, other=0)

    tl.store(y + i_n * stride_y_n + o_d * stride_y_d, tl.cast(b_y,
             dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_d)

    if USE_INITIAL_STATE:
        p_cache_write = tl.make_block_ptr(
            cache + i_n * D*W,
            shape=(D, W),
            strides=(W, 1),
            offsets=(i_d * BD, 0),
            block_shape=(BD, BW),
            order=(1, 0)
        )
        tl.store(p_cache_write, tl.cast(b_cache, dtype=cache.dtype.element_ty,
                 fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'USE_ACTIVATION': lambda args: args['y'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def tmo_compute_dh0_kernel(
    dy,
    y,
    weight,
    dh0,
    cu_seqlens,
    stride_dy_n,
    stride_dy_t,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    USE_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Compute dh0 (gradient w.r.t. initial_state) in a separate kernel.
    This avoids Triton compiler bugs on some architectures (e.g., GB200).

    Grid: (cdiv(D, BD), N)
    """
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    # Get sequence boundaries
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        # For varlen, dy is [1, total_T, D], offset by bos
        dy_base = dy + bos * stride_dy_t
    else:
        seq_len = T
        # For non-varlen, dy is [B, T, D], offset by i_n * stride_dy_n
        dy_base = dy + i_n * stride_dy_n

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # For each i_w in [1, W), compute dh0[i_n, :, i_w]
    for i_w in tl.static_range(1, W):
        b_dh0 = tl.zeros([BD], dtype=tl.float32)

        # Accumulate contributions from t = 0 to min(i_w, seq_len) - 1
        for t in tl.static_range(0, W - 1):
            if t < i_w:
                w_idx = i_w - 1 - t

                # Load dy[t, :] relative to dy_base
                p_dy = dy_base + t * stride_dy_t + o_d
                m_t = (t < seq_len) & m_d
                b_dy = tl.load(p_dy, mask=m_t, other=0).to(tl.float32)

                if USE_ACTIVATION:
                    if IS_VARLEN:
                        p_y = y + bos * stride_dy_t + t * stride_dy_t + o_d
                    else:
                        p_y = y + i_n * stride_dy_n + t * stride_dy_t + o_d
                    b_y = tl.load(p_y, mask=m_t, other=0).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))

                # Get weight[:, w_idx]
                b_w_col = tl.load(weight + o_d * W + w_idx, mask=m_d, other=0).to(tl.float32)

                # Accumulate
                b_dh0 += tl.where(m_t, b_dy * b_w_col, 0)

        # Store dh0[i_n, :, i_w]
        p_dh0 = dh0 + i_n * D * W + o_d * W + i_w
        tl.store(p_dh0, b_dh0.to(dh0.dtype.element_ty), mask=m_d)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def tmo_causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    stride_x_n,
    stride_x_t,
    stride_x_d,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    # o_d Shape: [BD]
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        seq_len = eos - bos
        p_x = x + bos * stride_x_t
    else:
        seq_len = T
        p_x = x + i_n * stride_x_n

    p_x = tl.make_block_ptr(p_x, (seq_len, D), (stride_x_t, stride_x_d), (seq_len - BW, i_d * BD), (BW, BD), (1, 0))

    # b_x Shape: [BW, BD]
    b_x = tl.load(p_x, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    if USE_INITIAL_STATE:
        if seq_len < BW:
            o_c = W - (BW - seq_len) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)

            p_init = initial_state + i_n * D*W + o_d[None, :] * W + o_c[:, None]
            mask_init = m_d[None, :] & m_c[:, None]

            b_cache = tl.load(p_init, mask=mask_init, other=0)
            b_x += b_cache

    # final_state: [N, D, W] (Channel Major inside sample)
    # o_w Shape: [BW]
    o_w = W - BW + tl.arange(0, BW)

    # o_d[:, None] -> [BD, 1]
    # o_w[None, :] -> [1, BW]
    # p_final Shape -> [BD, BW]
    p_final = final_state + i_n * D*W + o_d[:, None] * W + o_w[None, :]

    # m_final Shape -> [BD, BW]
    m_final = m_d[:, None] & (o_w[None, :] >= 0)

    tl.store(p_final, tl.trans(b_x).to(final_state.dtype.element_ty), mask=m_final)


def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if x.dim() == 2:
            stride_x_n = 0
            stride_x_t, stride_x_d = x.stride()
            T = x.shape[0]
        else:
            stride_x_n = x.stride(0)
            stride_x_t, stride_x_d = x.stride(1), x.stride(2)
            T = x.shape[1]
        D = x.shape[-1]
    else:
        B, T, D = x.shape
        N = B
        stride_x_n, stride_x_t, stride_x_d = x.stride()

    W = state_len
    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)

    BD = min(triton.next_power_of_2(D), 256)
    BW = triton.next_power_of_2(W)

    grid = (triton.cdiv(D, BD), N)

    tmo_causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        stride_x_n=stride_x_n,
        stride_x_t=stride_x_t,
        stride_x_d=stride_x_d,
        BW=BW,
        BD=BD,
    )
    return final_state


def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    shape = x.shape
    # if weight is not None and x.shape[-1] != weight.shape[0]:
    #     x = rearrange(x, 'b t ... -> b t (...)')

    D = x.shape[-1]
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BD = 8
    BW = triton.next_power_of_2(W)

    if x.dim() == 2:
        # Case: (N, D)
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(1)
    elif x.dim() == 3 and x.shape[0] == 1:
        # Case: (1, N, D) -> Time=1, Batch=N, Dim=D
        # Batch 在 dim 1
        stride_x_n = x.stride(1)
        stride_x_d = x.stride(2)
    elif x.dim() == 3:
        # Case: (N, 1, D) -> Batch=N, Time=1, Dim=D
        # Batch 在 dim 0
        stride_x_n = x.stride(0)
        stride_x_d = x.stride(2)
    else:
        # Fallback / Error case
        raise ValueError(f"Unsupported input shape: {x.shape}")

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    if y.dim() == 2:
        stride_y_n, stride_y_d = y.stride(0), y.stride(1)
    elif y.dim() == 3 and y.shape[0] == 1:
        stride_y_n, stride_y_d = y.stride(1), y.stride(2)
    elif y.dim() == 3:
        stride_y_n, stride_y_d = y.stride(0), y.stride(2)

    def grid(meta): return (triton.cdiv(D, meta['BD']), N)

    tmo_causal_conv1d_update_kernel[grid](
        x=x,
        cache=cache,
        residual=residual,
        y=y,
        weight=weight,
        bias=bias,
        stride_x_n=stride_x_n,
        stride_x_d=stride_x_d,
        stride_y_n=stride_y_n,
        stride_y_d=stride_y_d,
        D=D,
        W=W,
        BD=BD,
        BW=BW,
        ACTIVATION=activation,
        num_warps=STATIC_WARPS,
    )
    return y.view(shape), cache
