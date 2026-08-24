import torch
import inspect
from typing import Optional
from torch_mlu_ops._utils import add_gen_case_decorator
from torch_mlu_ops import triton

__CLASSES__ = []

__FUNCTIONS__ = [
    "reduce_to_label_cache",
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_fn",
    "causal_conv1d",
    "causal_conv1d_fn"
]

__all__ = __CLASSES__ + __FUNCTIONS__

def reduce_to_label_cache(k: torch.Tensor,
                          cu_seq_lens: torch.Tensor,
                          max_seq: int,
                          block_size: int,
                          k_max: Optional[torch.Tensor] = None,
                          k_min: Optional[torch.Tensor] = None,
                          min_paged_cache: Optional[torch.Tensor] = None,
                          max_paged_cache: Optional[torch.Tensor] = None,
                          slot_mapping: Optional[torch.Tensor] = None):
    """
    Reduce input sequences to label cache by computing max/min values per block.

    Math:
        for each sequence i in batch:
            split sequence into blocks of size block_size
            for each block j:
                k_max[i,j] = max(block_j_values)
                k_min[i,j] = min(block_j_values)
                if paged_cache_enabled:
                    store k_max/k_min to paged_cache[slot_mapping[i,j]]

    Args:
        k (torch.Tensor): Input sequence tensor. Shape is (total_tokens, head_num, head_size).
        cu_seq_lens (torch.Tensor): Cumulative sequence lengths. Shape is (batch_size + 1).
        k_max (torch.Tensor, optional): Output max cache. Shape is (batch_size, head_num, max_blocks, head_size).
        k_min (torch.Tensor, optional): Output min cache. Shape is (batch_size, head_num, max_blocks, head_size).
        min_paged_cache (torch.Tensor, optional): Paged cache for min values. Shape is (block_num, head_num, block_size, head_size).
        max_paged_cache (torch.Tensor, optional): Paged cache for max values. Shape is (block_num, head_num, block_size, head_size).
        slot_mapping (torch.Tensor, optional): Slot mapping indices. Shape is (batch_size, max_blocks).
        max_seq (int): Maximum sequence length.
        block_size (int): Size of each block.

    Types:
        k: half, bfloat16
        cu_seq_lens: int32
        k_max: same as k
        k_min: same as k
        min_paged_cache: same as k
        max_paged_cache: same as k
        slot_mapping: int32

    Returns:
        k_max (torch.Tensor): Computed max values. Shape is (batch_size, head_num, max_blocks, head_size).
        k_min (torch.Tensor): Computed min values. Shape is (batch_size, head_num, max_blocks, head_size).
        block_nums (torch.Tensor): Actual block counts per sequence. Shape is (batch_size).
        min_paged_cache (torch.Tensor): Updated paged cache for min values.
        max_paged_cache (torch.Tensor): Updated paged cache for max values.
    """
    assert cu_seq_lens.dim() == 1, "cu_seq_lens must be a 1-dimensional tensor"
    assert k.dim() == 3, "k must be a 3-dimensional tensor"
    B = cu_seq_lens.size(0) - 1
    H, D = k.size(1), k.size(2)
    max_blkn = max_seq // block_size
    if k_max is None:
        k_max = torch.empty(B, H, max_blkn, D, dtype=k.dtype, device=k.device)
    if k_min is None:
        k_min = torch.empty(B, H, max_blkn, D, dtype=k.dtype, device=k.device)
    block_nums = torch.empty(B, dtype=torch.int32, device=k.device)
    return triton.reduce_to_label_cache(k, cu_seq_lens, max_seq, block_size, k_max, k_min,
                                     min_paged_cache, max_paged_cache, slot_mapping, block_nums)

def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    allow_tf32: bool = True,
    state_v_first: bool = False):
    """
    Apply linear attention operation on q, k and v.

    Math:
        v_new = v_curr - alpha * S_pre * k_curr
        kk_t = beta * mm(v_new, k_t)
        S_curr = alpha * S_pre + kk_t
        O = scale * S_curr * q

    Args:
        q (torch.Tensor):
            The input query tensor. Shape should be [B, T, H, K].
        k (torch.Tensor):
            The input key tensor. Shape should be [B, T, H, K].
        v (torch.Tensor):
            The input value tensor. Shape should be [B, T, Hv, V].
        g (torch.Tensor):
            gating tensor. Shape should be [B, T, Hv].
        beta (torch.Tensor):
            gating tensor. Shape should be [B, T, Hv].
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`.
            Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, Hv, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, Hv, K, V]`. Default: `False`.
        allow_tf32 (Optional[bool]):
            Whether to allow tl.dot() operator to used tfloat32_t to accelerate matmul operation. Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Type:
        q: half, bfloat16.
        k: same as q.
        v: same as q.
        g: float, half, bfloat16.
        beta: float, half, bfloat16.
        scale: float.
        initial_state: float.
        output_final_state: True.
        cu_seqlens: int32.

    Return:
        o (torch.Tensor):
            Outputs of shape `[B, T, Hv, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, Hv, K, V]` if `output_final_state=True` else `None`.
    """
    from .triton.fla.chunk import ChunkGatedDeltaRuleFunction
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        None,
        allow_tf32,
        state_v_first,
    )
    return o, final_state
    # return triton.chunk_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens)

def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None):
    """
    Apply linear attention operation on q, k and v.

    Math:
        S[t] = alpha * S[t-1] + beta * (v[t] -alpha * S[t-1] @ k[t]) @ k_trans[t]
        O[t] = S[t] @ q[t]

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.
    """
    return triton.fused_recurrent_gated_delta_rule_fwd(q, k, v, g, gk, gv, beta, scale, initial_state,
                                                output_final_state, use_qk_l2norm_in_kernel, cu_seqlens)

def fused_recurrent_gated_delta_rule_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor = None,
    num_accepted_tokens: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False):
    """
    Apply linear attention operation on q, k and v.

    Math:
        S[t] = alpha * S[t-1] + beta * (v[t] -alpha * S[t-1] @ k[t]) @ k_trans[t]
        O[t] = S[t] @ q[t]

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        inplace_final_state: bool:
            Whether to store the final state in-place to save memory.
            Default: `True`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        ssm_state_indices (Optional[torch.Tensor]):
            Indices to map the input sequences to the initial/final states.
        num_accepted_tokens (Optional[torch.Tensor]):
            Number of accepted tokens for each sequence during decoding.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]`.
    """
    return triton.fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale, initial_state, inplace_final_state,
                            cu_seqlens, ssm_state_indices, num_accepted_tokens, use_qk_l2norm_in_kernel)

def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    backend: str | None = 'triton',
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    cp_context = None,
    **kwargs,
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (Optional[torch.Tensor]):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (Optional[torch.Tensor]):
            Bias tensor of shape [D]. Default: `None`.
        residual (Optional[torch.Tensor]):
            Residual tensor of shape [B, T, D]. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Only `'triton'` is supported.
            Default: `'triton'`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths (optional)
        chunk_indices (Optional[torch.LongTensor]):
            Chunk indices for variable-length sequences (optional)

    Returns:
        Tuple of (output, final_state).
        If `output_final_state` is `False`, the final state is `None`.
    """
    # Import here to avoid circular dependencies
    # from .triton.conv.triton.ops import CausalConv1dFunction

    if cp_context is not None:
        raise ValueError(f"[CausalConv1d] CP Context not supported.")
    if output_final_state is True:
        raise ValueError(f"[CausalConv1d] output_final_state true not supported.")

    if backend == 'triton':
        if weight is not None and x.shape[-1] != weight.shape[0]:
            raise ValueError(f"[CausalConv1d] x.shape[-1] must be equal to weight.shape[0].")

        y, final_state = triton.CausalConv1dFunctionFla.apply(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
        )
        return y, final_state
    else:
        raise ValueError(f"[CausalConv1d] Unsupported backend: {backend}")

def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = -1,
    null_block_id: int = 0,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if cache_indices are not provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block containing the initial state is located.
    num_computed_tokens: (batch,), dtype int32
        The number of tokens already completed for each sequence
    block_size_to_align: int
        The block size to align the cached states to
    out: same shape as `x`
    """

    if weight is not None and x.shape[0] != weight.shape[0]:
        raise ValueError(f"[CausalConv1d] x.shape[0] must be equal to weight.shape[0].")
    if weight is not None and weight.shape[-1] > 4:
        raise ValueError(f"[CausalConv1d] weight.shape[-1] must be less than or equal to 4.")

    y = triton.conv.causal_conv1d_fwd_vllm(x,
                                           weight,
                                           bias,
                                           conv_states,
                                           query_start_loc,
                                           cache_indices,
                                           has_initial_state,
                                           activation,
                                           pad_slot_id,
                                           null_block_id,
                                           block_idx_first_scheduled_token,
                                           block_idx_last_scheduled_token,
                                           initial_state_idx,
                                           num_computed_tokens,
                                           block_size_to_align,
                                           metadata,
                                           validate_data)
    return y


# add dump_gese as decorator automatically for every function mentioned in __all__
add_gen_case_decorator(inspect.currentframe())
