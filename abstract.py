from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
import torch._custom_ops


@torch._custom_ops.impl_abstract("torch_mlu_ops::flash_attention")
def flash_attention_abstract(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    output_lse: Optional[Tensor],
    cu_seq_lens_q: Optional[Tensor],
    cu_seq_lens_kv: Optional[Tensor],
    alibi_slope: Optional[Tensor],
    attn_bias: Optional[Tensor],
    q_quant_scale: Optional[Tensor],
    k_quant_scale: Optional[Tensor],
    v_quant_scale: Optional[Tensor],
    out_quant_scale: Optional[Tensor],
    block_tables: Optional[Tensor],
    max_seq_len_q: int,
    max_seq_len_kv: int,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    compute_dtype: str,
    return_lse: bool,
    q2k_block_idx: Optional[torch.Tensor],
    q2k_block_num: Optional[torch.Tensor],
    variable_block_sizes: Optional[torch.Tensor],
    q_block_size: int,
    k_block_size: int,
    sink: Optional[Tensor]
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::aot_flash_attention")
def aot_flash_attention_abstract(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seq_lens_q: Optional[Tensor],
    cu_seq_lens_kv: Optional[Tensor],
    alibi_slope: Optional[Tensor],
    attn_bias: Optional[Tensor],
    q_quant_scale: Optional[Tensor],
    k_quant_scale: Optional[Tensor],
    v_quant_scale: Optional[Tensor],
    out_quant_scale: Optional[Tensor],
    block_tables: Optional[Tensor],
    max_seq_len_q: int,
    max_seq_len_kv: int,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    compute_dtype: str,
    return_lse: bool,
    q2k_block_idx: Optional[torch.Tensor],
    q2k_block_num: Optional[torch.Tensor],
    variable_block_sizes: Optional[torch.Tensor],
    q_block_size: int,
    k_block_size: int,
    out_dtype: str,
    sink: Optional[Tensor]
) -> List[Tensor]:
    dtype = torch.half if out_dtype == "half" else (torch.float if out_dtype == "float" else torch.bfloat16)
    if v.dtype not in {torch.int8, torch.float8_e4m3fn}:
        tmo_out = torch.empty(q.size()[:-1] + (v.size()[-1],), dtype=v.dtype, device=q.device)
    elif v_quant_scale.dtype in {torch.float8_e8m0fnu, torch.bfloat16}:
        tmo_out = torch.empty(q.size()[:-1] + (v.size()[1],), dtype=dtype, device=q.device)  # v shape as [h,c,total_v]
    else:
        tmo_out = torch.empty(q.size()[:-1] + (v.size()[-1],), dtype=dtype, device=q.device)
    outs = [tmo_out]
    if return_lse:
        lse_shape = q.shape[:-3] + (q.shape[-2],) + (q.shape[-3],)
        out_lse = torch.empty(lse_shape, dtype=torch.float, device=q.device)
        outs.append(out_lse)
    return outs


@torch._custom_ops.impl_abstract("torch_mlu_ops::single_query_cached_kv_attn")
def single_query_cached_kv_attn_abstract(
    q_ori: Tensor,
    k_cache: Tensor,
    output: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    v_cache: Optional[Tensor],
    output_lse: Optional[Tensor],
    q_quant_scale: Optional[Tensor],
    k_cache_quant_scale: Optional[Tensor],
    v_cache_quant_scale: Optional[Tensor],
    out_quant_scale: Optional[Tensor],
    alibi_slopes: Optional[Tensor],
    attn_mask: Optional[Tensor],
    compute_dtype: str,
    max_context_len: int,
    windows_size_left: int,
    windows_size_right: int,
    softmax_scale: float,
    return_lse: bool,
    kv_cache_quant_bit_size: int,
    cu_seq_q: Optional[Tensor],
    max_seq_q: int,
    sink: Optional[Tensor]
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::apply_rotary")
def apply_rotary_abstract(
    input: Tensor,
    output: Tensor,
    sin_cache: Tensor,
    cos_cache: Tensor,
    position_ids: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    interleaved: bool,
    discrete: bool,
    dynamic_ntk: bool,
    max_seqlen: int,
    is_inverse: bool
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::reshape_linear_cache")
def reshape_linear_cache_abstract(
    key: Tensor,
    value: Optional[Tensor],
    key_cache: Tensor,
    value_cache: Optional[Tensor],
    context_lengths: Tensor,
    max_context_len: int,
    packed: bool,
    context_seq_offset: Optional[Tensor],
    cache_bs_id: Optional[Tensor],
    cache_seqlen_offset: Optional[Tensor],
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::reshape_paged_cache")
def reshape_paged_cache_abstract(
    k: Tensor, v: Optional[Tensor], k_cache: Tensor, v_cache: Optional[Tensor], slot_mapping: Tensor, direction: bool
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::multi_layer_kv_transfer")
def multi_layer_kv_transfer_abstract(
    key_value: Tensor, key_value_ptrs: Tensor, slot_mapping: Tensor, paged_memory_device: int,
    page_buffer_size: int, direction: int, kv_format: int, block_size: int, head_size: int,
    skip_prefix_n_tokens: int
) -> None:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::multi_layer_block_kv_transfer")
def multi_layer_block_kv_transfer_abstract(
    paged_buffer_ptrs: Tensor, lmcache_objects_ptrs: Tensor, block_ids: Tensor,
    paged_memory_device: int, direction: int, kv_size: int, block_num: int,
    block_size: int, num_heads: int, head_size: int, dtype_size: int,
    lmcache_chunk_size: int, kv_format: int, skip_prefix_n_blocks: int
) -> None:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::quant_to_paged_cache")
def quant_to_paged_cache_abstract(
    k: Tensor,
    v: Optional[Tensor],
    k_cache: Tensor,
    v_cache: Optional[Tensor],
    k_cache_scale: Tensor,
    v_cache_scale: Optional[Tensor],
    slot_mapping: Tensor,
) -> None:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::quant_mx_to_paged_cache")
def quant_mx_to_paged_cache_abstract(
    k: Tensor,
    v: Optional[Tensor],
    k_cache: Tensor,
    v_cache: Optional[Tensor],
    k_cache_scale: Tensor,
    v_cache_scale: Optional[Tensor],
    slot_mapping: Tensor,
    cu_seqlens: Optional[Tensor],
    recent_v: Optional[Tensor],
    recent_seqlens: Optional[Tensor],
    recent_slotmapping: Optional[Tensor],
    quant_bits: int,
) -> None:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::offline_quant_to_paged_cache")
def offline_quant_to_paged_cache_abstract(
    k: Tensor,
    v: Optional[Tensor],
    k_cache_scale_per_channel: Optional[Tensor],
    v_cache_scale_per_channel: Optional[Tensor],
    slot_mapping: Tensor,
    k_cache: Tensor,
    v_cache: Optional[Tensor],
    k_cache_scale_per_tensor: float,
    v_cache_scale_per_tensor: float,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::quant_to_linear_cache")
def quant_to_linear_cache_abstract(
    key: Tensor,
    value: Optional[Tensor],
    key_cache: Tensor,
    value_cache: Optional[Tensor],
    key_cache_scale: Tensor,
    value_cache_scale: Optional[Tensor],
    context_lengths: Tensor,
    max_context_len: int,
    packed: bool,
    context_seq_offset: Optional[Tensor],
    cache_bs_id: Optional[Tensor],
    cache_seqlen_offset: Optional[Tensor],
    quant_bit: int = 8,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::offline_quant_to_linear_cache")
def offline_quant_to_linear_cache_abstract(
    key: Tensor,
    value: Optional[Tensor],
    key_cache: Tensor,
    value_cache: Optional[Tensor],
    key_cache_scale: Optional[Tensor],
    value_cache_scale: Optional[Tensor],
    context_lengths: Tensor,
    max_context_len: int,
    quant_mode: int,
    packed: bool,
    context_seq_offset: Optional[Tensor],
    cache_bs_id: Optional[Tensor],
    cache_seqlen_offset: Optional[Tensor],
    key_cache_scale_per_tensor: float,
    value_cache_scale_per_tensor: float,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::swap_blocks")
def swap_blocks_abstract(
    dst: Tensor, src: Tensor, block_mapping: Dict[int, int], block_size_in_bytes: Optional[int] = None
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::copy_blocks")
def copy_blocks_abstract(k_caches: List[Tensor], v_caches: List[Tensor], block_mapping: Dict[int, List[int]]) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::copy_blocks_out_of_place")
def copy_blocks_out_of_place_abstract(
    k_caches: List[Tensor], v_caches: List[Tensor], block_mapping: Dict[int, List[int]]
) -> (List[Tensor], List[Tensor]):
    return ([torch.empty_like(k) for k in k_caches], [torch.empty_like(v) for v in v_caches])


@torch._custom_ops.impl_abstract("torch_mlu_ops::active")
def active_abstract(
    input: Tensor,
    output: Tensor,
    bias: Optional[Tensor],
    cusum_token_count: Optional[Tensor],
    act_mode: str,
    is_gated: bool,
    start_expert_id: int = 0,
    expert_size: int = 0,
    active_coef: float = 1.0,
    high_precision: bool = False,
    gelu_approximate: str = "none",
    swiglu_limit: int = 0,
    weight: Optional[Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_layernorm")
def fused_layernorm_abstract(
    input: Tensor,
    output: Tensor,
    residual: Optional[Tensor],
    gamma: Optional[Tensor],
    beta: Optional[Tensor],
    bias: Optional[Tensor],
    quant_scale: Optional[Tensor],
    residual_out: Optional[Tensor],
    smooth_quant_scale: Optional[Tensor],
    normed_out: Optional[Tensor],
    norm_mode: str,
    eps: float,
    store_output_before_norm: bool,
    store_output_after_norm: bool,
    dynamic_quant: bool,
    mx_quant: bool = False,
    transpose_4d_1_2: bool = False
)-> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::layernorm_forward")
def layernorm_forward_abstract(
    input: Tensor, output: Tensor, gamma: Optional[Tensor], beta: Optional[Tensor], eps: float, gamma_add_coef: float
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::matmul")
def matmul_abstract(
    a: Tensor,
    b: Tensor,
    bias: Optional[Tensor],
    c: Optional[Tensor],
    a_scale_tensor: Optional[Tensor],
    b_scale_tensor: Optional[Tensor],
    data_type: Optional[str],
    tile_config: Optional[Dict[str, int]],
    act_mode: str,
    alpha: float,
    beta: float,
    fast_act: bool,
    approximate: bool,
    a_scale: float,
    b_scale: float,
    trans_a: bool,
    trans_b: bool,
) -> Tensor:
    if a.dim() == 3:
        batch = max(a.size(0), b.size(0))
        m = a.size(2) if trans_a else a.size(1)
        n = b.size(1) if trans_b else b.size(2)
    else:
        m = a.size(1) if trans_a else a.size(0)
        n = b.size(0) if trans_b else b.size(1)
    if data_type is None:
        output_type = a.dtype
    elif data_type == "float":
        output_type = torch.float32
    elif data_type == "bfloat16":
        output_type = torch.bfloat16
    else:
        output_type = torch.half
    if a.dim() == 3:
        return torch.empty(batch, m, n, dtype=output_type, device=a.device)
    return torch.empty(m, n, dtype=output_type, device=a.device)


@torch._custom_ops.impl_abstract("torch_mlu_ops::matmul_aot_inductor")
def matmul_aot_inductor_abstract(
    a: Tensor,
    b: Tensor,
    bias: Optional[Tensor],
    c: Optional[Tensor],
    a_scale_tensor: Optional[Tensor],
    b_scale_tensor: Optional[Tensor],
    data_type: Optional[str],
    act_mode: str,
    alpha: float,
    beta: float,
    fast_act: bool,
    approximate: bool,
    a_scale: float,
    b_scale: float,
    trans_a: bool,
    trans_b: bool,
) -> Tensor:
    if a.dim() == 3:
        batch = max(a.size(0), b.size(0))
        m = a.size(2) if trans_a else a.size(1)
        n = b.size(1) if trans_b else b.size(2)
    else:
        m = a.size(1) if trans_a else a.size(0)
        n = b.size(0) if trans_b else b.size(1)
    if data_type is None:
        output_type = a.dtype
    elif data_type == "float":
        output_type = torch.float32
    elif data_type == "bfloat16":
        output_type = torch.bfloat16
    else:
        output_type = torch.half
    if a.dim() == 3:
        return torch.empty(batch, m, n, dtype=output_type, device=a.device)
    return torch.empty(m, n, dtype=output_type, device=a.device)


@torch._custom_ops.impl_abstract("torch_mlu_ops::matmul_inplace")
def matmul_inplace_abstract(
    a: Tensor,
    b: Tensor,
    output: Tensor,
    bias: Optional[Tensor],
    c: Optional[Tensor],
    a_scale_tensor: Optional[Tensor],
    b_scale_tensor: Optional[Tensor],
    tile_config: Optional[Dict[str, int]],
    act_mode: str,
    alpha: float,
    beta: float,
    fast_act: bool,
    approximate: bool,
    a_scale: float,
    b_scale: float,
    trans_a: bool,
    trans_b: bool,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::matmul_v2")
def matmul_v2_abstract(
    a: Tensor,
    b: Tensor,
    output: Tensor,
    bias: Optional[Tensor],
    c: Optional[Tensor],
    a_scale_tensor: Optional[Tensor],
    b_scale_tensor: Optional[Tensor],
    act_mode: str,
    alpha: float,
    beta: float,
    fast_act: bool,
    approximate: bool,
    a_scale: float,
    b_scale: float,
    trans_a: bool,
    trans_b: bool,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::batch_matmul_inplace")
def batch_matmul_inplace_abstract(
    a: Tensor,
    b: Tensor,
    d: Tensor,
    c: Optional[Tensor],
    bias: Optional[Tensor],
    a_scale_tensor: Optional[Tensor],
    b_scale_tensor: Optional[Tensor],
    act_mode: str,
    alpha: float,
    beta: float,
    a_scale: float,
    b_scale: float,
    trans_a: bool,
    trans_b: bool,
    use_hp_active: bool,
    approximate: bool,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::batch_matmul")
def batch_matmul_abstract(
    a: Tensor,
    b: Tensor,
    c: Optional[Tensor],
    bias: Optional[Tensor],
    dtype: Optional[str],
    a_scale_tensor: Optional[Tensor],
    b_scale_tensor: Optional[Tensor],
    act_mode: str,
    alpha: float,
    beta: float,
    a_scale: float,
    b_scale: float,
    trans_a: bool,
    trans_b: bool,
    use_hp_active: bool,
    approximate: bool,
) -> Tensor:
    batch = max(a.size(0), b.size(0))
    m = a.size(2) if trans_a else a.size(1)
    n = b.size(1) if trans_b else b.size(2)
    if dtype is None:
        output_type = a.dtype
    elif dtype == "float":
        output_type = torch.float32
    elif dtype == "bfloat16":
        output_type = torch.bfloat16
    else:
        output_type = torch.half
    return torch.empty(batch, m, n, dtype=output_type, device=a.device)


@torch._custom_ops.impl_abstract("torch_mlu_ops::group_gemm")
def group_gemm_abstract(
    a: Tensor,
    b: Tensor,
    dim_list: Tensor,
    d: Tensor,
    expand_idx: Optional[Tensor],
    c: Optional[Tensor],
    alpha: Optional[Tensor],
    beta: Optional[Tensor],
    a_scale: Optional[Tensor],
    b_scale: Optional[Tensor],
    bias: Optional[Tensor],
    a_calibration: Optional[Tensor],
    b_calibration: Optional[Tensor],
    quant_flag: Optional[List],
    b_offset: Optional[Tensor],
    tile_config: Dict[str, int],
    max_dim: int,
    trans_a: bool,
    trans_b: bool,
    a_quant_bit: int,
    a_lora: Optional[Tensor] = None,
    b_lora: Optional[Tensor] = None,
    idx_offset: Optional[Tensor] = None,
    allow_tf32: bool = False,
    is_symmetric_quant: bool = True
) -> Tensor:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::group_gemm_v2")
def group_gemm_v2_abstract(
    a: Tensor,
    b: Tensor,
    dim_list: Tensor,
    d: Tensor,
    expand_idx: Optional[Tensor],
    c: Optional[Tensor],
    alpha: Optional[Tensor],
    beta: Optional[Tensor],
    a_scale: Optional[Tensor],
    b_scale: Optional[Tensor],
    bias: Optional[Tensor],
    a_calibration: Optional[Tensor],
    b_calibration: Optional[Tensor],
    quant_flag: Optional[Tensor],
    b_offset: Optional[Tensor],
    max_dim: int,
    trans_a: bool,
    trans_b: bool,
    a_quant_bit: int,
    a_lora: Optional[Tensor] = None,
    b_lora: Optional[Tensor] = None,
    idx_offset: Optional[Tensor] = None,
    allow_tf32: bool = False,
    is_symmetric_quant: bool = True
) -> Tensor:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::preload")
def preload_abstract(weight: Tensor, size: int) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_active_topk")
def moe_active_topk_abstract(
    input,
    topk,
    num_expert_group,
    topk_group,
    normalize,
    mask: Optional[torch.Tensor],
    normed_by: str,
    act_type: str,
    route_scale: float,
    score_bias: Optional[torch.Tensor],
    reduce_weight: Optional[torch.Tensor],
    expert_id: Optional[torch.Tensor],
) -> None:
    out_shape = list(input.size())[:-1] + [topk]
    if reduce_weight is None:
        reduce_weight = torch.empty(out_shape, dtype=torch.float32, device=input.device)
    if expert_id is None:
        expert_id = torch.empty(out_shape, dtype=torch.int, device=input.device)
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_softplus_topk")
def moe_softplus_topk_abstract(input,
                               input_ids: Optional[torch.Tensor],
                               tid2eid: Optional[torch.Tensor],
                               bias: Optional[torch.Tensor],
                               topk,
                               route_scale,
                               reduce_weight: Optional[torch.Tensor],
                               expert_id: Optional[torch.Tensor]) -> None:
    out_shape = list(input.size())[:-1] + [topk]
    if reduce_weight is None:
        reduce_weight = torch.empty(out_shape, dtype=torch.float32, device=input.device)
    if expert_id is None:
        expert_id = torch.empty(out_shape, dtype=torch.int, device=input.device)
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_expand_input")
def moe_expand_input_abstract(
    input: Tensor,
    gather_idx: Tensor,
    cusum_token_count: Optional[Tensor] = None,
    start_expert_id: int = 0,
    expert_size: int = 0,
) -> Tensor:
    return torch.empty(gather_idx.size(0), input.size(-1), dtype=input.dtype, device=input.device)


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_expand_input_inplace")
def moe_expand_input_inplace_abstract(
    input: Tensor,
    gather_idx: Tensor,
    cusum_token_count: Optional[Tensor],
    start_expert_id: int,
    expert_size: int,
    output,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_gen_idx")
def moe_gen_idx_abstract(
    expert_id: Tensor, expert_num: int, return_token2expert_idx: bool = False
) -> List[torch.Tensor]:
    token_num, topk = expert_id.size(0), expert_id.size(1)
    expand_idx = torch.empty((token_num * topk), dtype=torch.int32, device=expert_id.device)
    combine_idx = torch.empty((token_num * topk), dtype=torch.int32, device=expert_id.device)
    token_count = torch.empty((expert_num,), dtype=torch.int32, device=expert_id.device)
    cusum_token_count = torch.empty((expert_num + 1,), dtype=torch.int32, device=expert_id.device)

    outs = [expand_idx, combine_idx, token_count, cusum_token_count]
    if return_token2expert_idx:
        token2expert_idx = torch.empty((token_num * topk), dtype=torch.int32, device=expert_id.device)
        outs.append(token2expert_idx)

    return outs


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_combine_result")
def moe_combine_result_abstract(
    input: torch.Tensor,
    output: torch.Tensor,
    reduce_weight: torch.Tensor,
    gather_ids: torch.Tensor,
    residual: Optional[torch.Tensor],
    cusum_token_count: Optional[torch.Tensor],
    start_expert_id: int,
    expert_size: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_all2all_gen_send_layout")
def moe_all2all_gen_send_layout_abstract(input: torch.Tensor, nrank: int) -> torch.Tensor:
    return torch.empty(nrank, 2, dtype=input.dtype, device=input.device)


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_rope")
def fused_rope_abstract(
    qkv: torch.Tensor,
    key_cache_hp: torch.Tensor,
    value_cache_hp: torch.Tensor,
    key_cache_lp: Optional[torch.Tensor],
    value_cache_lp: Optional[torch.Tensor],
    sin_table: torch.Tensor,
    cos_table: torch.Tensor,
    position_ids: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor],
    key_scale_hp: Optional[torch.Tensor],
    value_scale_hp: Optional[torch.Tensor],
    key_scale_lp: Optional[torch.Tensor],
    value_scale_lp: Optional[torch.Tensor],
    cache_bs_id_hp: Optional[torch.Tensor],
    cache_seq_offsets_hp: Optional[torch.Tensor],
    cache_bs_id_lp: Optional[torch.Tensor],
    cache_seq_offsets_lp: Optional[torch.Tensor],
    slot_mapping_hp: Optional[torch.Tensor],
    slot_mapping_lp: Optional[torch.Tensor],
    norm_type: Optional[str],
    rope_dim_offset: Optional[int],
    eps: float,
    q_gamma: Optional[torch.Tensor],
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_cast_gating")
def moe_cast_gating_abstract(input: torch.Tensor, weight: torch.Tensor) -> Tensor:
    output_shape = input.shape[:-1] + (weight.shape[0],)
    output = torch.empty(output_shape, dtype=torch.float, device="mlu")
    return output


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_cast_gating_v2")
def moe_cast_gating_v2_abstract(
    input: torch.Tensor, weight0: torch.Tensor, weight1: torch.Tensor, alpha: float
) -> Tensor:
    output_shape = input.shape[:-1] + (weight0.shape[0],)
    output = torch.empty(output_shape, dtype=torch.float, device="mlu")
    return output


@torch._custom_ops.impl_abstract("torch_mlu_ops::update_out_and_lse")
def update_out_and_lse_abstract(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    seq_offsets: Optional[torch.Tensor] = None,
    cu_seqs: Optional[torch.Tensor] = None,
    block_cu_seqs: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::dequant_from_linear_cache")
def dequant_from_linear_cache_abstract(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: Optional[torch.Tensor],
    key_cache_quant_scale: torch.Tensor,
    value_cache_quant_scale: Optional[torch.Tensor],
    context_lengths: torch.Tensor,
    max_context_len: int,
    context_seq_offset: Optional[torch.Tensor],
    cache_bs_id: Optional[torch.Tensor],
    cache_seq_offset: Optional[torch.Tensor],
    quant_mode: int = 0,
    quant_bit: int = 8,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::dequant_from_paged_cache")
def dequant_from_paged_cache_abstract(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: Optional[torch.Tensor],
    key_cache_quant_scale: torch.Tensor,
    value_cache_quant_scale: Optional[torch.Tensor],
    context_lengths: torch.Tensor,
    max_context_len: int,
    context_seq_offset: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    quant_mode: int = 0,
    quant_bit: int = 8,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::dynamic_per_channel_quant")
def dynamic_per_channel_quant_abstract(
    input: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq: int,
    quant_out: torch.Tensor,
    quant_scale: torch.Tensor,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::reshape_from_cache")
def reshape_from_cache_abstract(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: Optional[torch.Tensor],
    context_lengths: torch.Tensor,
    max_context_len: int,
    context_seq_offset: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    cache_seq_offset: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_append_shared_expert")
def moe_append_shared_expert_abstract(
    reduce_weight: torch.Tensor,
    expert_id: torch.Tensor,
    num_expert: int,
    shared_expert_num: int,
    world_size: int,
    parallel_mode: str = "ep",
) -> Tuple[torch.Tensor]:
    topk = reduce_weight.size(-1)
    token_num = reduce_weight.numel() // topk
    new_reduce_weight = torch.empty(token_num, topk + shared_expert_num, dtype=torch.float32, device=reduce_weight.device)
    new_expert_id = torch.empty(new_reduce_weight.shape, dtype=torch.int32, device=reduce_weight.device)
    return (new_reduce_weight, new_expert_id)


@torch._custom_ops.impl_abstract("torch_mlu_ops::pow2")
def pow2_abstract(output: torch.Tensor, input: torch.Tensor) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::scaled_quantize")
def scaled_quantize_abstract(
    x: torch.Tensor,
    output: torch.Tensor,
    output_scale: Optional[torch.Tensor],
    scale: Optional[torch.Tensor] = None,
    zero: Optional[torch.Tensor] = None,
    m_list: Optional[torch.Tensor] = None,
    gather_idx: Optional[torch.Tensor] = None,
    gather_index_start_position: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None,
    quant_type: torch.dtype = torch.int8,
    quant_mode: str = "static",
    act_mode: str = "none",
    active_coef: float = 1,
    is_gated: bool = False,
    quant_bit_size: int = 8,
    need_output_scale_trans: bool = False,
    output_reduced: Optional[torch.Tensor] = None,
    group_size: int = 1,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::scaled_matmul")
def scaled_matmul_abstract(
    output: Tensor,
    a_tensor: Tensor,
    b_tensor: Tensor,
    a_scale: Optional[Tensor],
    a_zero: Optional[Tensor],
    a_calib: Optional[Tensor],
    b_scale: Optional[Tensor],
    b_zero: Optional[Tensor],
    b_calib: Optional[Tensor],
    bias: Optional[Tensor],
    c_tensor: Optional[Tensor],
    c_scale: Optional[Tensor],
    c_zero: Optional[Tensor],
    gemm_output_scale: Optional[Tensor],
    gemm_output_zero: Optional[Tensor],
    quant_algo: str,
    a_quant_layout: str,
    b_quant_layout: str,
    a_quant_bit_size: int = -1,
    b_quant_bit_size: int = 8,
    act_mode: str = "none",
    use_hp_active: bool = False,
    act_coef: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    trans_a: bool = False,
    trans_b: bool = True,
) -> None:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::scaled_matmul_tile")
def scaled_matmul_tile_abstract(
    output: Tensor,
    a_tensor: Tensor,
    b_tensor: Tensor,
    a_scale: Optional[Tensor],
    a_zero: Optional[Tensor],
    a_calib: Optional[Tensor],
    b_scale: Optional[Tensor],
    b_zero: Optional[Tensor],
    b_calib: Optional[Tensor],
    bias: Optional[Tensor],
    c_tensor: Optional[Tensor],
    c_scale: Optional[Tensor],
    c_zero: Optional[Tensor],
    gemm_output_scale: Optional[Tensor],
    gemm_output_zero: Optional[Tensor],
    quant_algo: str,
    a_quant_layout: str,
    b_quant_layout: str,
    a_quant_bit_size: int = -1,
    b_quant_bit_size: int = 8,
    act_mode: str = "none",
    use_hp_active: bool = False,
    act_coef: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    trans_a: bool = False,
    trans_b: bool = True,
    tile_config: Optional[Dict[str, int]]=None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::ssparse_matmul")
def ssparse_matmul_abstract(
    a: Tensor,
    b: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    output: Tensor,
    act_mode: str,
    m_list: Optional[Tensor],
    gather_idx: Optional[Tensor],
    bias: Optional[Tensor],
    c: Optional[Tensor],
    max_m: int,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans_a: bool = False,
    trans_b: bool = True,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_indexer_q")
def fused_indexer_q_abstract(
    q: torch.Tensor,
    output: torch.Tensor,
    output_scale: Optional[torch.Tensor],
    w_q: torch.Tensor,
    w_q_scale: Optional[torch.Tensor],
    hadamard_matrix: Optional[torch.Tensor],
    sin: torch.Tensor,
    cos: torch.Tensor,
    position_id: torch.Tensor,
    output_quant_mode: str,
    interleaved: bool,
    rope_at_front: bool,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_mla_q")
def fused_mla_q_abstract(
    input: torch.Tensor,
    output: Optional[torch.Tensor],
    output_scale: Optional[torch.Tensor],
    output_norm: Optional[torch.Tensor],
    gamma: torch.Tensor,
    smooth_quant_scale: torch.Tensor,
    weight_b: torch.Tensor,
    weight_b_scale: torch.Tensor,
    weight_c: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    position_id: torch.Tensor,
    quant_mode: str,
    eps: float,
    interleaved: bool,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_mla_kv")
def fused_mla_kv_abstract(
    kv: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    position_id: torch.Tensor,
    gamma: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_cache_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_bs_id: torch.Tensor,
    cache_seq_offset: torch.Tensor,
    is_paged_cache: bool,
    eps: float,
    interleaved: bool,
    quant_mode: str,
) -> Tuple[torch.Tensor]:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::quant_per_block")
def quant_per_block_abstract(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    seq_lens_q: torch.Tensor,
    seq_lens_k: Optional[torch.Tensor],
    seq_lens_v: Optional[torch.Tensor],
    max_seq_q: int,
    max_seq_k: int,
    max_seq_v: int,
    block_size_q: int,
    block_size_k: int,
    smooth_k: bool,
    quant_q: torch.Tensor,
    q_scale: torch.Tensor,
    quant_k: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    quant_v: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    k_mean: Optional[torch.Tensor],
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::quant_mx_qkv")
def quant_mx_qkv_abstract(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    quant_q: torch.Tensor,
    q_scale: torch.Tensor,
    quant_k: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    quant_v: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    k_mean: Optional[torch.Tensor],
    cu_seq_lens_q: Optional[torch.Tensor],
    cu_seq_lens_kv: Optional[torch.Tensor],
    max_seq_q: int,
    max_seq_kv: int,
    smooth_k: bool,
    trans_v: bool,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::flat_quant")
def flat_quant_abstract(
    input: torch.Tensor,
    affine_weight_left: torch.Tensor,
    affine_weight_right: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    clip_factor_max: Optional[torch.Tensor],
    clip_factor_min: Optional[torch.Tensor],
    quant_mode: int,
    asym_quant: bool,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    output_calibration: torch.Tensor,
    fast_compute: bool = False,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::svd_quant")
def svd_quant_abstract(
    input: torch.Tensor,
    weight_lora_down: torch.Tensor,
    smooth: torch.Tensor,
    lora_scales: torch.Tensor,
    output_lora_down: torch.Tensor,
    output_quanted: torch.Tensor,
    output_quant_scales: torch.Tensor,
    quant_mode: int,
    asym_quant: bool,
    quant_dtype: str,
    act_mode: str = "none",
    active_coef: float = 1.0,
    is_gated: bool = False,
    m_list: Optional[torch.Tensor] = None,
    gather_index: Optional[torch.Tensor] = None,
    gather_index_start_position: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::quant_conv3d")
def quant_conv3d_abstract(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    stride: Optional[Tuple[int]],
    padding: Optional[Tuple[int]],
    dilation: Optional[Tuple[int]],
    groups: int,
    compute_dtype: str,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::rejection_sample")
def rejection_sample_abstract(
    output_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    uniform_rand: torch.Tensor,
    uniform_probs: torch.Tensor,
    max_spec_len: int,
    high_acc: bool = True,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::masked_dot_select_sparse_paged_kv")
def masked_dot_select_sparse_paged_kv_abstract(
    q_low_rank: torch.Tensor,
    label_cache: torch.Tensor,
    context_lens: torch.Tensor,
    label_cache_block_table: torch.Tensor,
    kv_cache_block_table: torch.Tensor,
    recent_window: int,
    kv_cache_blk_size: int,
    sparse_kv_length: int,
    sparse_block_table: torch.Tensor,
    sparse_context_lens: torch.Tensor,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::masked_indexer_select_paged_kv")
def masked_indexer_select_paged_kv_abstract(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    kv_cache_block_table: torch.Tensor,
    cu_seq_q_lens: Optional[torch.Tensor],
    cu_seq_k_lens: Optional[torch.Tensor],
    k_context_lens: Optional[torch.Tensor],
    k_cache_block_table: Optional[torch.Tensor],
    is_prefill: bool,
    index_topk: int,
    kv_cache_block_size: int,
    softmax_scale: float,
    q_scale: Optional[torch.Tensor] = None,
    k_scale_cache: Optional[torch.Tensor] = None,
    sparse_block_table: Optional[torch.Tensor] = None,
    sparse_context_lens: Optional[torch.Tensor] = None,
    is_score_float: bool = False,
    compress_ratio: int = 1,
    kv_cache_block_table_offset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::apply_topkp")
def apply_topkp_abstract(
    logits: torch.Tensor,
    index_in: torch.Tensor,
    per_slice_k: List,
    per_slice_p: List,
    logits_output: Optional[torch.Tensor] = None,
    sorted_logits_out: Optional[torch.Tensor] = None,
    index_out: Optional[torch.Tensor] = None,
    true_select_len: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::random_sample")
def random_sample_abstract(
    probs: torch.Tensor, is_gumbel_max: bool, generators: dict[int, torch.Generator] = None
) -> torch.Tensor:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::apply_topkp_v2")
def apply_topkp_v2_abstract(
    logits: torch.Tensor,
    index_in: torch.Tensor,
    temperature_list: torch.Tensor,
    minp_list: torch.Tensor,
    topk_list: torch.Tensor,
    topp_list: torch.Tensor,
    logits_output: Optional[torch.Tensor] = None,
    sorted_logits_out: Optional[torch.Tensor] = None,
    index_out: Optional[torch.Tensor] = None,
    true_select_len: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::index_selected_rope")
def index_selected_rope_abstract(
    input: Tensor, output: Tensor, sin_table: Tensor, cos_table: Tensor, ids: Tensor, input_discrete_only: bool
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::gather_split")
def gather_split_abstract(
    input: torch.Tensor,
    gather_index: torch.Tensor,
    valid_token_num: torch.Tensor,
    output1: torch.Tensor,
    output2: Optional[torch.Tensor] = None,
    output3: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::hshare")
def hshare_abstract(
    hshare_block_tables: torch.Tensor,
    hshare_kv_len_after_store: torch.Tensor,
    block_table: torch.Tensor,
    kv_len_after_store: torch.Tensor,
    disable_hshare_layer: Optional[torch.Tensor],
    ratios: torch.Tensor,
    indices_cache: torch.Tensor,
    indices_cache_offset: torch.Tensor,
    block_num_cache: torch.Tensor,
    actual_batch_size: int,
    max_seq_len: int,
    block_size: int,
    layer_num: int,
    kv_head_num: int,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_all2all_gen_gather_index")
def moe_all2all_gen_gather_index_abstract(
    gather_by_expert_index: torch.Tensor,
    gather_by_rank_index: torch.Tensor,
    token_count: torch.Tensor,
    cusum_token_count: torch.Tensor,
    token_sum: torch.Tensor,
    token_num: torch.Tensor,
    pad_num: int,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::compress_kv")
def compress_kv_abstract(
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seq_lens_opt: Optional[torch.Tensor],
    wk_cmp: torch.Tensor,
    wv_cmp: torch.Tensor,
    pe_table_k: torch.Tensor,
    pe_table_v: torch.Tensor,
    max_seq_len: int,
    compress_length: int,
    compress_stride: int,
    k_out: Optional[torch.Tensor],
    v_out: Optional[torch.Tensor],
    compress_lens_out: Optional[torch.Tensor],
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_mul_reduce_sum")
def fused_mul_reduce_sum_abstract(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    c = x.size(2)
    return torch.empty((n, c), dtype=x.dtype, device=x.device)

@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_compress_single_kv")
def fused_compress_single_kv_abstract(kv: torch.Tensor,
                                      score: torch.Tensor,
                                      position: torch.Tensor,
                                      ape: torch.Tensor,
                                      gamma: torch.Tensor,
                                      sin: torch.Tensor,
                                      cos: torch.Tensor,
                                      hadamard_matrix: Optional[torch.Tensor],
                                      slot_mapping: torch.Tensor,
                                      kv_cache: torch.Tensor,
                                      kv_cache_scale: Optional[torch.Tensor],
                                      eps: float,
                                      overlap: bool,
                                      state_cache: torch.Tensor,
                                      state_bt: torch.Tensor,
                                      state_width: int,
                                      state_block_size: int,
                                      cu_query_len: torch.Tensor,
                                      K: int = 0):
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::update_compressor_states")
def update_compressor_states_abstract(
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    accept_tokens: torch.Tensor,
    batch_to_kv_state: torch.Tensor,
    positions: torch.Tensor,
    cu_query_len: torch.Tensor,
    overlap: bool,
    K: int,
) -> None:
    # In-place operation: updates kv_state and score_state directly
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_all2all_create")
def moe_all2all_create_abstract(
    max_dispatch_token_byte: int,
    max_combine_token_byte: int,
    max_expert_num: int,
    max_token_num: int,
    rank: int,
    nrank: int,
    place_holder: torch.Tensor,
) -> List[torch.Tensor]:
    handle_tensor = torch.empty((2,), dtype=torch.int64, device="cpu")
    dispatch_info = torch.empty((handle_tensor[1],), dtype=torch.int8, device="cpu")
    dispatch_send = torch.empty((max_dispatch_token_byte * max_token_num,), dtype=torch.int8, device="mlu")
    dispatch_recv = torch.empty((nrank * max_dispatch_token_byte * max_token_num,), dtype=torch.int8, device="mlu")
    combine_send = torch.empty((max_combine_token_byte * max_token_num,), dtype=torch.int8, device="mlu")
    combine_recv = torch.empty((nrank * max_combine_token_byte * max_token_num,), dtype=torch.int8, device="mlu")
    return (handle_tensor, dispatch_info, dispatch_send, dispatch_recv, combine_send, combine_recv)


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_all2all_init")
def moe_all2all_init_abstract(all2all_handle: int, all_exchange_info: torch.Tensor, place_holder: torch.Tensor) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_all2all_dispatch")
def moe_all2all_dispatch_abstract(
    all2all_handle: int,
    token_byte: int,
    token_num: int,
    send_layout: torch.Tensor,
    send_token_num: torch.Tensor,
    recv_layout: torch.Tensor,
    recv_token_num: torch.Tensor,
    send_token: Optional[torch.Tensor] = None,
    recv_token: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_all2all_combine")
def moe_all2all_combine_abstract(
    all2all_handle: int,
    token_byte: int,
    token_num: int,
    send_src_layout: torch.Tensor,
    send_dst_layout: torch.Tensor,
    send_token: Optional[torch.Tensor] = None,
    recv_token: Optional[torch.Tensor] = None,
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::moe_all2all_destroy")
def moe_all2all_destroy_abstract(all2all_handle: int, place_holder: torch.Tensor) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_indexer_k")
def fused_indexer_k_abstract(
    x: torch.Tensor,
    wk: torch.Tensor,
    wproj: torch.Tensor,
    sin_table: torch.Tensor,
    cos_table: torch.Tensor,
    position_ids: torch.Tensor,
    slot_mapping: torch.Tensor,
    haed_weights: torch.Tensor,
    k_cache: torch.Tensor,
    k_cache_scale: torch.Tensor = None,
    hadamard_matrix: Optional[torch.Tensor] = None,
    interleaved: bool = True,
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> None:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::hc_split_sinkhorn")
def hc_split_sinkhorn_abstract(mixes: Tensor,
                               hc_scale: Tensor,
                               hc_base: Tensor,
                               pre_scale: Optional[torch.Tensor] = None,
                               hc_mult: int = 4,
                               sinkhorn_iter: int = 20,
                               eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (mixes.dim() == 3):
        B, S = mixes.size(0), mixes.size(1)
        pre = torch.empty((B, S, hc_mult), dtype=torch.float32, device=mixes.device)
        post = torch.empty((B, S, hc_mult), dtype=torch.float32, device=mixes.device)
        comb = torch.empty((B, S, hc_mult, hc_mult), dtype=torch.float32, device=mixes.device)
        return (pre, post, comb)
    elif (mixes.dim() == 2):
        B = mixes.size(0)
        pre = torch.empty((B, hc_mult), dtype=torch.float32, device=mixes.device)
        post = torch.empty((B, hc_mult), dtype=torch.float32, device=mixes.device)
        comb = torch.empty((B, hc_mult, hc_mult), dtype=torch.float32, device=mixes.device)
        return (pre, post, comb)

@torch._custom_ops.impl_abstract("torch_mlu_ops::solve_tril")
def solve_tril_abstract(
    input: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> None:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::transpose_all2all")
def transpose_all2all_abstract(
    cncl_comm: int,
    pre_num_block: int,
    pre_block_count: int,
    post_num_block: int,
    post_block_count: int,
    send: torch.Tensor,
    recv: torch.Tensor,
):
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::convert_vertical_slash_index")
def convert_vertical_slash_index_abstract(
    seqlens: torch.Tensor,
    ctxlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    max_seqlen_q: int,
    block_size_M: int,
    block_size_N: int,
):
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (max_seqlen_q + block_size_M - 1) // block_size_M
    block_count = torch.empty(batch_size, num_heads, num_rows, dtype=seqlens.dtype, device=seqlens.device)
    block_offset = torch.empty(batch_size, num_heads, num_rows, nnz_slash, dtype=seqlens.dtype, device=seqlens.device)
    column_count = torch.empty(batch_size, num_heads, num_rows, dtype=seqlens.dtype, device=seqlens.device)
    column_index = torch.empty(
        batch_size, num_heads, num_rows, nnz_vertical, dtype=seqlens.dtype, device=seqlens.device
    )
    return (block_count, block_offset, column_count, column_index)


@torch._custom_ops.impl_abstract("torch_mlu_ops::hamming_score")
def hamming_score_abstract(
    query_code: Tensor,
    key_codes: Tensor,
    block_table_opt: Optional[Tensor],
    seq_len: Tensor,
    max_seq_len: int,
    sink: int,
    recent: int,
) -> Tensor:
    return torch.empty(query_code.size(0), 1, max_seq_len, dtype=torch.float16, device=query_code.device)


@torch._custom_ops.impl_abstract("torch_mlu_ops::concat_block_table")
def concat_block_table_abstract(
    first_block_table: torch.Tensor,
    first_context_lens: torch.Tensor,
    second_block_table: torch.Tensor,
    second_context_lens: torch.Tensor,
    new_block_table: Optional[torch.Tensor] = None,
    new_context_lens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_mhc_post")
def fused_mhc_post_abstract(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    output: torch.Tensor,
    output_rms: Optional[torch.Tensor],
    compute_rms: bool,
    eps: float,
):
    return None


@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_compress_multi_kv")
def fused_compress_multi_kv_abstract(
    kv: torch.Tensor,
    score: torch.Tensor,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    positions: torch.Tensor,
    ape: torch.Tensor,
    max_seqlen: int,
    overlap: bool,
    compressed_kv: torch.Tensor,
):
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::fused_mla_q_v2")
def fused_mla_q_v2_abstract(input_q: torch.Tensor,
                output: Optional[torch.Tensor],
                output_norm: Optional[torch.Tensor],
                gamma: torch.Tensor,
                smooth_quant_scale: Optional[torch.Tensor],
                weight_b: torch.Tensor,
                weight_b_scale: Optional[torch.Tensor],
                sin: torch.Tensor,
                cos: torch.Tensor,
                position_id: torch.Tensor,
                eps: float,
                interleaved: bool) -> None:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::get_compress_block_tables")
def get_compress_block_tables_abstract(compress_block_tables: torch.Tensor,
                                     compress_context_lens: torch.Tensor,
                                     seq_k_lens: torch.Tensor,
                                     query_start_loc: torch.Tensor,
                                     offset: torch.Tensor,
                                     block_table: torch.Tensor,
                                     block_size: int,
                                     ratio: int) -> Tuple[torch.Tensor]:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::get_window_block_tables")
def get_window_block_tables_abstract(window_block_tables: torch.Tensor,
                                     window_context_lens: torch.Tensor,
                                     seq_k_lens: torch.Tensor,
                                     query_start_loc: torch.Tensor,
                                     block_table: torch.Tensor,
                                     block_size: int,
                                     window_size: int) -> Tuple[torch.Tensor]:
    return None

@torch._custom_ops.impl_abstract("torch_mlu_ops::single_layer_kv_transfer")
def single_layer_kv_transfer_abstract(
    lmc_key_value_cache: Tensor, vllm_key_value_cache: Tensor,
    slot_mapping: Tensor, direction: int, gpu_kv_format: int, token_major: bool
) -> None:
    return None
