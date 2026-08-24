import torch
import torch_mlu
from torch_mlu_ops import _TMOC
import inspect
from typing import Tuple, List, Dict, Optional, Union
from ._utils import (torchDtype2Str as _torchDtype2Str,
                     add_gen_case_decorator,
                     get_tmo_header_path,
                     get_tmo_library_path,
                    )

import numpy as np
from enum import IntEnum
from typing_extensions import deprecated
_cnnl_version = 0
_cnnl_extra_version = 0

__CLASSES__ = [
    "KVFormat",
    "TransferDirection",
    "PageBufferShapeDesc",
]

__FUNCTIONS__ = [
    "active",
    "apply_rotary",
    "apply_topkp",
    "apply_topkp_v2",
    "batch_matmul",
    "copy_blocks",
    "compress_kv",
    "concat_block_table",
    "dequant_from_linear_cache",
    "dequant_from_paged_cache",
    "dynamic_per_channel_quant",
    "flat_quant",
    "svd_quant",
    "flash_attention",
    "fused_compress_single_kv",
    "fused_indexer_k",
    "fused_layer_norm",
    "fused_mhc_post",
    "fused_mla_q",
    "fused_mla_kv",
    "fused_rms_norm",
    "fused_rope",
    "fused_transpose_layernorm",
    "get_cnnl_extra_version",
    "get_cnnl_version",
    "get_compress_block_tables",
    "get_tmo_header_path",
    "get_tmo_library_path",
    "get_window_block_tables",
    "group_gemm",
    "gather_split",
    "hshare",
    "index_selected_rotary_embedding",
    "masked_dot_select_sparse_paged_kv",
    "masked_indexer_select_paged_kv",
    "matmul",
    "moe_active",
    "moe_all2all_combine",
    "moe_all2all_create",
    "moe_all2all_destroy",
    "moe_all2all_dispatch",
    "moe_all2all_init",
    "moe_all2all_gen_gather_index",
    "moe_append_shared_expert",
    "moe_cast_gating",
    "moe_cast_gating_v2",
    "moe_combine_result",
    "moe_expand_input",
    "moe_gen_idx",
    "moe_all2all_gen_send_layout",
    "moe_quantize",
    "moe_softmax_topk",
    "moe_sigmoid_topk",
    "moe_softplus_topk",
    "moe_svd_quantize",
    "offline_quant_to_linear_cache",
    "offline_quant_to_paged_cache",
    "preload",
    "quant_conv2d",
    "quant_conv3d",
    "quant_to_linear_cache",
    "quant_to_paged_cache",
    "quant_mx_to_paged_cache",
    "quant_per_block",
    "quant_mx_qkv",
    "rejection_sample",
    "reshape_from_cache",
    "reshape_linear_cache",
    "reshape_paged_cache",
    "random_sample",
    "multi_layer_kv_transfer",
    "multi_layer_block_kv_transfer",
    "sage_attn",
    "scaled_matmul",
    "scaled_quantize",
    "single_layer_kv_transfer",
    "single_query_cached_kv_attn",
    "single_query_mixed_cached_kv_attn",
    "sliding_window_compress_attention",
    "smooth_quant_group_gemm",
    "ssparse_group_gemm",
    "ssparse_matmul",
    "swap_blocks",
    "transpose_all2all",
    "fused_mul_reduce_sum",
    "update_compressor_states",
    "update_out_and_lse",
    "fused_indexer_q",
    "fused_masked_mul_topk_select_paged_kv",
    "convert_vertical_slash_index",
    "hamming_score",
    "solve_tril",
    "fused_compress_multi_kv",
    "fused_mla_q_v2",
    "fused_rmsnorm_rope_store_paged_cache",
    "hc_split_sinkhorn"
]

__all__ = __CLASSES__ + __FUNCTIONS__

def get_cnnl_version():
    global _cnnl_version
    if _cnnl_version == 0:
        _place_holder = torch.tensor([], device="mlu")
        _cnnl_version = torch.ops.torch_mlu_ops.get_lib_version("cnnl", _place_holder)
    return _cnnl_version

def get_cnnl_extra_version():
    global _cnnl_extra_version
    if _cnnl_extra_version == 0:
        _place_holder = torch.tensor([], device="mlu")
        _cnnl_extra_version = torch.ops.torch_mlu_ops.get_lib_version("cnnl_extra", _place_holder)
    return _cnnl_extra_version

def fused_transpose_layernorm(x: torch.Tensor,
                              gamma: Optional[torch.Tensor],
                              beta: Optional[torch.Tensor],
                              eps: float,
                              out: torch.Tensor = None):
    """
    Apply transpose and layernorm to the input tensor.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm

    Math:
        x = x.permute(0, 2, 1, 3).contiguous()
        out = layernorm(x, gamma, beta, eps)

    Args:
        x (torch.Tensor): The input tensor to be normalized. Shape is (n1, c1, n2, c2).
        gamma (torch.Tensor): The weight of layernorm. Could be None, if not None, shape is (c1 * c2).
        beta (torch.Tensor): The bias of layernorm. Could be None, if not None, shape is (c1 * c2).
        eps (float): The eps of layernorm.
        out (torch.Tensor): The output tensor of normalizes. Shape is (n1, n2, c1 * c2).

    Type:
        x: half, bfloat16
        gamma: same as x
        beta: same as x
        output: same as x

    Return:
        Support inplace output.
        Return a tensor after applying transpose and layernorm.
    """
    if out is None:
        n1, c1, n2, c2 = x.shape
        out = torch.empty((n1, n2, c1 * c2), dtype=x.dtype, device=x.device)
    torch.ops.torch_mlu_ops.fused_layernorm(x, out, None, gamma, beta, None, None, None, None, None,
                                            "layernorm", eps, False, False, False, False, True)

    output = [out]
    return output[0] if len(output) == 1 else tuple(output)

def fused_layer_norm(x: torch.Tensor,
                     residual: Optional[torch.Tensor],
                     gamma: Optional[torch.Tensor],
                     beta: Optional[torch.Tensor],
                     bias: Optional[torch.Tensor],
                     eps: float,
                     store_output_before_norm: bool,
                     quant_scale: torch.Tensor = None,
                     out: torch.Tensor = None,
                     dynamic_quant: bool = False,
                     store_output_after_norm: bool = False,
                     quant_type: torch.dtype = torch.int8,
                     gamma_add_coef: float = 0.0,
                     scale_type: torch.dtype = torch.float,
                     mx_quant: bool = False):
    """
    Apply layernorm to the input tensor.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm

    Math:
       if gamma.dim() == 2:
            gamma = gamma + gamma_add_coef
            normed_out = layernorm(x) # layernorm.weight is 1, layernorm.bias is 0
            normed_out = normed_out * gamma.unsqueeze(1) + beta.unsqueeze(1)
            return normed_out
        else:
            if bias is not None:
                x1 = x + bias
            if residual is not None:
                x1 += residual
            normed = layernorm(x1, gamma, beta, eps)
            if mx_quant:
                if quant_scale:
                    normed = normed * quant_scale
                normed, mx_quant_scale = mx_quantize(normed)
            elif dynamic_quant:
                max, _ = (normed * quant_scale).abs().max(dim=-1, keepdim=True)
                smooth_quant_scale = max.to(torch.float) / int_max
                normed = (normed / smooth_quant_scale).round().to(int8)
            elif quant_scale:
                normed = (normed * quant_scale).round().clamp(-128, 127).to(torch.int8)
            if store_output_before_norm == True:
                output = (normed, x1)
            else:
                output = normed

    Args:
        x (torch.Tensor): The input tensor to be normalized. Shape is (p1, p2, ..., pn, C).
        residual (torch.Tensor): The tensor add to x. Shape is the same as x.
        gamma (torch.Tensor): The weight of layernorm. Could be None, if not None, shape is (C) or (N, C). If gamma shape is (N, C), x shape must be (N, T, C).
        beta (torch.Tensor): The bias of layernorm. Could be None, if not None, shape is (C) or (N, C). If beta shape is (N, C), x shape must be (N, T, C).
        bias (torch.Tensor): The bias of input tensor before layernorm. Shape is (C).
        eps (float): The eps of layernorm.
        store_output_before_norm (bool): If return the tensor before applying layernorm.
        quant_scale (torch.Tensor): The smooth_scale of output if dynamic_quant, else it is the scale of quantization. Shape is (C).
        out (torch.Tensor): The output tensor of normalizes. Shape is the same as x.
        dynamic_quant(bool): Flag quant_mode of output, dynamic quant or not.
        store_output_after_norm: If return the tensor before applying quantize.
        quant_type(torch.dtype): specific the data type of output if quantization. Support torch.int8 and torch.float8_e4m3fn.
        gamma_add_coef (float): A scalar added to gamma, only support when gamma is 2dim.
        mx_quant(bool): Flag quant_mode of output, mx quant or not.

    Type:
        x: float, half, bfloat16
        residual: same as x
        beta: same as x
        gamma: same as x
        bias: same as x
        quant_scale: float
        output: same as x, int8, float8, float8_e4m3fn, float4_e2m1fn_x2
        smooth_quant_scale: float, float8_e8m0fnu, bfloat16
        gamma_add_coef: float

    Return:
        Support inplace output.
        Return a tensor after applying layernorm and quantization.
        If store_output_before_norm is true, in addition return a tensor before applying layernorm.
        If dynamic_quant is true or mx_quant is true, in addition return smooth_quant scale of out tensor.
        If store_output_after_norm is true, in addition return a tensor before quantization.
    """
    input_shape = x.shape
    if gamma is not None and gamma.dim() == 2:
        if out is None:
            out = torch.empty(input_shape, dtype=x.dtype, device=x.device)
        assert x.dim() == 3
        assert residual is None
        assert bias is None
        assert store_output_before_norm is False
        assert quant_scale is None
        assert dynamic_quant is False
        assert store_output_after_norm is False
        torch.ops.torch_mlu_ops.layernorm_forward(x, out, gamma, beta, eps, gamma_add_coef)
        return out
    else:
        if out is None:
            if quant_scale is not None:
                out = torch.empty(input_shape, dtype=quant_type, device=x.device)
            else:
                if mx_quant:
                    out = torch.empty(input_shape, dtype=quant_type, device=x.device)
                else:
                    out = torch.empty(input_shape, dtype=x.dtype, device=x.device)
        residual_out = torch.empty(input_shape, dtype=x.dtype, device=x.device) if store_output_before_norm else None
        if mx_quant:
            block_size = 32
            smooth_quant_scale = torch.empty(x.shape[:-1] + (x.shape[-1] // block_size,), dtype=scale_type, device=x.device)
        else:
            smooth_quant_scale = torch.empty(x.shape[:-1], dtype=torch.float, device=x.device) if dynamic_quant else None
        normed_out = torch.empty(*x.shape, dtype=x.dtype, device=x.device) if store_output_after_norm else None
        torch.ops.torch_mlu_ops.fused_layernorm(x, out, residual, gamma, beta, bias, quant_scale, residual_out, smooth_quant_scale, normed_out,
                                                "layernorm", eps, store_output_before_norm, store_output_after_norm, dynamic_quant, mx_quant)
        output = [out]
        if store_output_before_norm:
            output.append(residual_out)
        if dynamic_quant or mx_quant:
            output.append(smooth_quant_scale)
        if store_output_after_norm:
            output.append(normed_out)
        return output[0] if len(output) == 1 else tuple(output)

def fused_rms_norm(x: torch.Tensor,
                   residual: Optional[torch.Tensor],
                   gamma: Optional[torch.Tensor],
                   beta: Optional[torch.Tensor],
                   bias: Optional[torch.Tensor],
                   eps: float,
                   store_output_before_norm: bool,
                   quant_scale: torch.Tensor = None,
                   out: torch.Tensor = None,
                   dynamic_quant: bool = False,
                   store_output_after_norm: bool = False,
                   quant_type: torch.dtype = torch.int8,
                   scale_type: torch.dtype = torch.float,
                   mx_quant: bool = False):
    """
    Apply rmsnorm to the input tensor.
    Parameter limitations the same as "fused_layer_norm", except rms_norm support quant_scale shape is [H, C].
    """
    input_shape = x.shape
    if out is None:
        if quant_scale is not None:
            out = torch.empty(input_shape, dtype=quant_type, device=x.device)
        else:
            if mx_quant:
                out = torch.empty(input_shape, dtype=quant_type, device=x.device)
            else:
                out = torch.empty(input_shape, dtype=x.dtype, device=x.device)
    residual_out = torch.empty(input_shape, dtype=x.dtype, device=x.device) if store_output_before_norm else None
    if dynamic_quant:
        assert quant_scale is not None, f"quant_scale can not be None when dynamic_quant"
        scale_dim = x.dim() - quant_scale.dim()
        smooth_quant_scale_shape = x.shape[:scale_dim]
    if mx_quant:
        block_size= 32
        smooth_quant_scale_shape = x.shape[:-1] + (x.shape[-1] // block_size,)
        smooth_quant_scale = torch.empty(smooth_quant_scale_shape, dtype=scale_type, device = x.device)
    else:
        smooth_quant_scale = torch.empty(smooth_quant_scale_shape, dtype=torch.float, device = x.device) if dynamic_quant else None
    normed_out = torch.empty(*x.shape, dtype=x.dtype, device=x.device) if store_output_after_norm else None
    torch.ops.torch_mlu_ops.fused_layernorm(x, out, residual, gamma, beta, bias, quant_scale, residual_out, smooth_quant_scale, normed_out, "rmsnorm", eps, store_output_before_norm, store_output_after_norm, dynamic_quant, mx_quant)
    output = [out]
    if store_output_before_norm:
        output.append(residual_out)
    if dynamic_quant or mx_quant:
        output.append(smooth_quant_scale)
    if store_output_after_norm:
        output.append(normed_out)
    return output[0] if len(output) == 1 else tuple(output)

def flat_quant(input: torch.Tensor,
               affine_weight_left: torch.Tensor,
               affine_weight_right: torch.Tensor,
               input_scale: Optional[torch.Tensor],
               clip_factor_max: Optional[torch.Tensor],
               clip_factor_min: Optional[torch.Tensor],
               quant_mode: int,
               asym_quant: bool,
               output: torch.Tensor = None,
               output_scale: torch.Tensor = None,
               output_calibration: torch.Tensor = None):
    """

    Math:
        init_shape = input.shape
        ntoken = torch.tensor(init_shape[:-1]).prod().item()
        input = input.reshape(ntoken, affine_weight_left.shape[0], -1)
        input = torch.matmul(affine_weight_left.to(input), input)
        input = input.reshape(-1, affine_weight_right.shape[0])
        input = torch.matmul(input, affine_weight_right.transpose(0, 1))
        input = input.reshape(init_shape)
        bits = 4
        q_max = torch.tensor(2**bits - 1)
        xmax = input.amax(1, keepdim=True)
        xmin = input.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax = torch.maximum(xmax, tmp),
        xmin = torch.minimum(xmin, tmp)
        if clip_factor_a_max and clip_factor_a_min:
            xmax = xmax * torch.nn.functional.sigmoid(clip_factor_a_max)
            xmin = xmin * torch.nn.functional.sigmoid(clip_factor_a_min)
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
        output_scale = (xmax - xmin) / q_max
        zero = xmin / scale
        output = torch.clamp(round_ste(x / output_scale + zero), -8, 7)
        calibration = torch.stack((zero.flatten(), torch.sum(output, axis=1)), dim=1)

    Args
        input (torch.Tensor): The tensor to be quantized. Shape is (token_num, hidden_size). The tensor must be continuous.
        affine_weight_left (torch.Tensor): The left rotate matrix. Shape is (k1, k1).
        affine_weight_right (torch.Tensor): The right rotate matrix. Shape is (k2, k2), it must be transposed.
        input_scale (torch.Tensor): The scale multipled to the input tensor. Shape is (hidden_size).
        clip_factor_max (torch.Tensor): The max clip tensor. Shape is (token_num), but now do not support and must set to null.
        clip_factor_min (torch.Tensor): The min clip tensor. Shape is (token_num), but now do not support and must set to null.
        token_num (int32_t): total number of tokens.
        hidden_size (int32_t): must be same as k1 * k2.
        k1 (int32_t): row and col of left rotate matrix.
        k2 (int32_t): row and col of right rotate matrix.
        quant_mode (int32_t): only can be set to 1, represented per token quantize.
        asym_quant (bool): only can be set to false, represented sym quant, where output and output_scale are needed.
        output (torch.Tensor): quantized from input. shape is (token_num, hidden_size / 2).
        output_scale (torch.Tensor): quantized scale of output.
        output_calibration (torch.Tensor): calibration matrix for asym_quant with shape of (token_num, 2), where it is None when asym_quant is False.

    Type:
        input: FP16, BF16
        affine_weight_left: FP16, BF16
        affine_weight_right: FP16, BF16
        input_scale: FP32
        clip_factor_max: FP32
        clip_factor_min: FP32
        token_num: INT32
        hidden_size: INT32
        k1: INT32
        k2: INT32
        quant_mode: INT32
        asym_quant: BOOL
        output: INT8(INT4x2)
        output_scale: FP32
        output_calibration: FP32

    Return:
        output, output_scale, output_calibration
    """

    if output is None:
        output = torch.empty((*input.shape[:-1], input.shape[-1]//2), dtype=torch.int8, device=input.device)
    if output_scale is None:
        output_scale = torch.empty(input.size(0), dtype=torch.float32, device=input.device)
    if output_calibration is None and asym_quant:
        output_calibration = torch.empty(input.size(0), 2, dtype=torch.float32, device=input.device)
    torch.ops.torch_mlu_ops.flat_quant(input, affine_weight_left, affine_weight_right, input_scale, clip_factor_max,
                                       clip_factor_min, quant_mode, asym_quant, output, output_scale, output_calibration, False)

    # return output, output_scale, output_calibration
    if asym_quant or output_calibration is not None:
        return output, output_scale, output_calibration
    else:
        return output, output_scale, None


def svd_quant(input: torch.Tensor,
              weight_lora_down: torch.Tensor,
              smooth: torch.Tensor,
              lora_scales: torch.Tensor,
              quant_mode: int,
              asym_quant: bool,
              quant_dtype: str,
              output_lora_down: torch.Tensor = None,
              output_quant: torch.Tensor = None,
              output_quant_scales: torch.Tensor = None):
    """
    Perform svd quant operation(ABSORBING OUTLIERS BY LOW-RANK
        COMPONENTS FOR 4-BIT DIFFUSION MODELS).
        For details,https://arxiv.org/abs/2411.05007.

    Math:
        inputs-parameters:
            input: shape[token_num, hidden_size]
            weight_lora_down: shape[lora_rank, hidden_size]
            smooth: shape[hidden_size]
            lora_scales: shape[lora_rank]
        outputs-parameters:
            output_lora_down: shape[token_num, lora_rank]
            output_quant: shape[token_num, hidden_size]
            output_quant_scales: shape[token_num]
        process:
            output_lora_down = torch.matmul(input, weight_lora_down.transpose(0, 1))
            output_lora_down = output_lora_down * lora_scale
            input_smooth = input * smooth
            input_smooth = input_smooth.to(torch.float32)
            max_values = torch.max(input_smooth, dim=1)
            min_values = torch.max(input_smooth, dim=1)
            min_values = min_values.abs()
            max_values = torch.max(max_values, min_values)
            scales = max_values / 7 for int4
            output_quant_scales = 1 / scales
            output_quant = input_smooth * output_quant_scales
            output_quant = torch.clamp(torch.round(output_quant), -8, 7) for int4

    Args:
        input (torch.Tensor): The input tensor. Shape is (token_num, hidden_size).
        weight_lora_down(torch.Tensor): The weight of lora-down projection matmul. Shape is (lora_rank, hidden_size).
        smooth(torch.Tensor): The input smooth factor for quantization process. Shape is (hidden_size).
        lora_scales(torch.Tensor): The factor of lora-down projection output. Shape is (lora_rank).
        quant_mode(int): 0 - The quant-mode of input quantization process.Current only support token quantization.
        asym_quant(bool): false - The flag indentify asymmetric quantization. Current only support symmetric quantization.
        quant_dtype(str): "int4" - The flag indentify data type of quantified input. Current only support "int4", for furture extension.
        output_lora_down(torch.Tensor):The lora-down projection output. Shape is (token_num, lora_rank).
        output_quant(torch.Tensor):The quantified input. Shape (token_num, hidden_size // 2). For data type is int4x2, one packed two int4.
        output_quant_scales(torch.Tensor): The Reciprocal of quantization process. Shape is (token_num). See Math segment for detail.

    Type:
        input: FP16，BF16
        weight_lora_down: FP16，BF16
        smooth: FP16，BF16
        lora_scales: FP32
        quant_mode:: int
        asym_quant: bool
        quant_dtype: str, 'int4' or 'fp4', current only support 'int4'

    Return:
        output_lora_down: FP16, BF16
        output_quant: int8, one packed two int4 or fp4, shape (token_num, hidden_size // 2).
        output_quant_scales: FP32
    """

    if output_lora_down is None:
        output_lora_down = torch.empty((input.size(0),
                weight_lora_down.size(0)), dtype=input.dtype,
                device=input.device)
    if output_quant is None:
        output_quant = torch.empty((input.size(0),
                input.size(1) // 2), dtype=torch.int8,
                device=input.device)
    if output_quant_scales is None:
        # for per-token quantization
        if quant_mode == 0:
            output_quant_scales = torch.empty(input.size(0),
                        dtype=torch.float32, device=input.device)
    torch.ops.torch_mlu_ops.svd_quant(input, weight_lora_down,
            smooth, lora_scales,
            output_lora_down, output_quant, output_quant_scales,
            quant_mode, asym_quant, quant_dtype,
            "none", 0, 0,
            None, None, None)

    return output_lora_down, output_quant, output_quant_scales


def flash_attention(q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    out: Optional[torch.Tensor],
                    cu_seq_lens_q: Optional[torch.Tensor],
                    cu_seq_lens_kv: Optional[torch.Tensor],
                    alibi_slope: Optional[torch.Tensor],
                    attn_bias: Optional[torch.Tensor],
                    max_seq_len_q: int,
                    max_seq_len_kv: int,
                    softmax_scale: float,
                    is_causal: bool,
                    window_size_left: int = -1,
                    window_size_right: int = -1,
                    compute_dtype: torch.dtype = torch.float,
                    return_lse: bool = False,
                    block_tables: Optional[torch.Tensor] = None,
                    k_quant_scale: Optional[torch.Tensor] = None,
                    v_quant_scale: Optional[torch.Tensor] = None,
                    q_quant_scale: Optional[torch.Tensor] = None,
                    out_quant_scale: Optional[torch.Tensor] = None,
                    out_dtype: torch.dtype = torch.half,
                    q2k_block_idx: Optional[torch.Tensor] = None,
                    q2k_block_num: Optional[torch.Tensor] = None,
                    variable_block_sizes: Optional[torch.Tensor] = None,
                    q_block_size: int = 256,
                    k_block_size: int = 128,
                    sink: Optional[torch.Tensor] = None,
                    ):
    """
    Apply attention operation on q, k and v.

    Math:
        qk = bmm(q, k, q_scale, k_scale) * softmax_scale
        if alibi_slope is not None:
            qk += create_alibi(alibi_slope)
        if attn_bias is not None:
            qk += attn_bias
        if is_causal:
            qk = tril_mask(qk)
        if return_lse:
            lse = logsumexp(qk, dim=-1)
        p = softmax(qk, dim=-1)
        scale_attn = None
        if quant_v:
            p, p_scale = quant(p)
        attn_out = bmm(p, v, p_scale, v_scale)
        output = (attn_out, lse) if return_lse else attn_out

    Args:
        q (torch.Tensor): The query tensor. Shape is (batch, seq_q, head_num_q, head_size_qk) or (total_seq_q, head_num_q, head_size_qk).
        k (torch.Tensor): The key tensor. If block_tables is None, the shape of k is (batch, seq_kv, head_num_kv, head_size_qk) or (total_seq_kv, head_num_kv, head_size_qk).
                                          If block_tables is not None, if use paged attention, shape is (total_blocks, head_num_kv, block_size, head_size_qk),
                                            where the total_blocks = max_batch * (memory_cache_len / block_size). If not use paged attention, shape is (max_batch, head_num_kv, memory_cache_len, head_size_qk).
        v (torch.Tensor): The value tensor. If v is per_groupwise_quant shape is [head_num_k, head_size_v, batch * seq_k_pad].
                                            Else shape is the same as k, except, head_size_v could be different with head_size_qk.
        out (torch.Tensor): The output tensor. Shape is (batch, seq_q, head_num_q, head_size_v) or (total_seq_q, head_num_q, head_size_v).
        cu_seq_lens_q (torch.Tensor): The cusum of seq_q if q is 3-D tensor. Shape is (batch+1).
        cu_seq_lens_kv (torch.Tensor): The cusum of seq_kv if k/v is 3-D tensor. Shape is (batch+1).
        alibi_slope (torch.Tensor): The alibi_slope to generate alibi position embedding. Shape is (head_num_q).
        attn_bias (torch.Tensor): The bias added to qk. Shape is (batch, head_num_q, max_seq_len_q, max_seq_len_kv).
        max_seq_len_q (int): The maximum value of seq_q. Useless when pad-mode, could set -1.
        max_seq_len_kv (int): The maximum value of seq_kv. Useless when pad-mode, could set -1.
        softmax_scale (float): The scale factor multiplied to qk.
        is_causal (bool): Controlling if use causal mask.
        window_size_left (int): left window size, set -1 if it is unlimited. The maximum lengths of key and value involved in the attention calculation is windows_size_left + 1.
        window_size_right (int): right window size, set -1 if it is unlimited. The maximum lengths of key and value involved in the attention calculation is windows_size_right + 1
        compute_dtype (torch.dtype): The dtype used to calculate softmax. Support torch.half, torch.float and torch.bfloat16, but use float actually when set to torch.bfloat16.
                                     When all q/k/v's dtype are int8, use half as actual compute_dtype no matter which dtype is in input.
        return_lse (bool): Controlling if return lse tensor.
        k_quant_scale (torch.Tensor): The quantized scale of k, if k is per_tensor quant, shape is [1].
                                      If k is per_block quant, shape is [batch, head_num_k, max_block_num_k].
                                      If k is MX format, shape is [head_num_k, total_k, qk_sacle_block_num] or [head_num_k, batch, seq_k, qk_sacle_block_num]
        v_quant_scale (torch.Tensor): The quantized scale of v, if v is per_tensor quant, shape is [1].
                                      If v is per_channel quant, shape is [batch, head_num_k, head_size_v].
                                      If v is MX format, shape is (head_num_k, head_size_v, total_v_scale_num).
        q_quant_scale (torch.Tensor): The quantized scale of q, if q is per_tensor quant, shape is [1].
                                      If q is per_block quant, shape is [batch, head_num_q, max_block_num_q].
                                      If q is MX format, shape is [head_num_q, total_q, qk_sacle_block_num] or [head_num_q, batch, seq_k, qk_sacle_block_num].
        out_quant_scale: Reserved parameter, the quantized scale of out, must be None.
        block_tables (torch.Tensor): The tensor recording the k/v_cache's positions of each batch.
            If use paged attention, shape is (batch, max_block_num), where max_block_num = memory_cache_len / block_size, block_size only support 16.
            If not use paged attention, shape is (batch, 1).
        out_dtype: dtype of output when v's dtype is float8_e4m3fn, int8 or mx-int8, out_dtype must be half or bfloat16.
                   Else out_dtype must be same as v.
        q2k_block_idx (torch.Tensor): The block indices of keys and values to be computed. Shape is [batch, head_num_q, max_block_per_q, max_block_per_k].
        q2k_block_num (torch.Tensor): The count of block for keys and values to be computed, must not be zero. Shape is [batch, head_num_q, max_block_per_q].
        variable_block_sizes (torch.Tensor): The valid sequence lengths within each k_block, must not be zero. Shape is [max_block_per_k].
                                             If None, then all sequence lengths within the k_blocks are valid.
        q_block_size (int): The block_sparsity size of query.
        k_block_size (int): The block_sparsity size of key and value.
        sink(torch.Tensor): sink token. Shape is (head_num_q).

    Type:
        q: float, half, bfloat16, float8_e4m3fn, int8, mx-int8.
        k: same as q.
        v: float, half, bfloat16, float8_e4m3fn, int8, mx-int8.
        cu_seq_lens_q: int32.
        cu_seq_lens_kv: int32.
        alibi_slope: float.
        attn_bias: must be float when q.dtype is float8_e4m3fn, else same as q.
        out : must be bfloat16 or half when v.dtype is float8_e4m3fn, int8, mx-int8, else same as v.
        q_quant_scale: float, float8_e8m0, bfloat16.
        k_quant_scale: float, float8_e8m0, bfloat16.
        v_quant_scale: float, float8_e8m0, bfloat16.
        lse: float.
        q2k_block_idx: int32
        q2k_block_num: int32
        variable_block_sizes: int32
        sink: float

    Return:
        If return_lse is True, return a tuple of (attn_out, lse), else return attn_out.
    """
    if out is None:
        output = torch.ops.torch_mlu_ops.aot_flash_attention(q, k, v,
                                            cu_seq_lens_q, cu_seq_lens_kv, alibi_slope, attn_bias,
                                            q_quant_scale, k_quant_scale, v_quant_scale, out_quant_scale,
                                            block_tables, max_seq_len_q, max_seq_len_kv, softmax_scale, is_causal,
                                            window_size_left, window_size_right, _torchDtype2Str(compute_dtype),
                                            return_lse, q2k_block_idx, q2k_block_num, variable_block_sizes,
                                            q_block_size, k_block_size, _torchDtype2Str(out_dtype), sink)
        return (output[0], output[1]) if return_lse else output[0]
    else:
        out_lse = None
        if return_lse:
            lse_shape = q.shape[:-3] + (q.shape[-2],) + (q.shape[-3],)
            out_lse = torch.empty(lse_shape, dtype=torch.float, device=q.device) #[h, total_q] for pack, [b, h, t] for pad
        torch.ops.torch_mlu_ops.flash_attention(q, k, v, out, out_lse,
                                                cu_seq_lens_q, cu_seq_lens_kv, alibi_slope, attn_bias,
                                                q_quant_scale, k_quant_scale, v_quant_scale, out_quant_scale,
                                                block_tables, max_seq_len_q, max_seq_len_kv, softmax_scale, is_causal,
                                                window_size_left, window_size_right, _torchDtype2Str(compute_dtype),
                                                return_lse, q2k_block_idx, q2k_block_num, variable_block_sizes, q_block_size, k_block_size, sink)
        return (out, out_lse) if return_lse else out

def sage_attn(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              cu_seq_lens_q: Optional[torch.Tensor],
              cu_seq_lens_kv: Optional[torch.Tensor],
              max_seq_len_q: int,
              max_seq_len_kv: int,
              softmax_scale: float,
              is_causal: bool,
              compute_dtype: torch.dtype = torch.float,
              return_lse: bool = False,
              smooth_k: bool = True,
              quantize_v: bool = False,
              quant_dtype: torch.dtype = torch.int8
            ):
    """
    Apply SageAttention operation on q, k and v.

    Math:
        quant_q, q_scale, quant_k, k_scale = quant_per_block(q, k)
        qk = bmm(quant_q, quant_k, q_scale, k_scale) * softmax_scale
        if is_causal:
            qk = tril_mask(qk)
        if return_lse:
            lse = logsumexp(qk, dim=-1)
        attn = softmax(qk, dim=-1)
        attn_out = bmm(attn, v)
        output = (attn_out, lse) if return_lse else attn_out

    Args:
        q (torch.Tensor): The query tensor. Shape is (total_seq_q, head_num_q, head_size_qk) or (batch, max_seq_q, head_num_q, head_size_qk).
        k (torch.Tensor): The key tensor. The shape of k is (total_seq_kv, head_num_kv, head_size_qk) or or (batch, max_seq_kv, head_num_kv, head_size_qk)..
        v (torch.Tensor): The value tensor. Shape is the same as k, except, head_size_v could be different with head_size_qk.
        cu_seq_lens_q (torch.Tensor): The cusum of seq_q when q is packed. Shape is (batch+1).
        cu_seq_lens_kv (torch.Tensor): The cusum of seq_kv when k/v are packed. Shape is (batch+1).
        max_seq_len_q (int): The maximum value of seq_q. Useless when pad-mode, could set -1.
        max_seq_len_kv (int): The maximum value of seq_kv. Useless when pad-mode, could set -1.
        softmax_scale (float): The scale factor multiplied to qk.
        is_causal (bool): Controlling if use causal mask.
        compute_dtype (torch.dtype): The dtype used to calculate softmax, support torch.half, torch.float and torch.bfloat16, but use float actually when set to torch.bfloat16.
                                     When all q/k/v are quanted to int8, use half as actual compute_dtype no matter which dtype is in input.
        return_lse (bool): Controlling if return lse tensor.
        smooth_k (bool): If subtracting the mean data before doing quantization for key.
        quantize_v (bool): Quantize v by per_channel, only support when quant_dtype is torch.int8.
        quant_dtype (torch.dtype): Support torch.int8 and torch.float8_e4m3fn.

    Type:
        q: float, half, bfloat16
        k: same as q.
        v: same as q.
        cu_seq_lens_q: int32.
        cu_seq_lens_kv: int32.
        attn_out: the same as v.
        lse: float.

    Return:
        If return_lse is True, return a tuple of (attn_out, lse), else return attn_out.
    """

    # quant a/k
    assert q.dim() == 3 or q.dim() == 4, "dim of q must be equal to 3 or 4"
    is_packed = q.dim() == 3
    if is_packed: # q/k packed or not at the same time
        assert cu_seq_lens_q is not None and cu_seq_lens_q.dim() == 1, "packed: dim of cu_seq_lens_q must be equal to 1"
        assert cu_seq_lens_kv is not None and cu_seq_lens_kv.dim() == 1, "packed: dim of cu_seq_lens_kv must be equal to 1"
    block_size = 64
    seq_q = max_seq_len_q if is_packed else q.shape[1]
    seq_kv = max_seq_len_kv if is_packed else k.shape[1]
    batch = cu_seq_lens_q.size(0) - 1 if is_packed else q.shape[0]
    head_num_q = q.size(-2)
    max_block_num_q = (seq_q + block_size - 1) // block_size
    head_num_k = k.size(-2)
    head_size_v = v.size(-1)
    max_block_num_k = (seq_kv + block_size - 1) // block_size
    quant_q = torch.empty(*q.shape, dtype=quant_dtype, device=q.device)
    q_scale = torch.empty(batch, head_num_q, max_block_num_q, 1, dtype=torch.float, device=q.device)
    quant_k = torch.empty(*k.shape, dtype=quant_dtype, device=k.device)
    k_scale = torch.empty(batch, head_num_k, max_block_num_k, 1, dtype=torch.float, device=k.device)
    quant_v, v_scale = v, None

    if quantize_v:
        quant_v = torch.empty(*v.shape, dtype=quant_dtype, device=q.device)
        v_scale = torch.empty(batch, head_num_k, head_size_v, dtype=torch.float, device=q.device)
        quant_per_block(q, k,
                        cu_seq_lens_q if is_packed else None,
                        cu_seq_lens_kv if is_packed else None,
                        seq_q, seq_kv,
                        block_size, block_size,
                        smooth_k, quant_q, q_scale, quant_k, k_scale,
                        quant_dtype, v, quant_v, v_scale,
                        cu_seq_lens_kv if is_packed else None,
                        seq_kv)
    else:
        quant_per_block(q, k,
                        cu_seq_lens_q if is_packed else None,
                        cu_seq_lens_kv if is_packed else None,
                        seq_q, seq_kv,
                        block_size, block_size,
                        smooth_k, quant_q, q_scale, quant_k, k_scale,
                        quant_dtype)
    # flash_attn
    tmo_out = torch.empty(q.size()[:-1] + (v.size()[-1],), dtype=v.dtype, device=q.device)
    return flash_attention(quant_q, quant_k, quant_v, tmo_out,
                            cu_seq_lens_q, cu_seq_lens_kv, None, None,
                            max_seq_len_q, max_seq_len_kv,
                            softmax_scale, is_causal, -1, -1, compute_dtype,
                            return_lse, None,
                            k_scale.view(batch, head_num_k, max_block_num_k),
                            v_scale,
                            q_scale.view(batch, head_num_q, max_block_num_q),
                            q_block_size=-1, k_block_size=-1)

def single_query_cached_kv_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    out: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    k_cache_quant_scale: Optional[torch.Tensor],
    v_cache_quant_scale: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_context_len: int,
    windows_size_left: int,
    windows_size_right: int,
    softmax_scale: float,
    return_lse: bool = False,
    kv_cache_quant_bit_size = -1,
    q_quant_scale: Optional[torch.Tensor] = None,
    out_quant_scale: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.half,
    compute_dtype: torch.dtype = torch.float,
    head_size_v: int = -1,
    mask: Optional[torch.Tensor] = None,
    cu_seq_q: Optional[torch.Tensor] = None,
    max_seq_q: int = -1,
    sink: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply attention operation on query and cached key/value.

    Math:
        out_list = []
        lse_list = []
        for i in range(q.shape[0]):
            key = collect(k_cache, block_tables[i], context_lens[i])
            value = collect(v_cache, block_tables[i], context_lens[i])
            qk = bmm(q[i], key) * softmax_scale
            if alibi_slope is not None:
                alpha = qk + create_alibi(alibi_slope)
            attn = softmax(alpha, dim=-1)
            attn_out = bmm(attn, value)
            lse = logsumexp(qk, dim = -1)
            out_list.append(attn_out)
            lse_list.append(lse)
        output = concat(out_list, dim = 0)
        output_lse = concat(lse_list, dim = 0)

    Args:
        q (torch.Tensor): The query tensor, shape is (batch, seq_q, head_num_q, head_size_qk) or (total_q, head_num_q, head_size_qk).
        k_cache (torch.Tensor): The key cache tensor. If use paged attention, shape is (total_blocks, head_num_kv, block_size, cache_dim),
            where the total_blocks = max_batch * (memory_cache_len / block_size), block_size could be [1, 2, 4, 8, 16, 32, 64], and block_size = max_context_len when non-pages mode.
            If not use paged attention, shape is (total_blocks, head_num_kv, block_size, cache_dim) where total_blocks = max_batch, block_size=memory_cache_len.
            If kv_cache_quant_bit_size=-1 or 8, cache_dim = head_size_qk, if v_cache is None, cache_dim = max(head_size_qk, head_size_v).
            else cache_dim=head_size_qk//2 when kv_cache_quant_bit_size=4, cache_dim=head_size_qk * 3//4 when kv_cache_quant_bit_size=6.
        v_cache (torch.Tensor): The value cache tensor. shape is the same as k_cache, except cache_dim could be different.
            If v_cache is None, v_cache is included by k_cache. key cache = k_cache[..., :head_size_qk], value cache = v_cache[..., :head_size_v].
        out (torch.Tensor): The output tensor, shape is the same as q
        block_tables (torch.Tensor): The tensor recording the k/v_cache's positions of each batch.
            If use sparse_kv, shape is (batch, head_num_kv, max_block_num) or (batch, head_num_q, max_block_num), else shape is (batch, max_block_num).
            max_block_num = max_cache_len / block_size if not use paged attention else max_block_num = 1.
        context_lens (torch.Tensor): The tensor recording the context lengths of each batch, shape is (batch).
        k_cache_quant_scale (torch.Tensor): The quantized scale of k_cache.
                            When quantize mode is per_token, shape is (total_blocks, head_num_kv, block_size) or (total_blocks, head_num_kv, block_size, 1), support stride in high 2-dim.
                            when quantize mode is per_channel, shape is (head_num_kv, head_size), must be contiguous.
                            when quantize mode is per_tensor, shape is (1).
                            when quantize mode is per_groupwise, shape is (total_blocks, head_num_kv, block_size, head_size//32), must be contiguous.
        v_cache_quant_scale (torch.Tensor): The quantized scale of v_cache, when quantize mode is per_groupwise shape is (total_blocks, head_num_kv, block_size//32, head_size_v),
                            else shape is the same as k_cache_quant_scale.
        alibi_slopes (torch.Tensor): The alibi_slope to generate alibi position embedding. Shape is (batch, head_num_q).
        max_context_len (int): The maximum value of context_lens.
                              If max_context_len <= 0, indicating an uncertain maximum value of context_lens.
                              If 0 < max_context_len < max(context_lens), the correctness is not guaranteed.
        windows_size_left (int): left window size, set -1 if it is unlimited. The maximum lengths of key and value involved in the attention calculation is windows_size_left + 1.
        windows_size_right (int): Reserved parameter, set -1.
        softmax_scale (float): The scale factor multiplied to qk.
        return_lse (bool): Controlling if return lse tensor.
        kv_cache_quant_bit_size (int): The data format of kv_cache, 4 for int4, 8 for int8, and -1 for the same as query.
        q_quant_scale (torch.Tensor): The quantized scale of q. If quant_mode is per_tensor shape is [1].
                                       If q's quant_mode is per_token shape is [total_q, head_num] or [batch, seq_q, head_num_q].
                                       If q's quant_mode is per_groupwise, shape is (batch, seq_q, head_num, scale_num).
        out_quant_scale(torch.Tensor): Reserved parameter, The quantized scale of out, must be None.
        out_dtype(torch.dtype): Dtype of output, must be torch.half when q's dtype is torch.float8_e4m3fn.
        compute_dtype(torch.dtype): The dtype used to calculate softmax, support torch.float16, torch.float32 and torch.bfloat16. Bf16 only support when qkv's dtype are fp8.
        head_size_v(int): The head_size of value_cache when both v_cache and out are None.
        mask(torch.Tensor): Mask of attention, shape is [batch, seq_q, mask_len], mask_len % seq_q=0. Only support when seq_q<=16 and mask_len<=64, and mask_len%seq_len=0.
                            Dtype is the same as compute_dtype, mask is not support when q/k/v/out's dtype is torch.float.
        cu_seq_q(torch.Tensor): The cusum of seq_q if q is 3-D tensor. Shape is (batch+1).
        max_seq_q(int): The maximum value of seq_q. If max_seq_q <= 0, indicating an uncertain maximum value of context_lens, useless if pad-mode,
        sink(torch.Tensor): sink token. Shape is (head_num_q).

    Type:
        q: float, half, bfloat16, float8_e4m3fn, int8, mx-int8.
        k_cache: the same as q or int8, int8(pairwise-int4), float4_e2m1fn_x2, float8_e4m3fn, float8_e4m3fn(3 substitutions for 4 fp6)
        v_cache: same as k_cache.
        block_tables: int32.
        context_lens: int32.
        k_cache_quant_scale: float, bfloat16.
        v_cache_quant_scale: float, bfloat16.
        alibi_slopes: float.
        q_quant_scale: float, float8_e8m0, bfloat16.
        out: float, half, bfloat16.
        output_lse: float.
        mask: float, half, bfloat16.
        sink: float.

    Return:
        If return_lse is True, return a tuple of (output, lse), else return output.
    """
    tmo_out = out
    if out is None:
        if v_cache is not None:
            head_size_v = v_cache.shape[-1] * 4 // 3  if kv_cache_quant_bit_size == 6 else v_cache.shape[-1]
        if q.dtype not in {torch.float8_e4m3fn, torch.int8}:
            out_dtype = q.dtype
        tmo_out = torch.empty(q.size()[:-1] + (head_size_v,), dtype=out_dtype, device=q.device)
    out_lse = None
    if return_lse:
        lse_shape = q.shape[:-3] + (q.shape[-2],) + (q.shape[-3],)
        out_lse = torch.empty(lse_shape, dtype=torch.float, device=q.device)
    torch.ops.torch_mlu_ops.single_query_cached_kv_attn(
            q, k_cache, tmo_out, block_tables, context_lens, v_cache, out_lse,
            q_quant_scale, k_cache_quant_scale, v_cache_quant_scale, out_quant_scale,
            alibi_slopes, mask, _torchDtype2Str(compute_dtype), max_context_len, windows_size_left,
            windows_size_right, softmax_scale, return_lse, kv_cache_quant_bit_size, cu_seq_q, max_seq_q, sink)
    return (tmo_out, out_lse) if return_lse else tmo_out


def single_query_mixed_cached_kv_attn(
    q: torch.Tensor,
    k_cache_lp: torch.Tensor,
    v_cache_lp: torch.Tensor,
    k_cache_hp: torch.Tensor,
    v_cache_hp: torch.Tensor,
    out: torch.Tensor,
    block_tables_lp: torch.Tensor,
    block_tables_hp: torch.Tensor,
    context_lens_lp: torch.Tensor,
    context_lens_hp: torch.Tensor,
    k_cache_quant_scale_lp: Optional[torch.Tensor],
    v_cache_quant_scale_lp: Optional[torch.Tensor],
    k_cache_quant_scale_hp: Optional[torch.Tensor],
    v_cache_quant_scale_hp: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    max_contxt_len_lp: int,
    max_contxt_len_hp: int,
    softmax_scale: float,
    return_lse: bool = False,
    kv_cache_quant_bit_size_lp = -1,
    kv_cache_quant_bit_size_hp = -1,
) -> Union[Tuple[torch.Tensor], torch.Tensor]:
    """
    Apply attention operation on query and cached key/value.

    Math:
        lse1, attn_out1 = single_query_cached_kv_attn(query, key_cache_lp, value_cache_lp, ...)
        lse2, attn_out2 = single_query_cached_kv_attn(query, key_cache_hp, value_cache_hp, ...)
        output = update_out_and_lse(lse1, attn_out1, lse2, attn_out2)

    Args:
        q (torch.Tensor): The query tensor, shape is (batch, seq_q, head_num_q, head_size). seq_q must be 1 now.
        k_cache_lp/k_cache_hp (torch.Tensor): The key cache tensor.
            If use paged attention, shape is (total_blocks, head_num_kv, block_size, head_size), where the total_blocks = max_batch * (memory_cache_len / block_size).
            If not use paged attention, shape is (max_batch, head_num_kv, memory_cache_len, head_size).
        v_cache_lp/vcache_hp (torch.Tensor): The value cache tensor, shape is the same as k_cache, just, head_size_v could be different with head_size_qk.
        out (torch.Tensor): The output tensor, shape is the same as q
        block_tables_lp/block_tables_hp (torch.Tensor): The tensor recording the k/v_cache's positions of each batch.
            If use paged attention, shape is (batch, max_block_num), where max_block_num = memory_cache_len / block_size.
            If not use paged attention, shape is (batch, 1).
        context_lens_lp/context_lens_hp (torch.Tensor): The tensor recording the context lengths of each batch, shape is (batch).
        k_cache_quant_scale_lp/k_cache_quant_scale_hp (torch.Tensor): The quantized scale of k_cache.
                            When quantize mode is per_token, shape is
                            paged:(total_blocks, head_num_kv, block_size, 1), linear: (max_batch, head_num_kv, memory_cache_len, 1).
                            when quantize mode is per_channel, shape is (head_num_kv, head_size)
        v_cache_quant_scale_lp/v_cache_quant_scale_hp (torch.Tensor): The quantized scale of v_cache, shape is the same as k_cache_quant_scale.
        alibi_slopes (torch.Tensor): The alibi_slope to generate alibi position embedding. Shape is (batch, head_num_q).
        max_contxt_len_lp/max_contxt_len_hp (int): The maximum value of context_lens.
                              If max_contxt_len <= 0, indicating an uncertain maximum value of context_lens.
                              If 0 < max_contxt_len < max(context_lens), the correctness is not guaranteed.
        softmax_scale (float): The scale factor multiplied to qk.
        return_lse (bool): Controlling if return lse tensor.
        kv_cache_quant_bit_size_lp/kv_cache_quant_bit_size_hp (int): The data format of kv_cache, -1 for float-point.

    Type:
        q: float, half, bfloat16.
        k_cache_lp/k_cache_hp: float, half, bfloat16, int8, int8(pairwise-int4).
        v_cache_lp/v_cache_hp: same as k_cache.
        block_tables_lp/block_tables_hp: int32.
        context_lens: int32.
        k_cache_quant_scale_lp/k_cache_quant_scale_hp: float.
        v_cache_quant_scale_lp/v_cache_quant_scale_hp: float.
        alibi_slopes: float.

    Return:
        If return_lse is True, return a tuple of (output, lse), else return output.
    """

    batch = q.shape[0]
    seq_q = q.shape[1]
    head_num = q.shape[2]
    head_size_v = v_cache_lp.shape[-1]
    tmo_out = torch.empty((q.shape[0], q.shape[1], q.shape[2], head_size_v), dtype=q.dtype, device=q.device) if out is None else out
    out_hp = torch.empty((q.shape[0], q.shape[1], q.shape[2], head_size_v), dtype=q.dtype, device=q.device)
    out_lse = torch.empty((batch, head_num, seq_q), dtype=torch.float32, device=q.device)
    out_lse_hp = torch.empty((batch, head_num, seq_q), dtype=torch.float32, device=q.device)

    torch.ops.torch_mlu_ops.single_query_cached_kv_attn(
            q, k_cache_lp, tmo_out, block_tables_lp, context_lens_lp, v_cache_lp, out_lse,
            None, k_cache_quant_scale_lp, v_cache_quant_scale_lp, None, alibi_slopes, None, 'float',
            max_contxt_len_lp, -1, -1, softmax_scale, True, kv_cache_quant_bit_size_lp, None, -1, None)
    torch.ops.torch_mlu_ops.single_query_cached_kv_attn(
            q, k_cache_hp, out_hp, block_tables_hp, context_lens_hp, v_cache_hp, out_lse_hp,
            None, k_cache_quant_scale_hp, v_cache_quant_scale_hp, None, alibi_slopes, None, 'float',
            max_contxt_len_hp, -1, -1, softmax_scale, True, kv_cache_quant_bit_size_hp, None, -1, None)
    torch.ops.torch_mlu_ops.update_out_and_lse(tmo_out, out_lse, out_hp, out_lse_hp, None, None, None)
    return (tmo_out, out_lse) if return_lse else tmo_out

def sliding_window_compress_attention(q: torch.Tensor,
    cache_swa: torch.Tensor,
    cache_comp: torch.Tensor,
    out: torch.Tensor,
    block_tables_swa: torch.Tensor,
    block_tables_comp: torch.Tensor,
    cu_seq_q: Optional[torch.Tensor],
    context_lens_swa: torch.Tensor,
    context_lens_comp: torch.Tensor,
    q_quant_scale: Optional[torch.Tensor],
    cache_quant_scale_swa: Optional[torch.Tensor],
    cache_quant_scale_comp: Optional[torch.Tensor],
    out_quant_scale: Optional[torch.Tensor],
    sink: Optional[torch.Tensor],
    softmax_scale: float,
    max_contxt_len_swa: int,
    max_contxt_len_comp: int,
    max_seq_q: int = -1,
    head_size_out: int=-1,
    return_lse: bool = False,
    kv_cache_quant_bit_size = -1,
    out_dtype: torch.dtype = torch.half,
    compute_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Apply sliding window attention and compressed attention, then merge results.

    This operator combines two attention paths:
      - SWA (sliding window attention) on cache_swa with optional sink
      - Compressed attention on cache_comp without sink
    Both paths use KV overlap mode (v_cache=None, cache_dim = max(head_size_qk, head_size_v)).
    Results are merged via update_out_and_lse.

    Math:
        # SWA attention path (with sink)
        out_swa, lse_swa = single_query_cached_kv_attn(q, cache_swa, None, ..., sink)
        # Compressed attention path (without sink)
        out_comp, lse_comp = single_query_cached_kv_attn(q, cache_comp, None, ..., None)
        # Merge via log-sum-exp
        out, lse = update_out_and_lse(out_swa, lse_swa, out_comp, lse_comp)

    Args:
        q (torch.Tensor): The query tensor, shape is (batch, seq_q, head_num_q, head_size_qk).
        cache_swa (torch.Tensor): The key cache for SWA path. Shape is
            (total_blocks_swa, head_num_kv, block_size, cache_dim), where cache_dim = max(head_size_qk, head_size_v).
        cache_comp (torch.Tensor): The key cache for compressed attention path. Shape is
            (total_blocks_comp, head_num_kv, block_size, cache_dim).
        out (torch.Tensor): The output tensor, shape is (batch, seq_q, head_num_q, head_size_v).
            Can be None.
        block_tables_swa (torch.Tensor): Block table for SWA cache, shape is (batch, max_block_num_swa).
        block_tables_comp (torch.Tensor): Block table for compressed cache, shape is (batch, max_block_num_comp).
        cu_seq_q (torch.Tensor): Reserved parameter, must be None.
        context_lens_swa (torch.Tensor): Context lengths for SWA path, shape is (batch).
        context_lens_comp (torch.Tensor): Context lengths for compressed path, shape is (batch).
        q_quant_scale (torch.Tensor): The quantized scale of q. If quant_mode is per_tensor shape is (1).
            If per_token, shape is (batch, seq_q, head_num_q).
        cache_quant_scale_swa (torch.Tensor): The quantized scale of SWA cache.
            When per_token, shape is (total_blocks_swa, head_num_kv, block_size).
            When per_channel, shape is (head_num_kv, head_size).
            When per_tensor, shape is (1).
        cache_quant_scale_comp (torch.Tensor): The quantized scale of compressed cache, same shape rules as cache_quant_scale_swa.
        out_quant_scale (torch.Tensor): Reserved parameter, must be None.
        sink (torch.Tensor): The sink token attention bias for SWA path, shape is (head_num_q). None to disable sink.
        softmax_scale (float): The scale factor multiplied to qk, typically 1 / sqrt(head_size_qk).
        max_contxt_len_swa (int): The maximum value of context_lens_swa.
        max_contxt_len_comp (int): The maximum value of context_lens_comp.
        max_seq_q (int): The maximum query sequence length. Default -1 means equal to seq_q.
        head_size_out (int): The output head size (head_size_v). Default -1 means equal to head_size_qk.
        return_lse (bool): If True, return (output, lse). Default False.
        kv_cache_quant_bit_size (int): The data format of kv_cache. -1 for same as query, 8 for int8/float8_e4m3fn, 4 for int4/float4_e2m1fn_x2, 6 for float6_e2m3. Default -1.
        out_dtype (torch.dtype): Dtype of output, only work when q's dtype in {int8, float8}. Default torch.half.
        compute_dtype (torch.dtype): Dtype for computation, support {torch.half, torch.bfloat16, torch.float}. Default torch.float.

    Type:
        q: float, half, bfloat16.
        cache_swa/cache_comp: the same as q or int8, int8(pairwise-int4), float4_e2m1fn_x2, float8_e4m3fn, float8_e4m3fn(3 substitutions for 4 fp6).
        block_tables_swa/block_tables_comp: int32.
        context_lens_swa/context_lens_comp: int32.
        q_quant_scale: float, float8_e8m0fnu, bfloat16.
        k_cache_quant_scale_lp/k_cache_quant_scale_hp: float.
        v_cache_quant_scale_lp/v_cache_quant_scale_hp: float.
        alibi_slopes: float.

    Return:
        torch.Tensor: The attention output tensor with shape (batch, seq_q, head_num_q, head_size_v).
            If return_lse is True, returns (output, lse) tuple where lse has shape (batch, head_num_q, seq_q).
    """
    tmo_out = out
    if out is None:
        if q.dtype not in {torch.float8_e4m3fn, torch.int8}:
            out_dtype = q.dtype
        tmo_out = torch.empty(q.size()[:-1] + (head_size_out,), dtype=out_dtype, device=q.device)
    lse_shape = q.shape[:-3] + (q.shape[-2],) + (q.shape[-3],)
    out_lse = torch.empty(lse_shape, dtype=torch.float, device=q.device)
    out_comp = torch.empty_like(tmo_out)
    lse2_comp = torch.empty_like(out_lse)
    torch.ops.torch_mlu_ops.single_query_cached_kv_attn(
            q, cache_swa, tmo_out, block_tables_swa, context_lens_swa, None, out_lse,
            q_quant_scale, cache_quant_scale_swa, None, out_quant_scale, None, None,
            _torchDtype2Str(compute_dtype), max_contxt_len_swa, -1, -1,
            softmax_scale, True, kv_cache_quant_bit_size, cu_seq_q, max_seq_q, sink)
    torch.ops.torch_mlu_ops.single_query_cached_kv_attn(
            q, cache_comp, out_comp, block_tables_comp, context_lens_comp, None, lse2_comp,
            q_quant_scale, cache_quant_scale_comp, None, out_quant_scale, None, None,
            _torchDtype2Str(compute_dtype), max_contxt_len_comp, -1, -1,
            softmax_scale, True, kv_cache_quant_bit_size, cu_seq_q, max_seq_q, None)
    torch.ops.torch_mlu_ops.update_out_and_lse(tmo_out, out_lse, out_comp, lse2_comp, None, None, None)
    return (tmo_out, out_lse) if return_lse else tmo_out

def apply_rotary(
    input: torch.Tensor,
    sin_cache: torch.Tensor,
    cos_cache: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    interleaved: bool,
    discrete: bool,
    dynamic_ntk: bool,
    max_seqlen: int,
    output: Optional[torch.Tensor] = None,
    is_inverse = False
) -> torch.Tensor:
    """
    Apply rotary embedding to the input tensor.

    Math:
        rotary_dim = sin_cache.shape[-1]
        input_rot = input[..., :rotary_dim].clone()
        if interleaved is True:
            x1, x2 = input_rot[..., ::2].clone(), input_rot[..., 1::2].clone()
            input_rot[..., ::2], input_rot[..., 1::2] = -x2, x1
        else:
            x1, x2 = input_rot.chunk(2, dim=-1)
            input_rot = concat((-x2, x1), dim=-1)
        input = input_rot * sin_cache[position_ids] + input * cos_cache[position_ids]
        return input

    Args:
        input (torch.Tensor): Shape is (batch, seq, head_num, head_size) or (total_seq, head_num, head_size).
                              The head_num is usually equal to head_num_q+head_num_k.
        sin_cache (torch.Tensor): If dynamic_ntk is true, the shape must be (batch, rotary_seq_len, rotary_dim),
            otherwise shape must be (rotary_seq_len, rotary_dim).
        cos_cache (torch.Tensor): If dynamic_ntk is true, the shape must be (batch, rotary_seq_len, rotary_dim),
            otherwise shape must be (rotary_seq_len, rotary_dim).
        position_ids (torch.Tensor): The position index of input tokens.
            If discrete is True, shape must be (batch, seq) or (total_seq), which indicates each token has a position index.
            If discrete is False, a tensor shape of (batch) specifies the start position of each batch,
                                  a tensor of None indicates start position is 0 for each batch.
        cu_seqlens (torch.Tensor): The sequence length of each batch in pack mode, which records the
            cumulative sequence lengths with shape (batch + 1). If not set, each batch's sequence length is max_seqlen.
        interleaved (bool): If interleaved is True, apply cross rotary embedding, otherwise apply fold rotary embedding.
        discrete (bool): Indicates whether position_ids is discrete. If True, each token has its own position_ids.
                         If False, only the start position of each batch is needed.
        dynamic_ntk (bool): If True, each batch has a different sin_cache and cos_cache with shape (batch, rotary_seq_len, rotary_dim).
                           If False, all batches share the same sin_cache and cos_cache with shape (rotary_seq_len, rotary_dim).
        max_seqlen (int): The maximum sequence lengths of input. In pad mode it equals seq, in pack mode it equals
                         the maximum sequence length in current batches.
        output (torch.Tensor) optional: Shape is same as input. Default is None.
        is_inverse (bool): A boolean value indicates whether sin_cache need inverse. If True, sin_cache should be negated.
                          Default is False.

    Type:
        input: float, half, bfloat16.
        sin_cache: same as input.
        cos_cache: same as input.
        position_ids: int32.
        cu_seqlens: int32.
        output: same as input.

    Return:
        Return the output tensor if output exists, otherwise return input tensor.

    Note:
        1. head_size must be between 2 and 256.
        2. rope_dim must be between 2 and head_size.
        3. rope_dim % 2 must be 0.
        4. batch_size must be less than or equal to 10240.
        5. When seq is less than 9, discrete is False and dynamic_ntk is False, pad mode(cu_seqlens is None) can get better performance than pack mode.
        6. input.stride(-2) == head_size can get better performance than input.stride(-2) > head_size.
    """
    if output is None:
        output = input
    torch.ops.torch_mlu_ops.apply_rotary(
        input,
        output,
        sin_cache,
        cos_cache,
        position_ids,
        cu_seqlens,
        interleaved,
        discrete,
        dynamic_ntk,
        max_seqlen,
        is_inverse
    )
    return output

def reshape_linear_cache(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: Optional[torch.Tensor],
    context_lengths: torch.Tensor,
    max_context_len: int,
    packed: bool,
    context_seq_offset: Optional[torch.Tensor],
    cache_bs_id: Optional[torch.Tensor],
    cache_seqlen_offset: Optional[torch.Tensor],
) -> Union[Tuple[torch.Tensor], torch.Tensor]:
    """
    Put key and value into key_cache and value_cache.

    Math:
        for idx in range(batch):
            key_cache[idx] = key[idx][:, context_length[idx]]
            value_cache[idx] = value[idx][:, context_length[idx]]
    Args:
        key (torch.Tensor): The key tensor. If use pad mode, shape is (batch, seqlen, head_num_kv, head_size),
                            else, shape is (total_seqlen, head_num_kv, head_size).
        value (torch.Tensor): The value tensor. Shape is the same as key.
        key_cache (torch.Tensor): The key_cache tensor. Shape is (max_batch, head_num_kv, cache_mem_len, head_size).
        value_cache (torch.Tensor): The value_cache tensor. Shape is the same as key_cache.
        context_lengths (torch.Tensor): Store key or cache lengths. If packed, shape is (batch + 1), else
                                        shape is (batch).
        max_context_len (int): The maximum sequence length of context.
        packed (bool): A boolean value indicates whether to use pack mode.
        context_seq_offset (torch.Tensor): Store the sequence offset of context. Shape if (batch).
        cache_bs_id (torch.Tensor): Indicate context will be put into which batch of cache. A negative value
                                    indicates that the corresponding batch will be ignored. Shape is (batch).
        cache_seqlen_offset (torch.Tensor): Indicate the offset of the position in the cache. Shape is (batch).
    Type:
        key: float, half, bfloat16, int8.
        value: float, half, bfloat16, int8.
        key_cache: float, half, bfloat16, int8.
        value_cache: float, half, bfloat16, int8.
        context_lengths: int32.
        max_context_len: int32.
        packed: bool.
        context_seq_offset: int32.
        cache_bs_id: int32.
        cache_seqlen_offset: int32.
    Return:
        Support inplace outputs.
        Directly return the given key_cache and value_cache if value_cache existed, else return key_cache only.
    """
    torch.ops.torch_mlu_ops.reshape_linear_cache(
        key,
        value,
        key_cache,
        value_cache,
        context_lengths,
        max_context_len,
        packed,
        context_seq_offset,
        cache_bs_id,
        cache_seqlen_offset,
    )
    return (key_cache, value_cache) if value_cache is not None else key_cache

def reshape_paged_cache(
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    slot_mapping: torch.Tensor,
    direction: bool = False
) -> Union[Tuple[torch.Tensor], torch.Tensor]:
    """
    Perform reshape_paged_cache operation.

    Math:
        for i in range(num_tokens):
            if slot_mapping[i] >= 0:
                block_id = torch.div(slot_mapping[i], block_size, rounding_mode='floor')
                block_offset = slot_mapping[i] % block_size
                k_cache[block_id, :, block_offset, :] = k[i]
                v_cache[block_id, :, block_offset, :] = v[i]

    Args:
        k (torch.Tensor): The key tensor. Shape is (num_token, head_num_kv, head_size).
        v (torch.Tensor): The value tensor. Shape is the same as k.
        k_cache (torch.Tensor): The key_cache tensor. Shape is (block_num, head_num_kv, block_size, head_size).
        v_cache (torch.Tensor): The value_cache tensor. Shape is the same as key_cache.
        slot_mapping (torch.Tensor): The slot_mapping tensor. Shape is (num_token).
        direction (Bool): If direction is True, gather kv from kv_cache, otherwise, scatter kv to kv_cache.

    Types:
        k: float, half, bfloat16, int8.
        v: float, half, bfloat16, int8.
        k_cache: float, half, bfloat16, int8.
        v_cache: float, half, bfloat16, int8.
        slot_mapping: int32.

    Return:
        Support inplace outputs.
        If direction is True, directly return the given k and v if v existed, else return k only.
        If direction is False, directly return the given k_cache and v_cache if v_cache existed, else return k_cache only.
    """
    torch.ops.torch_mlu_ops.reshape_paged_cache(k, v, k_cache, v_cache, slot_mapping, direction)
    if direction:
        return (k, v) if v is not None else k
    else:
        return (k_cache, v_cache) if v_cache is not None else k_cache

class TransferDirection(IntEnum):
    """Specify the transfer direction from cache to context(H2D) or reverse. """
    H2D = 0
    D2H = 1

class KVFormat(IntEnum):
    """
        The KV Format of paged cache in device
        TWO_NB_HN_BS_HS: [2, num_blocks, num_heads, block_size, head_size]
        NB_TWO_HN_BS_HS: [num_blocks, 2, num_heads, block_size, head_size]
        NB_HN_BS_HS : [num_blocks, num_heads, block_size, head_size]
    """
    TWO_NB_HN_BS_HS = 0
    NB_TWO_HN_BS_HS = 1
    NB_HN_BS_HS = 2

class PageBufferShapeDesc:
    """
        kv_size: key value dim size, 1 or 2
        nl: number of layers
        nb: number of blocks
        bs: block size
        nh: number of heads
        hs: head size
        element_size: element size in bytes
    """
    def __init__(self, kv_size, nl, nb, bs, nh, hs, element_size):
        self.kv_size = kv_size
        self.nl = nl
        self.nb = nb
        self.bs = bs
        self.nh = nh
        self.hs = hs
        self.element_size = element_size

def single_layer_kv_transfer(lmc_key_value_cache: torch.Tensor,
                             vllm_key_value_cache: torch.Tensor,
                             slot_mapping: torch.Tensor,
                             direction: TransferDirection,
                             kv_format: KVFormat = 0,
                             token_major: bool = False) -> None:
    """
    Transfer data between lmc_key_value_cache(host or device) and vllm_key_value_cache(device only).
    Math:
        for i in range(num_tokens):
            if slot_mapping[i] >= 0:
                block_id = torch.div(slot_mapping[i], block_size, rounding_mode='floor')
                block_offset = slot_mapping[i] % block_size
                if direction == H2D:
                    if token_major:
                        vllm_key_value_cache[0, block_id, block_offset, :, :] = lmc_key_cache[i]
                        vllm_key_value_cache[1, block_id, block_offset, :, :] = lmc_value_cache[i]
                elif direction == D2H:
                    lmc_key_cache[i] = vllm_key_value_cache[0, block_id, block_offset, :, :]
                    lmc_value_cache[i] = vllm_key_value_cache[1, block_id, block_offset, :, :]
    Args:
        lmc_key_value_cache (torch.Tensor): The lmc_key_value_cache tensor. Supported shapes are as following:
            - [2, num_token, head_num*head_size], token_major is True.
            - [num_token, 2, head_num*head_size], token_major is False.
            - [num_token, head_num*head_size], MLA mode, token_major must be True.
            Both device "cpu" and "mlu" are supported.
        vllm_key_value_cache (torch.Tensor): The vllm_key_value_cache tensor. Supported shapes are as following:
            - [2, num_blocks, num_heads, block_size, head_size], kv_format is TWO_NB_HN_BS_HS(0).
            - [num_blocks, 2, num_heads, block_size, head_size], kv_format is NB_TWO_HN_BS_HS(1).
            - [num_block, num_heads, block_size, head_size], MLA, kv_format is NB_HN_BS_HS(2).
            Only device "mlu" is supported.
        slot_mapping (torch.Tensor): The slot_mapping tensor. Shape is (num_token).
        direction (TransferDirection): Control the transfer direction.
             - H2D context to cache, lmc_key_value_cache to vllm_key_value_cache.
             - D2H cache to context, vllm_key_value_cache to lmc_key_value_cache.
        kv_format (KVFormat): kv_format can only be followings:
            - TWO_NB_HN_BS_HS(0).
            - NB_TWO_HN_BS_HS(1).
            - NB_HN_BS_HS(2).
        token_major (bool): Refer to lmc_key_value_cache.

    Types:
        lmc_key_value_cache: float, half, bfloat16, int8.
        vllm_key_value_cache: float, half, bfloat16, int8.
        slot_mapping: int32.

    Return:
        return None.
    """
    # 修正：
    torch.ops.torch_mlu_ops.single_layer_kv_transfer(lmc_key_value_cache, vllm_key_value_cache, slot_mapping,
                                                    direction, kv_format, token_major)


def multi_layer_kv_transfer(
    key_value: torch.Tensor,
    kv_cache_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    paged_memory_device: torch.device,
    page_buffer_size: int,
    direction: TransferDirection,
    kv_format: KVFormat,
    block_size: int,
    head_size: int,
    skip_prefix_n_tokens: int) -> None:
    """
    Transfer data between kv and kv_cache.
    Math:
        slot_mapping = slot_mapping[skip_prefix_n_tokens:]
        kv = kv[skip_prefix_n_tokens:]
        for i in range(num_tokens):
            if slot_mapping[i] >= 0:
                block_id = torch.div(slot_mapping[i], block_size, rounding_mode='floor')
                block_offset = slot_mapping[i] % block_size
                for j in range(layer_num):
                  k_cur_layer = k[j]
                  v_cur_layer = v[j]
                  kv_cache_cur_layer = multi_kv_cache[j]
                  if direction:
                      k_cur_layer[i] = kv_cache_cur_layer[0, block_id, :, block_offset, :]
                      v_cur_layer[i] = kv_cache_cur_layer[1, block_id, :, block_offset, :]
                  else:
                      kv_cache_cur_layer[0, block_id, :, block_offset, :] = k_cur_layer[i]
                      kv_cache_cur_layer[1, block_id, :, block_offset, :] = v_cur_layer[i]
    Args:
        key_value (torch.Tensor): The key_value tensor.Supported shapes are as following:
            - [num_layer, num_token, head_num*head_size], MLA mode.
            - [2, num_layer, num_token, head_num*head_size].
            Both device "cpu" and "mlu" are supported.
        kv_cache_ptrs (torch.Tensor): A tensor stores kv_cache pointer of each layer. Shape is (num_layer).
        kv_cache_ptrs (torch.Tensor): A tensor stores kv_cache pointer of each layer. Shape is (num_layer).
            Each pointer points to a kv_cache, Supported shapes are as following:
            - [2, num_blocks, num_heads, block_size, head_size], kv_format is TWO_NB_HN_BS_HS(0).
            - [num_blocks, 2, num_heads, block_size, head_size], kv_format is NB_TWO_HN_BS_HS(1).
            - [num_block, num_heads, block_size, head_size], MLA, kv_format is NB_HN_BS_HS(2).
            Only device "mlu" is supported.
        slot_mapping (torch.Tensor): The slot_mapping tensor. Shape is (num_token).
        paged_memory_device (torch.device): The device of kv_cache. This parameter is reserved for future use.
        page_buffer_size (int): Total size of kv_cache, equals to blocks_num*blocks_size
        direction (TransferDirection): Control the transfer direction.
             - H2D context to cache, key_value to key_value_cache.
             - D2H cache to context, key_value_cache to key_value.
        kv_format (KVFormat): Format of kv_cache, can only be followings:
            - TWO_NB_HN_BS_HS(0).
            - NB_TWO_HN_BS_HS(1).
            - NB_HN_BS_HS(2).
        block_size (int): Block size of kv cache.
        head_size (int): Head size of kv cache.
        skip_prefix_n_tokens (int): Skip first n tokens when transferring.

    Types:
        kv: float, half, bfloat16, int8.
        kv_cache_ptrs: int64.
        slot_mapping: int32.

    Return:
        return None.
    """
    torch.ops.torch_mlu_ops.multi_layer_kv_transfer(key_value, kv_cache_ptrs, slot_mapping, 0,
                                                    page_buffer_size, direction, kv_format, block_size, head_size,
                                                    skip_prefix_n_tokens)


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs: torch.Tensor,
    lmcache_objects_ptrs: torch.Tensor,
    block_ids: torch.Tensor,
    paged_memory_device: torch.device,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    kv_format: KVFormat,
    skip_prefix_n_blocks: int) -> None:
    """
    Transfer KV cache data between LMCache objects and VLLM paged cache at block granularity.

    Math:
        For each lmc_obj in lmcache_objects_ptrs[skip_prefix_n_blocks:]:
            for each layer:
                for each block in lmc_obj:
                    block_id = block_ids[block_idx]
                    if direction == H2D:
                        vllm_kv_cache[layer][block_id] = lmc_obj[layer][block]
                    else:
                        lmc_obj[layer][block] = vllm_kv_cache[layer][block_id]

    Args:
        paged_buffer_ptrs (torch.Tensor): A tensor storing kv_cache pointer of each layer.
            Shape is (num_layer). Only device "mlu" is supported.
        lmcache_objects_ptrs (torch.Tensor): A tensor storing LMCache object pointers.
            Shape is (num_objects,). Supports both "cpu" and "mlu" devices.
        block_ids (torch.Tensor): Block ID mapping from lmc to vllm.
            Shape is (num_objects * blocks_per_object,). Type is int64.
        paged_memory_device (torch.device): The device of kv_cache, currently unused.
        direction (TransferDirection): Control the transfer direction.
            - H2D: context to cache, LMCache to VLLM paged cache.
            - D2H: cache to context, VLLM paged cache to LMCache.
        shape_desc (PageBufferShapeDesc): Shape descriptor for the paged cache.
        lmcache_chunk_size (int): Number of tokens per LMCache object.
        kv_format (KVFormat): Format of kv_cache, can only be:
            - TWO_NB_HN_BS_HS(0).
            - NB_TWO_HN_BS_HS(1).
            - NB_HN_BS_HS(2).
        skip_prefix_n_blocks (int): Skip first n blocks when transferring.

    Types:
        paged_buffer_ptrs: int64.
        lmcache_objects_ptrs: int64.
        block_ids: int64.

    Return:
        return None.
    """
    torch.ops.torch_mlu_ops.multi_layer_block_kv_transfer(
        paged_buffer_ptrs, lmcache_objects_ptrs, block_ids, 0,
        direction, shape_desc.kv_size, shape_desc.nb, shape_desc.bs,
        shape_desc.nh, shape_desc.hs, shape_desc.element_size,
        lmcache_chunk_size, kv_format, skip_prefix_n_blocks)


def quant_to_paged_cache(
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    k_cache_quant_scale: torch.Tensor,
    v_cache_quant_scale: Optional[torch.Tensor],
    slot_mapping: torch.Tensor,
) -> Tuple[torch.Tensor]:
    """
    Perform the dynamic per-token quantization on key and value, then store the
    quantized results to the paged caches and scales.

    Math:
        for i in range(tokens_num):
            key_i = k[i]
            max_value = max(key_i.abs(), dim=-1)
            if k_cache.dtype == int8:
                int_max = float(2 ** (quant_bit - 1) - 1)
            elif k_cache_dtype == fp8:
                int_max = 448
            scale_i = float(max_value / int_max)
            key_i /= scale_i
            if k_cache.dtype == int8:
                k_i = clip(round(key_i), -2 ** (quant_bit - 1), 2 ** (quant_bit - 1) - 1)
            k_cache[i] = k_i
            k_cache_quant_scale[i] = scale_i

    Args:
        k (torch.Tensor): The key tensor. Shape is [token_nums, head_num_kv, head_size].
        v (torch.Tensor): The value tensor. Shape is the same as k.
        k_cache (torch.Tensor): The k_cache tensor. Shape is [block_nums, head_num_kv, block_size, head_size].
        v_cache (torch.Tensor): The v_cache tensor. Shape is the same as v_cache.
        k_cache_quant_scale (torch.Tensor): The k_cache_quant_scale tensor. Shape is [block_nums, head_num_kv, block_size].
        v_cache_quant_scale (torch.Tensor): The v_cache_quant_scale tensor. Shape is the same as v_cache_quant_scale.
        slot_mapping (torch.Tensor): The slot_mapping tensor. Shape is [token_nums].

    Type:
        k: float, half, bfloat16.
        v: float, half, bfloat16.
        k_cache: int8, float8_e4m3fn.
        v_cache: int8, float8_e4m3fn.
        k_cache_quant_scale: float.
        v_cache_quant_scale: float.
        slot_mapping: int32.

    Return:
        Support inplace outputs.
        Directly return the given k_cache, v_cache, k_cache_quant_scale and v_cache_quant_scale if v_cache existed,
        else return k_cache and k_cache_quant_scale.
    """
    torch.ops.torch_mlu_ops.quant_to_paged_cache(
        k, v, k_cache, v_cache, k_cache_quant_scale, v_cache_quant_scale, slot_mapping
    )
    return (k_cache, v_cache, k_cache_quant_scale, v_cache_quant_scale) if v_cache is not None else (k_cache, k_cache_quant_scale)

def quant_mx_to_paged_cache(
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor],
    k_cache_quant_scale: torch.Tensor,
    v_cache_quant_scale: Optional[torch.Tensor],
    slot_mapping: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    recent_v: Optional[torch.Tensor],
    recent_seqlens: Optional[torch.Tensor],
    recent_slotmapping: Optional[torch.Tensor],
    quant_bits: int
) -> Tuple[torch.Tensor]:
    """
    Perform the microscaling(MX) quantization on key and value, then store the
    quantized results to the paged caches and scales.

    Math:
        Given k with shape of (num_tokens, num_heads, dim_k):
        k_chunks = k.split(mx_group_size, dim=-1)
        [(quant_ki, scale_ki) = per_tensor_quantize(qi) for qi in q_chunks]
        quant_k = torch.cat([quant_ki], dim=-1)
        k_scale = torch.cat([scale_ki], dim=-1)

        Given v with shape of (num_tokens, num_heads, dim_v) and recent_v (num_batch, num_heads, 32, dim_v):
        for i in range(0, num_batch):
            v_seg = concat(v[cu_seqlens[i]: cu_seqlens[i+1]], recent_v[0: recent_seqlens[i]])
            v_seg = padUp(v_seg[i], mx_group_size, dim=0)
            all_v = concat(all_v, v_seg)
        all_v = all_v.permute(1, 2, 0), doing the quantization the same as k.

    Args:
        k (torch.Tensor)                             : The key tensor, Shape is [num_tokens, num_heads, dim_k].
        v (Optional[torch.Tensor])                   : The value tensor, Shape is [num_tokens, num_heads, dim_v].
        k_cache (torch.Tensor)                       : The key cache tensor, Shape is [num_blocks, num_heads, block_size, dim_kcache].
        v_cache  (Optional[torch.Tensor])            : The value cache tensor, Shape is [num_blocks, num_heads, block_size, dim_vcache].
        k_cache_quant_scale (torch.Tensor)           : The key scale tensor, Shape is [num_blocks, num_heads, block_size, dim_kscale].
        v_cache_quant_scale (Optional[torch.Tensor]) : The value scale tensor, Shape is [num_blocks, num_heads, block_size//32, dim_vscale].
        slot_mapping (torch.Tensor)                  : The slot mapping tensor, Shape is [num_tokens].
        cu_seqlens (Optional[torch.Tensor])          : The seqlens cusum tensor, Shape is [num_batch + 1].
        recent_v (Optional[torch.Tensor])            : The recent value tensor, Shape is [num_batch, num_heads, 32, dim_v].
        recent_seqlens (Optional[torch.Tensor])      : The seqlens tensor of recent value, Shape is [num_batch].
        recent_slotmapping (Optional[torch.Tensor])  : The slot mapping tensor of recent value, Shape is [num_batch, 32].
        quant_bits (torch.int32)                     : The bit size of key/value cache element.

    Types:
        k : half, bfloat16
        v : half, bfloat16
        k_cache : float6_e3m2fn
        v_cache : float6_e3m2fn
        k_cache_quant_scale : bfloat16
        v_cache_quant_scale : bfloat16
        slot_mapping : int32
        cu_seqlens : int32
        recent_v : bfloat16
        recent_seqlens : int32
        recent_slotmapping : int32
        quant_bits : int32

    Returns:
        Support inplace outputs.
        Directly return the given k_cache, v_cache, k_cache_quant_scale and v_cache_quant_scale if v_cache existed,
        else return k_cache and k_cache_quant_scale.
        Write recent_v, recent_seqlens and recent_slotmapping inplace.
    """
    torch.ops.torch_mlu_ops.quant_mx_to_paged_cache(
        k, v, k_cache, v_cache, k_cache_quant_scale, v_cache_quant_scale, slot_mapping,
        cu_seqlens, recent_v, recent_seqlens, recent_slotmapping, quant_bits
    )
    return (k_cache, v_cache, k_cache_quant_scale, v_cache_quant_scale) if v_cache is not None else (k_cache, k_cache_quant_scale)

def offline_quant_to_paged_cache(
    k: torch.Tensor,
    v: Optional[torch.Tensor],
    k_cache_scale: Union[torch.Tensor, float],
    v_cache_scale: Optional[Union[torch.Tensor, float]],
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: Optional[torch.Tensor]
) -> Union[Tuple[torch.Tensor], torch.Tensor]:
    """
    Perform the static per-channel or per-tensor quantization on key and value, then store the
    quantized results to the paged caches and scales.

    Math:
        tokens_num = k.shape[0]
        block_size = k_cache.shape[2]
        for i in range(tokens_num):
            if slot_mapping[i] >= 0:
                key_i = k[i]
                value_i = v[i]
                block_id = torch.div(slot_mapping[i], block_size, rounding_mode='floor')
                block_offset = slot_mapping[i] % block_size
                key_cache_i = k_cache[block_id, :, block_offset, :]
                value_cache_i = v_cache[block_id, :, block_offset, :]
                if k_cache.dtype == int8:
                    quant_key_i = quant2int8(key_i, k_cache_scale)
                    quant_value_i = quant2int8(value_i, v_cache_scale)
                elif k_cache.dtype == fp8:
                    quant_key_i = quant2fp8(key_i, k_cache_scale)
                    quant_value_i = quant2fp8(value_i, v_cache_scale)
                k_cache_i[...] = quant_key_i
                v_cache_i[...] = quant_value_i

    Args:
        k (torch.Tensor): The key tensor, shape is (num_tokens, num_heads, head_size).
        v (torch.Tensor): The value tensor. shape is (num_tokens, num_heads, head_size).
        k_cache_scale (Union[torch.Tensor, float]): A tensor shape of (num_heads, head_size) indicates the per-channel quantization,
                                                    otherwise a float scalar indicates the per-tensor quantization.
        v_cache_scale (Optional[Union[torch.Tensor, float]]): A tensor shape of (num_heads, head_size) indicates the per-channel quantization,
                                                              otherwise a float scalar indicates the per-tensor quantization.
        slot_mapping (torch.Tensor): The slot_mapping tensor. shape is (num_tokens).
        k_cache (torch.Tensor): The key_cache tensor. shape is (num_blocks, num_heads, block_size, head_size).
        v_cache (torch.Tensor): The value_cache tensor. shape is (num_blocks, num_heads, block_size, head_size).

    Type:
        k: float, half, bfloat16.
        v: float, half, bfloat16.
        k_cache_scale: float.
        v_cache_scale: float.
        slot_mapping: int32.
        k_cache: int8, float8_e4m3fn.
        v_cache: int8, float8_e4m3fn.

    Return:
        Support inplace outputs.
        Directly return the given k_cache and v_cache if v_cache existed, else return k_cache only.

    """
    if torch.is_tensor(k_cache_scale):
        torch.ops.torch_mlu_ops.offline_quant_to_paged_cache(
            k, v, k_cache_scale, v_cache_scale, slot_mapping, k_cache, v_cache, 1.0, 1.0
        )
    else:
        torch.ops.torch_mlu_ops.offline_quant_to_paged_cache(
            k, v, None, None, slot_mapping, k_cache, v_cache, k_cache_scale,
            v_cache_scale if v_cache_scale is not None else 1.0,
        )
    return (k_cache, v_cache) if v_cache is not None else k_cache

def quant_to_linear_cache(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: Optional[torch.Tensor],
    key_cache_quant_scale: torch.Tensor,
    value_cache_quant_scale: Optional[torch.Tensor],
    context_lengths: torch.Tensor,
    max_context_len: int,
    packed: bool,
    context_seq_offset: Optional[torch.Tensor],
    cache_bs_id: Optional[torch.Tensor],
    cache_seqlen_offset: Optional[torch.Tensor],
    quant_bit: int = 8,
) -> Tuple[torch.Tensor]:
    """
    Perform the dynamic per-token quantization on key and value, then store the quantized
    results to the linear caches and scales.

    Math:
        for i in range(batch * seq_len):
            key_i = key[i]
            if groupwise quantize:
                key_i = key_i.reshape(head_num, -1, group_size)
            else:
                key_i = key_i.reshape(head_num, -1, head_size)
            max_value = max(key_i.abs(), dim=-1)
            if key_cache.dtype == int8:
                int_max = float(2 ** (quant_bit - 1) - 1)
            elif key_cache.dtype == fp8
                int_max = 448
            scale_i = max_value / int_max
            key_i /= scale_i
            if key_cache.dtype == int8:
                key_i = clip(round(key_i), -2 ** (quant_bit - 1), 2 ** (quant_bit - 1) - 1)
            elif key_cache.dtype == fp8:
                key_i = clip(key_i, -448, 448)
            if quant_bit == 4:
                key_i = key_i.flatten()
                d0 = key_i[0::2].to(uint8)
                d1 = key_i[0::2].to(uint8)
                dp = (d1 << 4) + (d0 & 0x0f)
                key_i = key_i.to(int8).reshape(head_num, -1, head_size // 2)
            k_cache[pos] = k_i
            k_cache_quant_scale[pos] = scale_i

    Args:
        key (torch.Tensor): The key tensor. If packed, shape is (total_seqlen, head_num_kv, head_size),
                            else, shape is (batch, seqlen, head_num_kv, head_size).
        value (torch.Tensor): The value tensor. Shape is the same as key.
        key_cache (torch.Tensor): The key cache tensor. Shape is (max_batch, head_num_kv, cache_mem_len, head_size).
        value_cache (torch.Tensor): The value cache tensor. Shape is the same as key_value.
        key_cache_quant_scale (torch.Tensor): The key_cache_quant_scale tensor. If use groupwise quantization, shape is
                                              (max_batch, head_num_kv, cache_mem_len, group_num), otherwise shape is
                                              (max_batch, head_num_kv, cache_mem_len).
        value_cache_quant_scale (torch.Tensor): The value_cache_quant_scale tensor. Shape is the same as key_cache_quant_scale.
        context_lengths (torch.Tensor): The context_lengths tensor. If packed, shape is (batch + 1), else, shape is (batch).
        max_context_len: The maximum sequence length of context.
        packed: A boolean value indicates whether to use pack mode.
        context_seq_offset (torch.Tensor): Indicate the sequence offset of context, shape is (batch).
        cache_bs_id (torch.Tensor): Indicate key and value will be put into which batch of kv_cache. A negative value indicates
                                    that the corresponding batch will be ignored. Shape is (batch).
        cache_seqlen_offset (torch.Tensor): Indicate the offset of the position in the cache. Shape is (batch).

    Types:
        key: float, half, bfloat16.
        value: float, half, bfloat16.
        key_cache: int8, float8_e4m3fn.
        value_cache: int8, float8_e4m3fn.
        key_cache_quant_scale: float32.
        value_cache_quant_scale: float32.
        context_lengths: int32.
        max_context_len: int32.
        packed: bool.
        context_seq_offset: int32.
        cache_bs_id: int32.
        cache_seqlen_offset: int32.

    Return:
        Support inplace outputs.
        Directly return the given key_cache, value_cache, key_cache_quant_scale and value_cache_quant_scale if value_cache existed,
        else return key_cache and key_cache_quant_scale.
    """
    torch.ops.torch_mlu_ops.quant_to_linear_cache(
        key,
        value,
        key_cache,
        value_cache,
        key_cache_quant_scale,
        value_cache_quant_scale,
        context_lengths,
        max_context_len,
        packed,
        context_seq_offset,
        cache_bs_id,
        cache_seqlen_offset,
        quant_bit,
    )
    return (key_cache, value_cache, key_cache_quant_scale, value_cache_quant_scale) if value_cache is not None else (key_cache, key_cache_quant_scale)

def offline_quant_to_linear_cache(
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    key_cache: torch.Tensor,
    value_cache: Optional[torch.Tensor],
    key_cache_quant_scale: Union[torch.Tensor, float],
    value_cache_quant_scale: Optional[Union[torch.Tensor, float]],
    context_lengths: torch.Tensor,
    max_context_len: int,
    quant_mode: int,
    packed: bool,
    context_seq_offset: Optional[torch.Tensor],
    cache_bs_id: Optional[torch.Tensor],
    cache_seqlen_offset: Optional[torch.Tensor],
) -> Tuple[torch.Tensor]:
    """
    Perform the static per-channel, per-head or per-tensor quantization on key and value, then store the
    quantized results to the linear caches and scales.

    Math:
        for i in range(batch):
            key_i = key[i].transpose(1, 0)
            if quant_mode == 0:
                key_i /= key_cache_quant_scale.reshape(head_num, 1, head_size)
            elif quant_mode == 1:
                key_i /= key_cache_quant_scale.reshape(head_num, seq, 1)
            else:
                key_i /= key_cache_quant_scale
            if key_cache.dtype == int8:
                key_i = clip(round(key_i), -128, 127)
            elif key_cache.dtype == fp8:
                key_i = clip(key_i, -448, 448)
            key_cache[pos] = key_i

    Args:
        key (torch.Tensor): The key tensor. If packed, shape is (total_seqlen, head_num_kv, head_size),
                            else, shape is (batch, seqlen, head_num_kv, head_size).
        value (torch.Tensor): The value tensor. Shape is the same as key.
        key_cache (torch.Tensor): The key_cache tensor. Shape is (max_batch, head_num, cache_mem_len, head_size).
        value_cache (torch.Tensor): The value_cache tensor. Shape is the same as key_cache.
        key_cache_quant_scale (Union[torch.Tensor, float]): If per_channel quantize, shape is (head_num_kv, head_size).
                                                            If per_head quantize, shape is (head_num_kv, cache_mem_len).
                                                            If per_tensor quantize, key_cache_quant_scale is a float scalar.
        value_cache_quant_scale (Optional[Union[torch.Tensor, float]]): Shape is the same as key_cache_quant_scale.
        context_lengths (torch.Tensor): A tensor indicate context lengths. Shape is (batch+1) if packed is True, which
                                        stores the cumsum of seqlen, otherwise shape is (batch).
        max_context_len: The maximum sequence length of context.
        quant_mode: A int value indicates which quantize mode will be used. Support 0, 1, 2, 3. If quant_mode is 0, use per_channel
                    quantize, if quant_mode is 1, use per_head quantize, otherwise use per_tensor quantize.
        packed: A boolean value indicates whether to use pack mode.
        context_seq_offset (torch.Tensor): Indicate the sequence offset of context, shape is (batch).
        cache_bs_id (torch.Tensor): Indicate context will be put into which batch of cache. A negative value
                                    indicates that the corresponding batch will be ignored. Shape is (batch).
        cache_seqlen_offset (torch.Tensor): Indicate the offset of the position in the cache. Shape is (batch).

    Types:
        key: float, half, bfloat16.
        value: float, half, bfloat16.
        key_cache: int8, float8_e4m3fn.
        value_cache: int8, float8_e4m3fn.
        key_cache_quant_scale: float32.
        value_cache_quant_scale: float32.
        context_lengths: int32.
        max_context_len: int32.
        quant_mode: int32.
        packed: bool.
        context_seq_offset: int32.
        cache_bs_id: int32.
        cache_seqlen_offset: int32.

    Return:
        Support inplace outputs.
        Directly return the given key_cache, value_cache, key_cache_quant_scale and value_cache_quant_scale if value_cache existed,
        else return key_cache and key_cache_quant_scale.
    """
    if torch.is_tensor(key_cache_quant_scale):
        torch.ops.torch_mlu_ops.offline_quant_to_linear_cache(
            key,
            value,
            key_cache,
            value_cache,
            key_cache_quant_scale,
            value_cache_quant_scale,
            context_lengths,
            max_context_len,
            quant_mode,
            packed,
            context_seq_offset,
            cache_bs_id,
            cache_seqlen_offset,
            1.0,
            1.0
        )
    else:
        torch.ops.torch_mlu_ops.offline_quant_to_linear_cache(
            key,
            value,
            key_cache,
            value_cache,
            None,
            None,
            context_lengths,
            max_context_len,
            quant_mode,
            packed,
            context_seq_offset,
            cache_bs_id,
            cache_seqlen_offset,
            key_cache_quant_scale,
            value_cache_quant_scale if value_cache_quant_scale is not None else 1.0
        )
    return (key_cache, value_cache, key_cache_quant_scale, value_cache_quant_scale) if value_cache is not None else (key_cache, key_cache_quant_scale)

def ssparse_group_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_list: torch.Tensor,
    max_m: int,
    gather_idx: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    dtype: torch.dtype = None,
    act_mode: str = None,
    alpha: float = 1.0,
    beta: float = .0,
    trans_a: bool = False,
    trans_b: bool = True,
    output: torch.Tensor = None
) -> torch.Tensor:
    """
    Perform structured sparse grouped matrix multiplication with quantization support.

    Math:
        1. For compressed matrix B (containing both data and indices):
           - Directly use the indices to select corresponding elements from A
           - Multiply selected A elements with B's data blocks
        2. Apply scaling: a = a * a_scale, b_data = b_data * b_scale
        3. Split tensors according to m_list (expert grouping)
        4. For each expert group:
           output[i] = alpha * sparse_matmul(a[i], b[i]) + beta * c[i] + bias[i]
           where sparse_matmul uses B's indices to skip zero computations

    Args:
        a (torch.Tensor): Input tensor A, shape (total_m, k) if not trans_a else (k, total_m).
        b (torch.Tensor): Compressed input tensor B (data + indices), shape (experts_num, k, n) if trans_b else (experts_num, n, k).
        a_scale (torch.Tensor): Scaling factors for A, shape (total_m, 1).
        b_scale (torch.Tensor): Scaling factors for B, shape (experts_num, n, 1).
        m_list (torch.Tensor): List of row counts per expert, shape (experts_num,).
        max_m (int): Maximum value in m_list.
        gather_idx (Optional[torch.Tensor]): Optional gather index tensor.
        bias (Optional[torch.Tensor]): Optional bias tensor, shape (experts_num, n).
        c (Optional[torch.Tensor]): Optional residual tensor, shape (total_m, n).
        dtype (torch.dtype): Output data type.
        act_mode (str): Activation mode ('none').
        alpha (float): Scaling factor for matrix product.
        beta (float): Scaling factor for residual.
        trans_a (bool): Whether to transpose A.
        trans_b (bool): Whether to transpose B (default True).
        output (Optional[torch.Tensor]): Optional output tensor.

    Type:
        a: int8
        b: int8
        a_scale: float32
        b_scale: float32
        m_list: int32
        gather_idx: int32 (if provided)
        bias: float16/bfloat16 (matches dtype)
        c: float16/bfloat16 (matches dtype)
        output: float16/bfloat16 (matches dtype)

    Return:
        torch.Tensor: Output tensor of shape (total_m, n)
    """
    n = b.size(-2) if trans_b else b.size(-1)
    output = output if output is not None else torch.empty(a_scale.size(0), n, dtype=dtype, device=a.device)

    torch.ops.torch_mlu_ops.ssparse_matmul(
        a, b, a_scale, b_scale,
        output, act_mode,
        m_list, None,
        bias, c,
        max_m, alpha, beta,
        trans_a, trans_b
    )

    return output

def ssparse_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    dtype: torch.dtype = None,
    act_mode: str = None,
    alpha: float = 1.0,
    beta: float = .0,
    trans_a: bool = False,
    trans_b: bool = True,
    output: torch.Tensor = None
) -> torch.Tensor:
    """
    Perform structured sparse group matrix multiplication with compressed format.

    Math:
        1. Apply quantization scaling: a = a * a_scale
        2. Compute with compressed B (union_mat format):
           - B contains compressed weights and selection indices
           - For each group in B, use indices to select corresponding elements from A
           - Multiply selected A elements with compressed weights in B
        3. Scale result: out = alpha * computed_product
        4. Optional operations:
           - Add residual: out += beta * c if c exists
           - Add bias: out += bias if bias exists
           - Apply activation: out = activation(out) if act_mode specified

    Args:
        a (torch.Tensor): Input tensor A with shape (M, K) if not trans_a, else (K, M).
        b (torch.Tensor): Compressed input tensor B (union_mat) with shape (K, N) if not trans_b, else (N, K).
        a_scale (torch.Tensor): Quantization scale for A with shape (M, 1).
        b_scale (torch.Tensor): Quantization scale for B with shape (N, 1).
        bias (Optional[torch.Tensor]): Bias tensor with shape (N,).
        c (Optional[torch.Tensor]): Residual tensor with shape (M, N).
        dtype (torch.dtype): Output data type.
        act_mode (str): Activation mode, supports 'none' or 'gelu'.
        alpha (float): Scaling factor for matrix multiplication result.
        beta (float): Scaling factor for residual tensor.
        trans_a (bool): Whether to transpose A before multiplication (only support False).
        trans_b (bool): Whether to transpose B before multiplication (only support True).
        output (torch.Tensor): Optional output tensor for in-place operation.

    Type:
        a: int8 (quantized)
        b: int8 (compressed format containing weights and indices)
        a_scale: float32
        b_scale: float32
        bias: float16/bfloat16
        c: float16/bfloat16
        output: float16/bfloat16

    Return:
        torch.Tensor: Output tensor with shape (M, N)
    """
    m = a.size(-1) if trans_a else a.size(-2)
    n = b.size(-2) if trans_b else b.size(-1)
    output = output if output is not None else torch.empty(m, n, dtype=dtype, device=a.device)

    torch.ops.torch_mlu_ops.ssparse_matmul(
        a, b, a_scale, b_scale,
        output, act_mode,
        None, None,
        bias, c,
        0, alpha, beta,
        trans_a, trans_b
    )
    return output

def swap_blocks(dst: torch.Tensor,
                src: torch.Tensor,
                block_mapping: Dict[int, int],
                block_size_in_bytes: Optional[int] = None) -> torch.Tensor:
    """
    Copy src value to dst according to block_mapping.

    Math:
        for key, value in block_mapping.items():
            dst[value] = src[key]

    Args:
        dst (torch.Tensor): Destination tensor. Shape is (num_blocks, ...).
        src (torch.Tensor): Source tensor. Shape is (num_blocks, ...).
        block_mapping (dict): Mapping table of src and dst.
        block_size_in_bytes (int, optional): Size in bytes to copy from src to dst per block. If not set, copies the full src block.

    Types:
        dst: Unlimited.
        src: Unlimited.
        block_mapping: [int, int].
        block_size_in_bytes: int.

    Return:
        Support inplace outputs.
        Directly return the given dst.
    """
    torch.ops.torch_mlu_ops.swap_blocks(dst, src, block_mapping, block_size_in_bytes)
    return dst

def copy_blocks(
    k_caches: List[torch.Tensor],
    v_caches: Optional[List[torch.Tensor]],
    block_mapping: Dict[int, List[int]]
) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], List[torch.Tensor]]:
    """
    Remap the k_caches and v_caches with the block indexes pairs specified by block_mapping.

    Math:
        for src, dsts in block_mapping.items():
            srcs = [src for i in range(len(dsts))]
            for key_cache in k_caches:
                key_cache[dsts] = key_cache[srcs]
            for value_cache in v_caches:
                value_cache[dsts] = value_cache[srcs]

    Args:
        k_caches (List[torch.Tensor]): A tensor list of k_cache.
        v_caches (List[torch.Tensor]): A tensor list of v_cache.
        block_mapping (Dict[int, List[int]]): The pairs of source and destination block indexes.

    Type:
        k_caches: int8, uint8, int16, int32, int64, half, float, bfloat16.
        v_caches: int8, uint8, int16, int32, int64, half, float, bfloat16.

    Return:
        Support inplace outputs.
        Directly return the given k_caches and v_caches.
    """
    torch.ops.torch_mlu_ops.copy_blocks(k_caches, v_caches if v_caches is not None else [], block_mapping)
    return (k_caches, v_caches) if v_caches is not None else k_caches

def active(input: torch.Tensor, act_mode: str, is_gated: bool, active_coef: float = 1.0, high_precision: bool = False, gelu_approximate: str = 'none') -> torch.Tensor:
    """
    Apply activation to the input tensor.

    Math:
        C = input.shape[-1]
        if is_gated = True:
            output = active(input[..., :C//2]) * input[..., C//2:]
        else:
            output = active(input)

    Args:
        input (torch.Tensor): Shape is (..., C)
        act_mode (str): The activation mode, must be 'silu', 'gelu', 'swish' or 'quick_gelu'.
        is_gated (bool): If use gated activation.
        active_coef (float): The coefficient used in the swish activation. Default is 1.0.
        high_precision (bool): If adopt high precision calculation. Default is false.
        gelu_approximate (str): If use tanh_gelu. Can be 'none' or 'tanh'. Default is 'none'.

    Type:
        input: float, half, bfloat16.
        output: same as input.

    Return:
        A tensor with the same shape and dtype of input.
    """
    out_shape=input.shape[:-1] + (input.shape[-1] // (1+is_gated),)
    output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    torch.ops.torch_mlu_ops.active(input, output, None, None, act_mode, is_gated, 0, 0,
        active_coef, high_precision, gelu_approximate)
    return output

def matmul(
        a: torch.Tensor,
        b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        act_mode: str = 'none',
        alpha: float = 1.0,
        beta: float = .0,
        fast_act: bool = True,
        approximate: bool = True,
        dtype: torch.dtype = None,
        a_scale: Optional[Union[torch.Tensor, float]] = None,
        b_scale: Optional[Union[torch.Tensor, float]] = None,
        trans_a: bool = False,
        trans_b: bool = True,
        tile_config: Optional[Dict[str, int]] = None,
        output: torch.Tensor = None
    ) -> torch.Tensor:
    """
    Perform matrix multiplication operation.

    Math:
        ab = alpha * matmul(a, b)
        if c is not None:
            ab = ab + beta * c
        if bias is not None:
            ab += bias
        if act_mode is not 'none':
            ab = active(ab)
        output = ab

    Args:
        a (torch.Tensor): If trans_a, shape is (k, m) or (bs_a, k, m), else shape is (m, k) or (bs_a, m, k).
        b (torch.Tensor): If trans_b, shape is (n, k) or (bs_b, n, k), else shape is (k, n) or (bs_b, n, k). bs_a = bs_b, or one of (bs_a, bs_b) is 1.
        bias (torch.Tensor): Shape is (n) or (bs, 1, n).
        c (torch.Tensor): Shape is (m, n) or (bs, m, n).
        act_mode (str): The activation type, must be one of {'gelu', 'relu', 'silu', 'sigmoid', 'none'} when a/b is 2-dim,
                        and must be one of {'gelu', 'silu', 'none'} when a/b is 3-dim.
        alpha (float): The coefficient of ab.
        beta (float): The coefficient of c.
        fast_act (bool): Describing the algorithm that used in the implementation of the activation function.
             When the value is true, use the fastest algorithm of activation, otherwise use the high-precision algorithm.
        approximate(bool): Describes the implementation logic of different GELU approximation algorithms.
        a_scale (Optional[Union[torch.Tensor, float]]): Quant_scale of a. Could be a device tensor shape as (1), or a scalar. Must be scalar when 3-dim input.
                a_scale = input_max/dtype_max when input's dtype is float8, and a_scale = dtype_max/input_max when input's dtype is int8.
        b_scale (Optional[Union[torch.Tensor, float]]): Quant_scale of b. The same as a_scale.
        tile_config (Dict[str, int]): optional. Tile config, not support when a/b is 3-dim. including TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, WARP_SCHEDULER, SPLIT_K, in the meanwhile, could assign SWIZZLE_SIZE and DIRECTION.
            eg. tile_config = {'TILE_SIZE_M': 32, 'TILE_SIZE_N': 256, 'TILE_SIZE_K': 128, 'WARP_SCHEDULER': 4, 'SPILT_K': 1},
            or, tile_config = {'TILE_SIZE_M': 32, 'TILE_SIZE_N': 256, 'TILE_SIZE_K': 128, 'WARP_SCHEDULER': 4, 'SPILT_K': 1, 'SWIZZLE_SIZE': 2, 'DIRECTION': 2}.
            SWIZZLE_SIZE must be divisor of the cluster number of the device, DIRECTION support {1, 2}.

    Type:
        a: float, half, bfloat16, int8, float8_e4m3fn, float8_e5m2.
        b: the same as a.
        bias: float, half, bfloat16.
        c: float, half, bfloat16.
        a_scale: float.
        b_scale: float.
        output: float, half, bfloat16.

    Return:
        Return output tensor.
    """
    a_scale_scalar = 1.0
    b_scale_scalar = 1.0
    if a_scale is not None and not torch.is_tensor(a_scale):
        a_scale_scalar = a_scale
        a_scale = None
    if b_scale is not None and not torch.is_tensor(b_scale):
        b_scale_scalar = b_scale
        b_scale = None

    if output is None:
        d_dtype = None if dtype is None else _torchDtype2Str(dtype)
        if tile_config is None:
            return torch.ops.torch_mlu_ops.matmul_aot_inductor(a, b, bias, c, a_scale, b_scale,
                    d_dtype, act_mode, alpha, beta, fast_act, approximate, a_scale_scalar,
                    b_scale_scalar, trans_a, trans_b)
        else:
            return torch.ops.torch_mlu_ops.matmul(a, b, bias, c, a_scale, b_scale, d_dtype,
                    tile_config, act_mode, alpha, beta, fast_act, approximate,
                    a_scale_scalar, b_scale_scalar, trans_a, trans_b)
    else: # inplace
        if tile_config is None:
            torch.ops.torch_mlu_ops.matmul_v2(a, b, output, bias, c, a_scale, b_scale, act_mode,
                    alpha, beta, fast_act, approximate, a_scale_scalar, b_scale_scalar, trans_a, trans_b)
        else:
            torch.ops.torch_mlu_ops.matmul_inplace(a, b, output, bias, c, a_scale, b_scale,
                    tile_config, act_mode, alpha, beta, fast_act, approximate, a_scale_scalar,
                    b_scale_scalar, trans_a, trans_b)
        return output

def batch_matmul(
        a: torch.Tensor,
        b: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = .0,
        a_scale: Optional[Union[torch.Tensor, float]] = None,
        b_scale: Optional[Union[torch.Tensor, float]] = None,
        trans_a: bool = False,
        trans_b: bool = True,
        d: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        act_mode = "none",
        dtype: torch.dtype = None,
        use_hp_active: bool = False,
        approximate: bool = False
    ) -> torch.Tensor:
    """
    Perform matrix multiplication operation.

    Math:
        ab = batch_matmul(a, b)
        if c is not None:
            d = alpha * ab + beta * c
        else:
            d = alpha * ab

    Args:
        a (torch.Tensor): Shape is (bs_a, k, m) or (bs_a, m, k), a, b, c, d must be contiguous if bias is not None or act_mode is not 'none'.
        b (torch.Tensor): Shape is (bs_b, n, k) or (bs_b, k, n). bs_a = bs_b, or one of (bs_a, bs_b) is 1, batch = max(bs_a, bs_b).
        c (torch.Tensor): Shape is (batch, m, n), c will be changed when the calculation is completed if bias is not None or act_mode is not 'none'.
        alpha (float): The coefficient of ab.
        beta (float): The coefficient of c.
        a_scale (Optional[Union[torch.Tensor, float]]): Quant_scale of a. a_scale must be scalar when act_mode != 'none' or bias is not None.
            a_scale = input_max/dtype_max when input's dtype is float8, and a_scale = dtype_max/input_max when input's dtype is int8.
        b_scale (Optional[Union[torch.Tensor, float]]): Quant_scale of b. Must be same as a_scale.
        trans_a (bool): Flag transpose of a.
        trans_b (bool): Flag transpose of b.
        d (torch.Tensor): Shape is (batch, m, n).
        bias (torch.Tensor): Shape is [batch, 1, n], or [1, 1, n], or [n].
        act_mode (str): The activation type, must be 'gelu', 'silu' or 'none'.
        dtype (torch.dtype): The dtype of output.
        use_hp_active (bool): Flag use high-precision algorithm or not to imply activation function.
        approximate (bool): Describes the implementation logic of different GELU approximation algorithms.
                        If set True, gelu(x) = 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3))),
                        else, gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))).


    Type:
        a: float, half, bfloat16, int8, float8_e4m3fn, float8_e5m2.
        b: same as a.
        c: float, half, bfloat16.
        bias: float, half, bfloat16.
        a_scale: float.
        b_scale: float.
        d: float, half, bfloat16.

    Return:
        Return output tensor.
    """
    a_scale_scalar = 1.0
    b_scale_scalar = 1.0
    if a_scale is not None and not torch.is_tensor(a_scale):
        a_scale_scalar = a_scale
        a_scale = None
    if b_scale is not None and not torch.is_tensor(b_scale):
        b_scale_scalar = b_scale
        b_scale = None
    if d is None:  # non-inplace api
        d_dtype = None if dtype is None else _torchDtype2Str(dtype)
        return torch.ops.torch_mlu_ops.batch_matmul(a, b, c, bias, d_dtype, a_scale, b_scale,
                act_mode, alpha, beta, a_scale_scalar, b_scale_scalar, trans_a, trans_b,
                use_hp_active, approximate)
    else:
        torch.ops.torch_mlu_ops.batch_matmul_inplace(a, b, d, c, bias, a_scale, b_scale,
                act_mode, alpha, beta, a_scale_scalar, b_scale_scalar, trans_a, trans_b,
                use_hp_active, approximate)
        return d

def group_gemm(a: torch.Tensor,
               b: torch.Tensor,
               group_list: torch.Tensor,
               expand_idx: Optional[torch.Tensor],
               c: Optional[torch.Tensor],
               alpha: Optional[torch.Tensor],
               beta: Optional[torch.Tensor],
               max_in_group_list: int,
               bias: Optional[torch.Tensor] = None,
               d: Optional[torch.Tensor] = None,
               trans_a: Optional[bool] = False,
               trans_b: Optional[bool] = True,
               tile_config: Optional[Dict[str, int]] = None,
               idx_offset: Optional[torch.Tensor] = None,
               output_dtype: Optional[torch.dtype] = None
               ) -> torch.Tensor:
    '''
    Perform the grouped matrix multiplication operation.

    Math:
        if expand_idx:
            a = a[expand_idx]
        a_list = a.split(group_list)
        c_list = c.split(group_list)
        d_list = d.split(group_list)
        experts = len(group_list)
        for i in range(experts):
            d_list[i] = matmul(a_list[i], b[i]) * alpha[i] + beta[i] * c_list[i] + bias[i]
        d = concat(d_list, dim=0)

    Args:
        a (torch.Tensor): If trans_a, shape is (k, m), else shape is (m, k).
            If expand_idx exists, total_m = group_list.sum(), otherwise total_m = m.
        b (torch.Tensor): if trans_b, shape is (experts, n, k) or (n, k), else shape is (experts, k, n) or (k, n).
        group_list (torch.Tensor): Shape is (experts).
        expand_idx (torch.Tensor): optional. Shape is (total_m). Take effect only if b.dim() == 3.
        c (torch.Tensor): optional. Shape is (total_m, n) or (experts, n, k).
        alpha (torch.Tensor): optional. Shape is (experts).
        beta (torch.Tensor): optional. Shape is (experts).
        max_in_group_list (int): Maximum possible value in group_list.
        bias (torch.Tensor): optional. Shape is (experts, n).
        d (torch.Tensor): optional. Shape is (total_m, n) or (experts, n, k).
        trans_a (bool): Flag transpose of a, default is false.
        trans_b (bool): Flag transpose of b, default is true.
        tile_config (Dict[str, int]): optional. Tile config, including TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, WARP_SCHEDULER. eg. tile_config = {'TILE_SIZE_M': 32, 'TILE_SIZE_N': 256, 'TILE_SIZE_K': 128, 'WARP_SCHEDULER': 2}
        idx_offset (torch.Tensor): optional. Shape is (1).  The start position of expand_idx, used in expert parrallelism.
        output_dtype (torch.dtype). Specify the data type of output, must be torch.half, torch.bfloat16 or torch.float32. It only takes effect when d is None.

    Type:
        a: half, bfloat16, float.
        b: same as a.
        group_list: int32.
        expand_idx: int32.
        c: same as a.
        alpha: float.
        beta: float.
        d: same as a.
        bias: same as a.
        idx_offset: int32, int64.

    Return:
        Return d.

    Note:
        Supported combinations.
            a                b                 d                 trans_a       trans_b
        (m_list, k) * (experts, n, k)  -> (m_list, n)             False         True
        (k, m_list) * (experts, n, k)  -> (m_list, n)             True          True
        (m_list, k) * (experts, k, n)  -> (m_list, n)             False         False
        (k, m_list) * (experts, k, n)  -> (m_list, n)             True          False
        (m, k_list) * (n, k_list)      -> (experts, m, n)         False         True
        (k_list, m) * (n, k_list)      -> (experts, m, n)         True          True
        (m, k_list) * (k_list, n)      -> (experts, m, n)         False         False
        (k_list, m) * (k_list, n)      -> (experts, m, n)         True          False
    '''
    split_m = b.dim() != 2
    if d is None:
        d_dtype = a.dtype if output_dtype is None else output_dtype
        if split_m:
            tokens = a.size(1) if trans_a else a.size(0)
            n = b.size(1) if trans_b else b.size(2)
            total_m = tokens if expand_idx is None else expand_idx.size(0)
            d = torch.empty((total_m, n), dtype=d_dtype, device=a.device)
        else:
            tokens = a.size(1) if trans_a else a.size(0)
            n = b.size(0) if trans_b else b.size(1)
            d = torch.empty((group_list.size(0), tokens, n), dtype=d_dtype, device=a.device)
    allow_tf32 = False
    if torch.backends.mlu.matmul.fp32_precision == 'tf32' or \
            (torch.backends.mlu.matmul.fp32_precision == 'none' and torch.backends.fp32_precision == 'tf32'):
        allow_tf32 = True
    if tile_config is None:
        torch.ops.torch_mlu_ops.group_gemm_v2(a, b, group_list, d, expand_idx, c, alpha, beta, None, None,
                                        bias, None, None, None, None, max_in_group_list,
                                        trans_a, trans_b, -1, None, None, idx_offset, allow_tf32, True)
    else:
        torch.ops.torch_mlu_ops.group_gemm(a, b, group_list, d, expand_idx, c, alpha, beta, None, None,
                                        bias, None, None, None, None, tile_config, max_in_group_list,
                                        trans_a, trans_b, -1, None, None, idx_offset, allow_tf32, True)
    return d

def smooth_quant_group_gemm(a: torch.Tensor,
                            b: torch.Tensor,
                            m_list: torch.Tensor,
                            expand_idx: Optional[torch.Tensor],
                            c: Optional[torch.Tensor],
                            alpha: Optional[torch.Tensor],
                            beta: Optional[torch.Tensor],
                            a_scale: Optional[torch.Tensor],
                            b_scale: torch.Tensor,
                            dtype,
                            max_m: int,
                            bias: Optional[torch.Tensor] = None,
                            quant_flag: Optional[List[int]] = None,
                            d: Optional[torch.Tensor] = None,
                            tile_config: Optional[Dict[str, int]] = None,
                            a_calibration: Optional[torch.Tensor] = None,
                            b_calibration: Optional[torch.Tensor] = None,
                            a_quant_bit_size: int = 8,
                            a_lora: Optional[torch.Tensor] = None,
                            b_lora: Optional[torch.Tensor] = None,
                            idx_offset: Optional[torch.Tensor] = None,
                            is_symmetric_quant: bool = True
                            ) -> torch.Tensor:
    '''
    Perform smooth quantized group_gemm operation.

    Math:
        if expand_idx:
            a = a[expand_idx]
        a_list = a.split(m_list)
        c_list = c.split(m_list)
        d_list = d.split(m_list)
        a_scale_list = a_scale.split(m_list)
        experts = len(m_list)
        for i in range(experts):
            ab_scale = a_scale_list[i].outer(b_scale[i])
            ab = matmul(a_list[i], b[i]) * ab_scale
            d_list[i] = ab * alpha[i] + beta[i] * c_list[i] + bias[i]
        d = concat(d_list, dim=0)

    Args:
        a (torch.Tensor): Shape is (m, k). If expand_idx exists, total_m = m_list.sum(), otherwise total_m = m.
        b (torch.Tensor): Shape is (experts, n, k), or (experts, n, k // 2) for int 4 quantization.
        m_list (torch.Tensor): Shape is (experts).
        expand_idx (torch.Tensor): optional. Shape is (total_m).
        c (torch.Tensor): optional. Shape is (total_m, n).
        alpha (torch.Tensor): optional. Shape is (experts).
        beta (torch.Tensor): optional. Shape is (experts), the value can only be 0 or 1.
        a_scale (torch.Tensor): optional. Shape is (total_m) for smooth quantization,
            or (k_block, total_m) for a_groupwise-b_per_block quantization and MX format.
        b_scale (torch.Tensor): Shape is (experts, n) for smooth quantization,
            or (quant_group, experts, n) for groupwise quantization,
            or (experts, n_block, k_block) for a_groupwise-b_per_block quantization,
            or (experts, k_block, n) for b is MX format.
        dtype (torch.dtype). Specify the data type of output, must be torch.half or torch.bfloat16.
        max_m (int): Maximum possible value in m_list.
        bias (torch.Tensor): optional. Shape is (experts, n).
        quant_flag (List[int]): A list of int values used for groupwise quantization, the elements must be 4 or 8,
            which specify the quantization bits of each group,
            other values may cause undefined behavior. Length is (experts * quant_group).
        d (torch.Tensor): optional. Shape is (total_m, n).
        tile_config (Dict[str, int]): optional. Tile config, including TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, WARP_SCHEDULER. eg. tile_config = {'TILE_SIZE_M': 32, 'TILE_SIZE_N': 256, 'TILE_SIZE_K': 128, 'WARP_SCHEDULER': 2}
        a_calibration (torch.Tensor): optional. Shape is (total_m, 2) or (total_m, quant_group).
        b_calibration (torch.Tensor): optional. Shape is (experts * n, 2) or (experts * n, quant_group).
        a_quant_bit_size (int): Specifies the quantization bit size of a. When a_quant_bit_size = 8, a's shape is (total_m, k). When a_quant_bit_size = 4, a's shape is (total_m, k // 2).
        a_lora (torch.Tensor): optional. Shape is (total_m, lora_rank).
        b_lora (torch.Tensor): optional. Shape is (experts * n, lora_rank).
        idx_offset (torch.Tensor): optional. Shape is (1).  The start position of expand_idx, used in expert parallelism.
        is_symmetric_quant (bool): optional. Whether to use symmetric quantization when calibration data (a_calibration and b_calibration) is provided. Default is True.
            When True (symmetric), FlatQuant mode is used in W4A4 quantization mode, calibration dtype must be float32, and calibration shape is (total_m, 2) and (experts * n, 2) respectively.
            When False (asymmetric), asymmetric quantization is used in W4A8 quantization mode, calibration dtype must match output dtype, and calibration shape is (total_m, quant_group) and (experts * n, quant_group) respectively.

    Type:
        a: int8, int8(int4x2), float8, float4_e2m1fn_x2.
        b: int8, int8(int4x2), float8, float4_e2m1fn_x2.
        m_list: int32.
        expand_idx: int32.
        c: half, bfloat16.
        alpha: float.
        beta: float.
        a_scale: float, float8_e8m0fnu, bfloat16.
        b_scale: float, float8_e8m0fnu, bfloat16.
        d: half, bfloat16.
        bias: half, bfloat16.
        a_calibration: float, half, bfloat16.
        b_calibration: float, half, bfloat16.
        a_lora: half, bfloat16.
        b_lora: half, bfloat16.
        idx_offset: int32, int64.

    Return:
        Return d.

    Note:
        The elements in quant_flag must be 4 or 8.
        a_quant_bit_size must be 4 if flatquant(both a_calibration and b_calibration exist) or svdquant(both a_lora and b_lora exist) mode.
        b must be transposed.
        k_block * k_block_size must be equal to k, and n_block * n_block_size must be equal to n in a_groupwise-b_per_block quantization mode.
        a_groupwise-b_per_block quantization mode only support float8 data type.
        When a and b are both MX format, group_size only supports 32.
        When MX quantization mode, a_scale and b_scale data type is float8_e8m0fnu or bfloat16.

    '''
    if d is None:
        tokens = a.size(0)
        n = b.size(1) if quant_flag is None else b_scale.size(2)
        total_m = tokens if expand_idx is None else expand_idx.size(0)
        d = torch.empty((total_m, n), dtype=dtype, device=a.device)

    allow_tf32 = False
    if torch.backends.mlu.matmul.fp32_precision == 'tf32' or \
            (torch.backends.mlu.matmul.fp32_precision == 'none' and torch.backends.fp32_precision == 'tf32'):
        allow_tf32 = True
    if tile_config is None:
        quant_flag = None if quant_flag is None else torch.tensor(quant_flag)
        torch.ops.torch_mlu_ops.group_gemm_v2(a, b, m_list, d, expand_idx, c, alpha, beta, a_scale,
                                        b_scale, bias, a_calibration, b_calibration, quant_flag,
                                        None, max_m, False, True, a_quant_bit_size,
                                        a_lora, b_lora, idx_offset, allow_tf32, is_symmetric_quant)
    else:
        torch.ops.torch_mlu_ops.group_gemm(a, b, m_list, d, expand_idx, c, alpha, beta, a_scale,
                                        b_scale, bias, a_calibration, b_calibration, quant_flag,
                                        None, tile_config, max_m, False, True, a_quant_bit_size,
                                        a_lora, b_lora, idx_offset, allow_tf32, is_symmetric_quant)
    return d

def preload(
    weight: torch.Tensor,
    size: int,
) -> None:
    torch.ops.torch_mlu_ops.preload(weight, size)
    return weight

def moe_cast_gating(input: torch.Tensor,
                    weight: torch.Tensor) -> torch.Tensor:
    """
    Cast input data type to torch.float32, and perform gating operation.

    Math:
        input_fp32 = input.to(torch.float32).
        gating_output = F.Linear(input_fp32, weight).

    Args:
        input (torch.Tensor): The input tensor, shape is (..., hidden_size), hidden_size must be less than or equal to 16384. The tensor must be continuous between 0 and -2 dimensions.
        weight (torch.Tensor): The input tensor, shape is (expert_num, hidden_size), expert_num must be less than or equal to 512.

    Types:
        input: half, bfloat16.
        weight: float.

    Return:
        return a float tensor with shape [..., expert_num].
    """

    return torch.ops.torch_mlu_ops.moe_cast_gating(input, weight)

def moe_cast_gating_v2(input: torch.Tensor,
                       weight0: torch.Tensor,
                       weight1: torch.Tensor,
                       alpha: float) -> torch.Tensor:
    """
    Performs a high-performance fused gating operation using bfloat16 inputs.

    This function utilizes an optimized kernel that performs a fused computation,
    offering significant performance gains over the standard version(v1). All input
    tensors (`input`, `weight0`, `weight1`) must be bfloat16.

    Math:
        `output = torch.matmul(input, weight0.T) + alpha * torch.matmul(input, weight1.T)`

    Args:
        input (torch.Tensor):  The input tensor, shape is (..., hidden_size), hidden_size must be less than or equal to 16384. The tensor must be continuous between 0 and -2 dimensions.
        weight0 (torch.Tensor): The input tensor, shape is (expert_num, hidden_size), expert_num must be less than or equal to 512.
        weight1 (torch.Tensor): The input tensor, shape is (expert_num, hidden_size), expert_num must be less than or equal to 512.
        alpha (float): A scaling factor applied to the second matrix multiplication result.

    Types:
        input: bfloat16.
        weight0: bfloat16.
        weight1: bfloat16.
        alpha: float.

    Return:
        return a float tensor with shape [..., expert_num].
    """

    return torch.ops.torch_mlu_ops.moe_cast_gating_v2(input, weight0, weight1, alpha)

def moe_softmax_topk(input: torch.Tensor,
                     topk: int,
                     normalize: bool = False,
                     num_expert_group: int = -1,
                     topk_group: int = 0,
                     mask: Optional[torch.Tensor] = None,
                     normed_by : str = "topk_logit",
                     route_scale : float = 1.0,
                     reduce_weight: Optional[torch.Tensor] = None,
                     expert_id: Optional[torch.Tensor] = None,
                     score_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
    """
    Apply grouped softmax-topk to input tensor.

    Math:
        num_expert = input.shape[-1]
        softmax = torch.softmax(input.float(), dim=-1)
        out_shape = list(input.size())[:-1] + [topk]
        if origin_reduce_weight is None:
            origin_reduce_weight = torch.empty(out_shape, dtype=torch.float32, device=input.device)
        if origin_expert_id is None:
            origin_expert_id = torch.empty(out_shape, dtype=torch.int32, device=input.device)
        if num_expert_group <= 1:
            if origin_mask is not None:
                softmax = softmax * origin_mask
            original_softmax = softmax
            if score_bias is not None:
                softmax = softmax.view(-1, num_expert) + score_bias.unsqueeze(0)
                expert_id = torch.topk(softmax, k=topk, dim=-1)[1]
                reduce_weight = original_softmax.gather(-1, expert_id)
            else:
                reduce_weight, expert_id = torch.topk(softmax, k=topk, dim=-1)
            if normalize:
                if normed_by == "topk_logit":
                    reduce_weight = reduce_weight / reduce_weight.sum(dim=-1, keepdim=True)
                if normed_by == "softmax_logit":
                    reduce_weight = reduce_weight / softmax.sum(dim=-1, keepdim=True)
            reduce_weight = reduce_weight * route_scale
            origin_reduce_weight.copy_(reduce_weight)
            origin_expert_id.copy_(expert_id)
            return origin_reduce_weight, origin_expert_id
        else:
            group_size = softmax.shape[-1] // num_expert_group
            new_shape = softmax.shape[:-1] + (num_expert_group, group_size)
            original_softmax = softmax
            if score_bias is not None:
                softmax = softmax.view(-1, num_expert) + score_bias.unsqueeze(0)
            group_data = softmax.view(new_shape)
            group_max_value = group_data.max(dim=-1).values
            group_idx = torch.topk(group_max_value, k=topk_group, dim=-1)[1]
            mask_shape = softmax.shape[:-1] + (num_expert_group,)
            mask = torch.zeros((mask_shape), dtype = torch.bool, device = group_idx.device)
            mask.scatter_(-1, group_idx, True)
            mask = mask.unsqueeze(-1).expand(new_shape)
            masked_data = group_data.masked_fill(~mask, 0.0)
            masked_data = masked_data.reshape(softmax.shape)
            if score_bias is not None:
                expert_id = torch.topk(masked_data, k=topk, dim=-1)[1]
                reduce_weight = original_softmax.gather(-1, expert_id)
            else:
                reduce_weight, expert_id = torch.topk(masked_data, k=topk, dim=-1)
            if normalize:
                if normed_by == "topk_logit":
                    reduce_weight = reduce_weight / reduce_weight.sum(dim=-1, keepdim=True)
                if normed_by == "softmax_logit":
                    reduce_weight = reduce_weight / softmax.sum(dim=-1, keepdim=True)
            reduce_weight = reduce_weight * route_scale
            origin_reduce_weight.copy_(reduce_weight)
            origin_expert_id.copy_(expert_id)
            return origin_reduce_weight, origin_expert_id

    Args:
        input (torch.Tensor): The input tensor, shape is (..., expert_num).
        topk (int): The number of experts that each token would be dispatched to.
        normalize (bool): If do normalization to the reduce_weight after topk operation.
        num_expert_group(int): The number of groups that each expert would be grouped to.
        topk_group(int): The number of groups to be selected from num_expert_group.
        mask (torch.Tensor): The mask multiplied to the softmax result. The dimension must be the same as input.
            The last two dims must be the same as input and the other dims must be 1.
        normed_by (str): The mode of normalization, which can be "topk_logit" or "softmax_logit"
        route_scale(float): The number multiplied to the reduce_weight result.
        score_bias (torch.Tensor, optional): The score bias tensor added to the softmax result before computing expert_id.
        reduce_weight(torch.Tensor): Output reduce_weight tensor, shape is (..., topk).
        expert_id(torch.Tensor): Output expert_id tensor, shape is (..., topk).

    Type:
        input: float, half, bfloat16.
        mask: same as input
        reduce_weight: float
        expert_id : int32

    Return:
        Return the reduce_weight and expert_id tensor.

    Note:
        Input and mask must be contiguous.
    """
    out_shape = input.shape[:-1] + (topk,)
    if reduce_weight is None:
        reduce_weight = torch.empty(out_shape, dtype=torch.float32, device=input.device)
    if expert_id is None:
        expert_id = torch.empty(out_shape, dtype=torch.int32, device=input.device)
    torch.ops.torch_mlu_ops.moe_active_topk(input,
                                            topk,
                                            num_expert_group,
                                            topk_group,
                                            normalize,
                                            mask,
                                            normed_by,
                                            "softmax",
                                            route_scale,
                                            score_bias,
                                            reduce_weight,
                                            expert_id)
    return (reduce_weight, expert_id)

def moe_sigmoid_topk(input: torch.Tensor,
                     topk: int,
                     normalize: bool = False,
                     num_expert_group: int = -1,
                     topk_group: int = 0,
                     mask: Optional[torch.Tensor] = None,
                     normed_by : str = "topk_logit",
                     route_scale : float = 1.0,
                     score_bias: Optional[torch.Tensor] = None,
                     reduce_weight: Optional[torch.Tensor] = None,
                     expert_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
    """
    Apply grouped sigmoid-topk to input tensor.

    Math:
        sigmoid = torch.sigmoid(input.float())
        out_shape = list(input.size())[:-1] + [topk]
        if origin_reduce_weight is None:
            origin_reduce_weight = torch.empty(out_shape, dtype=torch.float32, device=input.device)
        if origin_expert_id is None:
            origin_expert_id = torch.empty(out_shape, dtype=torch.int32, device=input.device)
        if num_expert_group <= 1:
            if origin_mask is not None:
                sigmoid = sigmoid * origin_mask
            original_sigmoid = sigmoid
            if score_bias is not None:
                sigmoid = sigmoid + score_bias.unsqueeze(0)
                expert_id = torch.topk(sigmoid, k=topk, dim=-1)[1]
                reduce_weight = original_sigmoid.gather(-1, expert_id)
            else:
                reduce_weight, expert_id = torch.topk(sigmoid, k=topk, dim=-1)
            if normalize:
                if normed_by == "topk_logit":
                    reduce_weight = reduce_weight / reduce_weight.sum(dim=-1, keepdim=True)
                if normed_by == "sigmoid_logit":
                    reduce_weight = reduce_weight / sigmoid.sum(dim=-1, keepdim=True)
            reduce_weight = reduce_weight * route_scale
            origin_reduce_weight.copy_(reduce_weight)
            origin_expert_id.copy_(expert_id)
            return origin_reduce_weight, origin_expert_id
        else:
            group_size = sigmoid.shape[-1] // num_expert_group
            new_shape = sigmoid.shape[:-1] + (num_expert_group, group_size)
            original_sigmoid = sigmoid
            if score_bias is not None:
                sigmoid = sigmoid + score_bias.unsqueeze(0)
                group_data = sigmoid.view(new_shape)
                group_max_value = group_data.topk(2, dim=-1)[0].sum(dim=-1)
            else:
                group_data = sigmoid.view(new_shape)
                group_max_value = group_data.max(dim=-1).values
            group_idx = torch.topk(group_max_value, k=topk_group, dim=-1)[1]
            mask_shape = sigmoid.shape[:-1] + (num_expert_group,)
            mask = torch.zeros((mask_shape), dtype = torch.bool, device = group_idx.device)
            mask.scatter_(-1, group_idx, True)
            mask = mask.unsqueeze(-1).expand(new_shape)
            masked_data = group_data.masked_fill(~mask, float("-inf"))
            masked_data = masked_data.reshape(sigmoid.shape)
            if score_bias is not None:
                expert_id = torch.topk(masked_data, k=topk, dim=-1)[1]
                reduce_weight = original_sigmoid.gather(-1, expert_id)
            else:
                reduce_weight, expert_id = torch.topk(masked_data, k=topk, dim=-1)
            if normalize:
                if normed_by == "topk_logit":
                    reduce_weight = reduce_weight / reduce_weight.sum(dim=-1, keepdim=True)
                if normed_by == "sigmoid_logit":
                    reduce_weight = reduce_weight / original_sigmoid.sum(dim=-1, keepdim=True)
            reduce_weight = reduce_weight * route_scale
            origin_reduce_weight.copy_(reduce_weight)
            origin_expert_id.copy_(expert_id)
            return origin_reduce_weight, origin_expert_id

    Args:
        input (torch.Tensor): The input tensor, shape is (..., expert_num).
        topk (int): The number of experts that each token would be dispatched to.
        normalize (bool): If do normalization to the reduce_weight after topk operation.
        num_expert_group(int): The number of groups that each expert would be grouped to.
        topk_group(int): The number of groups to be selected from num_expert_group.
        mask (torch.Tensor): The mask multiplied to the sigmoid result. The dimension must be the same as input.
           The last two dims must be the same as input and the other dims must be 1.
        normed_by (str): The mode of normalization, which can be "topk_logit" or "sigmoid_logit"
        route_scale(float): The number multiplied to the reduce_weight result.
        score_bias (torch.tensor): The tensor added to the sigmoid output. Shape is (..., expert_num), the dimension must be the same as input.
        reduce_weight(torch.Tensor): Output reduce_weight tensor, shape is (..., topk).
        expert_id(torch.Tensor): Output expert_id tensor, shape is (..., topk).

    Type:
        input: float, half, bfloat16.
        mask: same as input
        score_bias: float, half, bfloat16.

    Return:
        Return the reduce_weight and expert_id tensor.

    Note:
        Input and mask must be contiguous.
    """
    out_shape = input.shape[:-1] + (topk,)
    if reduce_weight is None:
        reduce_weight = torch.empty(out_shape, dtype=torch.float32, device=input.device)
    if expert_id is None:
        expert_id = torch.empty(out_shape, dtype=torch.int32, device=input.device)
    torch.ops.torch_mlu_ops.moe_active_topk(input,
                                            topk,
                                            num_expert_group,
                                            topk_group,
                                            normalize,
                                            mask,
                                            normed_by,
                                            "sigmoid",
                                            route_scale,
                                            score_bias,
                                            reduce_weight,
                                            expert_id)
    return (reduce_weight, expert_id)

def moe_softplus_topk(input: torch.Tensor,
                      topk: int,
                      input_ids: Optional[torch.Tensor] = None,
                      tid2eid: Optional[torch.Tensor] = None,
                      bias: Optional[torch.Tensor] = None,
                      route_scale: float = 1.0,
                      reduce_weight: Optional[torch.Tensor] = None,
                      expert_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
    """
    Apply softplus-topk to input tensor.

    Math:
        scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices

    Args:
        input(torch.Tensor): The input tensor, shape is (..., num_expert).
        topk(int): The number of experts that each token would be dispatched to.
        input_ids(torch.Tensor): Input id tensor, shape = input.shape[:-1].
        tid2eid(torch.Tensor): Tid2eid table, shape is (vocab_size, topk).
        bias(torch.Tensor): Bias tensor, shape is (num_expert).
        route_scale(float): The number multiplied to the reduce_weight result.
        reduce_weight(torch.Tensor): Output reduce_weight tensor.
        expert_id(torch.Tensor): Output expert_id tensor.

    Type:
        input: float, half, bfloat16.
        bias: float, half, bfloat16.
        input_ids: int32.
        tid2eid: int32.

    Return:
        Return the reduce_weight and expert_id tensor.

    Note:
        1. This is an inference operator, nan/inf behavior is undefined.
        2. If tid2eid is provided, input_ids must also be provided.
        3. Input must be contiguous.
    """
    out_shape = list(input.size())[:-1] + [topk]
    if reduce_weight is None:
        reduce_weight = torch.empty(out_shape, dtype=torch.float32, device=input.device)
    if expert_id is None:
        expert_id = torch.empty(out_shape, dtype=torch.int32, device=input.device)
    torch.ops.torch_mlu_ops.moe_softplus_topk(input,
                                              input_ids,
                                              tid2eid,
                                              bias,
                                              topk,
                                              route_scale,
                                              reduce_weight,
                                              expert_id)
    return (reduce_weight, expert_id)


def moe_append_shared_expert(reduce_weight: torch.Tensor,
                             expert_id: torch.Tensor,
                             num_expert: int,
                             shared_expert_num: int,
                             world_size: int,
                             parallel_mode: str = 'ep') -> Tuple[torch.Tensor]:
    """
    Recalculate reduce_weight and expert_id when append shared_expert into moe.

    Math:
        Please refer to 'test/pytest/test_moe_append_shared_expert.py' in this repository.

    Args:
        reduce_weight (torch.Tensor): Represents the weight coefficients of the experts corresponding to each token. Shape is (..., topk).
        expert_id (torch.Tensor): Represents the position indices of the experts corresponding to each token. Shape is (..., topk).
        num_expert (int): The number of experts.
        shared_expert_num (int): The number of shared experts.
        world_size (int): The number of tensor parallel or expert parallel.
        parallel_mode (str): The type of parallel mode. It should be set to 'ep' or 'tp'.

    Type:
        reduce_weight: float.
        expert_id: int.

    Return:
        Return the new reduce_weight and expert_id.

    Note:
        Topk must less than or equal than num_expert; world_size must be divisible by shared_expert_num.
    """
    outputs = torch.ops.torch_mlu_ops.moe_append_shared_expert(reduce_weight,
                                                               expert_id,
                                                               num_expert,
                                                               shared_expert_num,
                                                               world_size,
                                                               parallel_mode)
    return tuple(outputs)

def moe_expand_input(input: torch.Tensor,
                     gather_idx: torch.Tensor,
                     cusum_token_count: Optional[torch.Tensor] = None,
                     start_expert_id: int = 0,
                     expert_size: int = 0,
                     output: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Expands the input tensor with the indices specified by gather_idx.

    Math:
        if cusum_token_count is None:
            output = input[gather_idx]
        else:
            idx = gather_idx[cusum_token_count[start_expert_id]:cusum_token_count[start_expert_id+expert_count]]
            output = input[idx]

    Args:
        input (torch.Tensor): The input tensor, shape is (token_num, hidden_size).
        gather_idx (torch.Tensor): The input tensor containing the indices to index. Shape is (expand_token_num), where expand_token_num = token_num * topk (topk is the number of experts selected for each token).
        cusum_token_count (torch.Tensor): The input tensor storing the prefix sum of token count. Shape is (expert_num + 1).
        output (torch.Tensor): The output tensor storing expanded input. Shape is (expand_token_num, hidden_size), where expand_token_num = token_num * topk (topk is the number of experts selected for each token).
        start_expert_id (int): the first expert id.
        expert_size (int): the number of experts currently being processed.

    Types:
        input: int8, float, half, bfloat16.
        gather_idx: int32.
        cusum_token_count: int32.
        output: int8, float, half, bfloat16.
        start_expert_id: int32.
        expert_size: int32.

    Return:
        return the input selected with gather idx.

    Note:
        output_token must be equal to gather_idx.size(0) in tensor parallelism,
        otherwise large than or be equal to min(input.size(0) * expert_size, gather_idx.size(0)) in expert parallelism.
        output_stride * sizeof(dtype) must not exceed UINT32_MAX, otherwise the result is undefined.
    """
    if output is None:
        return torch.ops.torch_mlu_ops.moe_expand_input(input, gather_idx, cusum_token_count,
                                                        start_expert_id, expert_size)
    else:
        torch.ops.torch_mlu_ops.moe_expand_input_inplace(input, gather_idx, cusum_token_count,
                                                        start_expert_id, expert_size, output)
        return output

def moe_gen_idx(expert_id: torch.Tensor,
                expert_num: int,
                return_token2expert_idx: bool = False) -> Tuple[torch.Tensor]:
    """
    Generate expand_idx, combine_idx, token_count, and cusum_token_count, and optionally token2expert_idx.

    Args:
        expert_id (torch.Tensor): The input tensor stores the expert id of each token, the shape must be [token_num, topk]. Values must be in range [0, expert_num).
        expert_num: the number of expert.
        return_token2expert_idx (bool, optional): Whether to return the sorted indices not divided by topk. Defaults to False.

    Types:
        expert_id: int32.
        expert_num: int32.
        return_token2expert_idx: bool.

    Return:
        If return_token2expert_idx is False, return (expand_idx, combine_idx, token_count, cusum_token_count).
        If return_token2expert_idx is True, return (expand_idx, combine_idx, token_count, cusum_token_count, token2expert_idx).
    """

    outputs = torch.ops.torch_mlu_ops.moe_gen_idx(expert_id, expert_num, return_token2expert_idx)
    return outputs

def moe_combine_result(input: torch.Tensor,
                       reduce_weight: torch.Tensor,
                       gather_ids: torch.Tensor,
                       residual: Optional[torch.Tensor],
                       cusum_token_count: Optional[torch.Tensor],
                       start_expert_id: int,
                       expert_size: int,
                       bias: Optional[torch.Tensor] = None,
                       output: Optional[torch.Tensor] = None) -> torch.Tensor:
    '''
    Perform combine result operation in moe.

    Math:
        if cusum_token_count:
            input = input[gather_ids - cusum_token_count[start_expert_id + expert_size]]
        else:
            input = input[gather_ids]

        if bias:
            for i in range(start_expert_id : start_expert_id + expert_size):
                output[cusum_token_count[i] : cusum_token_count[i+1]] += bias[i]

        if cusum_token_count:
            reduce_weight *= (gather_ids >= cusum_token_count[start_expert_id]) *
                             (gather_ids < cusum_token_count[start_expert_id + expert_size])
        output = input * reduce_weight
        output = output.sum(1)
        if residual:
            output += residual

    Args:
        input (torch.Tensor): Shape is (num_token * topk, hidden_size), where num_tokens = num_token * topk (topk is the number of experts selected for each token).
        reduce_weight (torch.Tensor): Shape is (num_token, topk).
        gather_ids (torch.Tensor): Shape is (num_token * topk).
        residual (torch.Tensor): optional. Shape is (num_token, hidden_size).
        bias (torch.Tensor): optional. Shape is (num_expert, hidden_size). Do not support yet, must be None.
        cusum_token_count (torch.Tensor): optional. Shape is (num_expert + 1).
        start_expert_id (int): begin expert, used in expert parrallelism.
        expert_size (int): expert size, used in expert parrallelism.
        output (torch.Tensor): optional. Shape is (num_token, hidden_size).

    Type:
        input: bfloat16, float16, float32.
        reduce_weight: float32.
        gather_ids: int32.
        residual: bfloat16, float16, float32.
        bias: bfloat16, float16, float32.
        cusum_token_count: int32.
        start_expert_id: int32.
        expert_size: int32.
        output: same as input.

    Return:
        Return output.
    '''
    if output is None:
        output = torch.empty((reduce_weight.size(0), input.size(1)), dtype=input.dtype, device=input.device)
    torch.ops.torch_mlu_ops.moe_combine_result(input,
                                               output,
                                               reduce_weight,
                                               gather_ids,
                                               residual,
                                               cusum_token_count,
                                               start_expert_id,
                                               expert_size,
                                               bias)
    return output

def moe_all2all_gen_send_layout(token_count: torch.Tensor,
                                nrank: int) -> torch.Tensor:
    """
    Compute the start index and token number of each rank in the all2all distributed strategy of MoE network.

    Math:
        expert_num = token_count.size(0)
        expert_num_rds = token_count.reshape(nrank, expert_num//nrank).sum(dim=-1)
        expert_num_rds_cusum = torch.cat((torch.tensor([0]).to('mlu'), torch.cumsum(expert_num_rds[:-1], dim=0)))
        layout = torch.cat((expert_num_rds_cusum.unsqueeze(1), expert_num_rds.unsqueeze(1)), dim=-1)

    Args:
        token_count (torch.Tensor): Shape is [expert_num].
        nrank (int): num of ranks.

    Type:
        token_count: int32.
        nrank: int32.
        output: int32.

    Return:
        A mlu tensor, shape is [nrank, 2].

    """
    return torch.ops.torch_mlu_ops.moe_all2all_gen_send_layout(token_count, nrank)

def moe_quantize(x: torch.Tensor,
                 smooth: torch.Tensor,
                 zero: Optional[torch.Tensor] = None,
                 token_count: Optional[torch.Tensor] = None,
                 gather_index: Optional[torch.Tensor] = None,
                 gather_index_start_position: Optional[torch.Tensor] = None,
                 output: Optional[torch.Tensor] = None,
                 output_scale: Optional[torch.Tensor] = None,
                 dynamic_quant: bool = True,
                 act_mode: str = "none",
                 active_coef: float = 1.0,
                 is_gated: bool = False,
                 quant_type: torch.dtype = torch.int8,
                 quant_bit_size = 8,
                 scale_type = torch.float32,
                 need_output_scale_trans: bool = False,
                 output_reduced: Optional[torch.Tensor] = None,
                 group_size: int = 1) -> Tuple[torch.Tensor]:
    """
    Apply quantization to the input tensor in MoE network.

    Math:
        if act_mode != "none":
            if is_gated:
                input = act_func(input[..., :C//2]) * input[..., C//2:]
            else:
                input = act_func(input)
        if gather_idx and gather_index_start_position:
            start_pos = gather_index_start_position[0]
            input = input[gather_index[start_pos : start_pos + token_count.sum()]]
        elif gather_idx:
            input = input[gather_index]

        if token_count:
            input_list = input.split(token_count)
            group = token_count.size()
            result = []
            for i in range(group):
                result.append(input_list[i] * smooth[i])
            smoothed = concat(result, dim=0)
        else:
            smoothed = input * smooth

        if dynamic_quant:
            max, _ = smoothed.abs().max(dim=-1, keepdim=True)
            output_scale = max.to(torch.float) / 127.0
            quanted = (smoothed / output_scale).round().clamp(-128, 127).to(torch.int8)
            output = (quanted, output_scale)
        else:
            quanted = smoothed.round().clamp(-128, 127).to(torch.int8)
            output = (quanted)

    Args:
        x (torch.Tensor): The tensor to be quantized. Shape is (..., C) or (token_count.sum(), C). The tensor must be continuous between 0 and -2 dimensions.
        smooth (torch.Tensor): The smooth scale multipled to the input tensor. If is_gated is true, Shape is (C//2) or(group, C//2),
                               otherwise shape is (C) or (group, C).
        zero (torch.Tensor): Not supported, must pass None.
        token_count (torch.Tensor): The tensor to separate input. Shape is (group).
        gather_index (torch.Tensor): The indices tensor to expand input. Shape is (expanded_tokens_num).
        gather_index_start_position (torch.Tensor): The tensor indicating the start position of gather_index. Shape is (1).
        output: (torch.Tensor): The tensor to store the output.
        output_scale: (torch.Tensor): The tensor to store output_scale.
        dynamic_quant: whether do dynamic quant.
        act_mode: The mode of activation, must be "none", "gelu", "silu", "swish".
        active_coef: The coefficient used in the swish activation. Default is 1.0.
        is_gated: A boolean parameter that indicates whether a gating mechanism is applied. It only
                         takes effect when act_mode is not "none".
        quant_type: Quant dtype. Support int8 and float8_e4m3fn, float8_e5m2, float4_e2m1fn_x2.
        quant_bit_size: Quantization bit width. Setting to 4 with int8 dtype enables int4x2 packing (two int4 in one int8).
        scale_type: Specifies the quantization scaling method. Set to 'float8_e8m0' or 'torch.bfloat16' in dynamic_quant mode to enable mx Quantization.
        need_output_scale_trans: A boolean parameter that indicates whether output scale transformation should be applied.
                                 It only effective in mx Quantization.
        output_reduced: An optional tensor to store the reduced output scale. Shape is (..., C // group_size),
                        or (..., C // (2 * group_size)) when is_gated.
        group_size: An integer parameter that indicates the group size for quantization. It only effective when output_reduced is not None,
                    group_size must be 128, 256, 512, 1024.
    DataType:
        x: float, half or bfloat16.
        smooth: float.
        token_count: int32.
        gather_index: int32.
        gather_index_start_position: int32.
        dynamic_quant: bool.
        act_mode: str.
        active_coef: float.
        is_gated: bool.
        output: int8, float8_e4m3fn or float8_e5m2, float4_e2m1fn_x2.
        output_scale: float, float8_e8m0fnu or bfloat16.
        output_reduced: float, half, bfloat16.
        group_size: int

    Return:
        Returns (output, output_scale) if dynamic_quant is True, otherwise returns output only.
    """
    # moe_quantize only consider dynamic_per_token and static_per_channel condition,
    # considering scale of flaot, [float8_e8m0fnu and bfloat16]
    tokens, ci = x.shape[0], x.shape[-1]
    if gather_index is not None:
        tokens = gather_index.size(0)
    if quant_type == torch.int8 and quant_bit_size == 4:
        ci = ci // 2 // (1 + is_gated)
    elif quant_type is torch.float4_e2m1fn_x2:
        ci = ci // 2 // (1 + is_gated)
    else :
        ci = ci // (1 + is_gated)

    output_shape = (tokens, ) + x.shape[1:-1] + (ci, )
    output_shape = output_shape[:x.dim()] # avioding 1-dim input

    if scale_type == torch.float32:
        output_scale_shape = (tokens,) + x.shape[1:-1]
        output_scale_shape = output_scale_shape[:x.dim()-1]
    elif scale_type in (torch.float8_e8m0fnu, torch.bfloat16):
        scale_factor = 2 if quant_type == torch.float4_e2m1fn_x2 else 1
        scale_elements = ci * scale_factor // 32

        output_scale_shape = (tokens,) + x.shape[1:-1] + (scale_elements,)

        if need_output_scale_trans:
            output_scale_shape = (scale_elements,) + (tokens,) + x.shape[1:-1]

    output = torch.empty(output_shape, dtype=quant_type, device=x.device) if output is None else output
    output_scale = torch.empty(output_scale_shape, dtype=scale_type, device=x.device) if output_scale is None and dynamic_quant else output_scale
    torch.ops.torch_mlu_ops.scaled_quantize(x, output, output_scale, smooth, zero, token_count,
                                            gather_index, gather_index_start_position, None,
                                            'dynamic_per_token' if dynamic_quant else 'static_per_channel',
                                            act_mode, active_coef, is_gated, quant_bit_size, need_output_scale_trans, output_reduced, group_size)
    return (output, output_scale) if dynamic_quant else (output,)

def moe_active(input: torch.Tensor,
               act_mode: str,
               is_gated: bool,
               output: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None,
               cusum_token_count: Optional[torch.Tensor] = None,
               start_expert_id: int = 0,
               expert_size: int = 0,
               high_precision: bool = False,
               gelu_approximate: str = 'none',
               swiglu_limit: int = 0,
               weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Apply activation to the input tensor.

    Math:
        C = input.shape[-1]
        if bias:
            start_idx = 0
            for i in range(start_expert_id : start_expert_id + expert_size):
                deal_token_num = cusum_token_count[i+1] - cusum_token_count[i]
                input[start_idx:start_idx+deal_token_num] += bias[i]
                start_idx += deal_token_num
        if is_gated:
            if swiglu_limit > 0:
                input[..., :C//2] = clamp(max = swiglu_limit)
                input[..., C//2:] = clamp(min = -swiglu_limit, max = swiglu_limit)
            output = active(input[..., :C//2]) * input[..., C//2:]
        else:
            if swiglu_limit > 0:
                input = clamp(max = swiglu_limit)
            output = active(input)
        if weight:
            output = output * weight

    Args:
        input (torch.Tensor): Shape is (..., C).
        act_mode (str): The activation mode, must be 'silu' or 'gelu'.
        is_gated (bool): If use gated activation.
        bias(torch.Tensor): optional, Shape is (expert_num, C).
        cusum_token_count(torch.Tensor): optional, Shape is (expert_num + 1).
        start_expert_id(int): optional, begin expert, used in expert parallelism.
        expert_size(int): optional, expert_size, used in expert parallelism.
        high_precision (bool): If adopt high precision calculation. Default is false.
        gelu_approximate (str): If use tanh_gelu. Can be 'none' or 'tanh'. Default is 'none'.
        swiglu_limit(int): If larger than 0, need clamp before active. Default is 0. The value must smaller than 256.
        weight(torch.Tensor): optional, if not None, need mul(weight) after active.

    Type:
        input: float, half, bfloat16.
        act_mode: str.
        is_gated: bool.
        bias: same as input.
        cusum_token_count: int32.
        start_expert_id: int32.
        expert_size: int32.
        output: same as input.
        weight: float.

    Return:
        if is_gated, output shape is (input.size()[:-1], C // 2),
        else, the same as input shape.
    """
    if output is None:
        output = torch.empty(input.size()[:-1] + (input.size()[-1] // (1+is_gated),), dtype=input.dtype, device=input.device)
    torch.ops.torch_mlu_ops.active(input,
                                   output,
                                   bias,
                                   cusum_token_count,
                                   act_mode,
                                   is_gated,
                                   start_expert_id,
                                   expert_size,
                                   1.0,
                                   high_precision,
                                   gelu_approximate,
                                   swiglu_limit,
                                   weight)
    return output


def moe_svd_quantize(input: torch.Tensor,
                     weight_lora_down: torch.Tensor,
                     smooth: torch.Tensor,
                     lora_scales: torch.Tensor,
                     quant_mode: int,
                     asym_quant: bool,
                     quant_type: str,
                     m_list: Optional[torch.Tensor] = None,
                     gather_index: Optional[torch.Tensor] = None,
                     gather_index_start_position: Optional[torch.Tensor] = None,
                     act_mode: str = "none",
                     active_coef: float = 1.0,
                     is_gated: bool = False,
                     output_lora_down: torch.Tensor = None,
                     output_quant: torch.Tensor = None,
                     output_quant_scales: torch.Tensor = None):
    """
    Perform svd quant operation(ABSORBING OUTLIERS BY LOW-RANK
        COMPONENTS FOR 4-BIT DIFFUSION MODELS).
        For details,https://arxiv.org/abs/2411.05007.

    Math:
        inputs-parameters:
            input: shape[token_num, hidden_size]
            weight_lora_down: shape[expert_num, lora_rank, hidden_size] if is_gated is
                false, else shape[expert_num, lora_rank, hidden_size / 2]
            smooth: shape[expert_num, hidden_size] if is_gated is false,
                else shape[expert_num, hidden_size / 2]
            lora_scales: shape[expert_num, lora_rank]
            m_list: shape[expert_num]
            gather_index: shape(expanded_tokens_num).
            gather_index_start_position: shape(1).
        outputs-parameters:
            output_lora_down: shape[token_num, lora_rank] if gather_index is None,
                shape[expanded_tokens_num, lora_rank] if gather_index not None
            output_quant: shape[token_num, hidden_size / 2] if gather_index is None and is_gated false,
                shape[expanded_tokens_num, hidden_size / 2] if gather_index not None and is_gated false,
                shape[expanded_tokens_num, hidden_size / 4] if gather_index not None and is_gated  true,
                shape[token_num, hidden_size / 4] if gather_index is None and is_gated  true.
            output_quant_scales:
                shape[token_num] if gather_index is None,
                shape[expanded_tokens_num] if gather_index not None
        process:
            input = active(input)
            input = gather(input)
            for i in range(experts):
                output_lora_down_list[i] = matmul(input, weight_lora_down)
            output_loradown = concat(output_loradown_list, dim=0)
            output_quant, output_quant_scales = dynamic_pertoken_quantize(input)

    Args:
        input (torch.Tensor): The input tensor. Shape is (token_num, hidden_size).
        weight_lora_down(torch.Tensor): The weight of lora-down projection matmul.
            Shape is (expert_num, lora_rank, hidden_size) if is_gated false, else
            shape is (expert_num, lora_rank, hidden_size / 2).
        smooth(torch.Tensor): The input smooth factor for quantization process.
            Shape is (expert_num, hidden_size) if is_gated false, else
            shape is (expert_num, hidden_size / 2).
        lora_scales(torch.Tensor): The factor of lora-down projection output.
            Shape is (expert_num, lora_rank).
        m_list(torch.Tensor): The token number of per expert-quantified group.
            Shape is (expert_num), m_list.sum() is expanded_tokens_num.
        gather_index (torch.Tensor): The indices tensor to expand input.
            Shape is (expanded_tokens_num) if gather_index_start_position is null.
            Shape is (>= gather_index_start_position + expanded_tokens_num) if gather_index_start_positi is not null.
        gather_index_start_position (torch.Tensor): The tensor indicating the start position of gather_index. Shape is (1).
        quant_mode(int): 0.The quant-mode of input quantization process.Current only support token quantization.
        asym_quant(bool):false. The flag indentify asymmetric quantization. Current only support symmetric quantization.
        quant_type(str): "int4".The flag indentify data type of quantified input. Current only support "int4", for furture extension.
        act_mode: The mode of activation, must be "none", "gelu", "swish".
        active_coef: The coefficient used in the swish activation. Default is 1.0.
        is_gated: A boolean parameter that indicates whether a gating mechanism is applied.It only takes effect when act_mode is not "none".
        output_lora_down(torch.Tensor):The lora-down projection output.
            shape[token_num, lora_rank] if gather_index is None,
            shape[expanded_tokens_num, lora_rank] if gather_index not None
        output_quant(torch.Tensor):The quantified input.
            shape[token_num, hidden_size / 2] if gather_index is None and is_gated false,
            shape[expanded_tokens_num, hidden_size / 2] if gather_index not None and is_gated false,
            shape[expanded_tokens_num, hidden_size / 4] if gather_index not None and is_gated  true,
            shape[token_num, hidden_size / 4] if gather_index is None and is_gated  true.
            For data type is int4x2, one packed two int4.
        output_quant_scales(torch.Tensor): The Reciprocal of quantization process.
            shape[token_num] if gather_index is None,
            shape[expanded_tokens_num] if gather_index not None

    Type:
        input: FP16, BF16
        weight_lora_down: FP16, BF16
        smooth: FP16, BF16
        lora_scales: FP32
        m_list: int32.
        gather_index: int32.
        gather_index_start_position: int32.
        quant_mode:: int
        asym_quant: bool
        quant_type: str, 'int4' or 'fp4', current only support 'int4'
        act_mode: str.
        active_coef: float.
        is_gated: bool.

    Return:
        output_lora_down: FP16, BF16
        output_quant: int8, one packed two int4 or fp4, shape (B, M // 2).
        output_quant_scales: FP32
    """
    token_num = input.size(0)
    if gather_index is not None:
        token_num = gather_index.size(0)
    assert m_list is not None, "m_list must be not None."
    assert weight_lora_down.dim() == 3, "dim of weight-loral-down must be equal 3."
    if output_lora_down is None:
        output_lora_down = torch.empty((token_num,
                weight_lora_down.size(-2)), dtype=input.dtype,
                device=input.device)
    if output_quant is None:
        output_quant = torch.empty((token_num,
                weight_lora_down.size(-1) // 2), dtype=torch.int8,
                device=input.device)
    if output_quant_scales is None:
        # for per-token quantization
        if quant_mode == 0:
            output_quant_scales = torch.empty(token_num,
                        dtype=torch.float32, device=input.device)

    torch.ops.torch_mlu_ops.svd_quant(input, weight_lora_down,
            smooth, lora_scales,
            output_lora_down, output_quant, output_quant_scales,
            quant_mode, asym_quant, quant_type,
            act_mode, active_coef, is_gated,
            m_list, gather_index, gather_index_start_position)

    return output_lora_down, output_quant, output_quant_scales

def fused_rope(qkv: torch.Tensor,
               k_cache_hp: torch.Tensor,
               v_cache_hp: torch.Tensor,
               sin_cache: torch.Tensor,
               cos_cache: torch.Tensor,
               position_id: torch.Tensor,
               k_gamma: torch.Tensor,
               k_beta: Optional[torch.Tensor],
               k_cache_lp: Optional[torch.Tensor] = None,
               v_cache_lp: Optional[torch.Tensor] = None,
               cache_bs_id_hp: Optional[torch.Tensor] = None,
               cache_seq_offsets_hp: Optional[torch.Tensor] = None,
               cache_bs_id_lp: Optional[torch.Tensor] = None,
               cache_seq_offsets_lp: Optional[torch.Tensor] = None,
               k_scale_hp: Optional[torch.Tensor] = None,
               v_scale_hp: Optional[torch.Tensor] = None,
               k_scale_lp: Optional[torch.Tensor] = None,
               v_scale_lp: Optional[torch.Tensor] = None,
               slot_mapping_hp: Optional[torch.Tensor] = None,
               slot_mapping_lp: Optional[torch.Tensor] = None,
               norm_type: str = "layernorm",
               rope_dim_offset: int = 0,
               eps: float = 1e-5,
               q_gamma: Optional[torch.Tensor] = None) :
    """
    Perform query and key fold rope + key layernorm/(query and key) rmsnorm + (key, value perchannel quantize with lp/hp scale) +
    reshape key value to kv cache.

    Math:
        q = qkv[:, :, 0:head_num_q]
        k = qkv[:, :, head_num_q:head_num_q + head_num_k].clone()
        v = qkv[:, :, head_num_q + head_num_k:].clone()
        if norm_type == "layernorm":
            q = apply_rotary(q, sin_cache, cos_cache, position_id)
            k = apply_rotary(k, sin_cache, cos_cache, position_id)
            k = layernorm(k, k_gamma, k_beta)
        if norm_type == "rmsnorm":
            if q_gamma is not None:
                q = rmsnorm(q, q_gamma)
            k = rmsnorm(k, k_gamma)
            k = apply_rotary(k, sin_cache, cos_cache, position_id)
            q = apply_rotary(q, sin_cache, cos_cache, position_id)

        if (key_scale is not None and value is not None):
            k = quantize(k, key_scale)
            v = quantize(v, value_scale)

        if (slot_mapping is not None):
            reshape_paged_cache(key_cache, value_cache, k, v, slot_mapping)
        else:
            reshape_linear_cache(key_cache, value_cache, k, v, cache_bs_id, cache_seq_offset)

    Args:
        qkv (torch.Tensor): The qkv tensor. Shape is (batch, seq_len, head_num_q + head_num_k * 2, head_size).
        k_cache_hp (torch.Tensor): The high precision key cache tensor. Shape is (max_bs, head_num_k, max_decode_len, head_size) or
            (num_blocks, head_num_k, block_size, head_size).
        v_cache_hp (torch.Tensor): The high precision value cache tensor. Shape is (max_bs, head_num_k, max_decode_len, head_size) or
            (num_blocks, head_num_k, block_size, head_size).
        sin_cache (torch.Tensor): The rotary sin table tensor. Shape is (rotary_seq, rope_dim).
        cos_cache (torch.Tensor): The rotary cos table tensor. Shape is (rotary_seq, rope_dim).
        position_id (torch.Tensor): The start RoPE position id of each batch. Shape is (batch).
        k_gamma (torch.Tensor): The normalization gamma tensor of k. Shape is (head_size).
        k_beta (torch.Tensor): optional. The normalization beta tensor of k. Shape is (head_size). Could be None.
        k_cache_lp (torch.Tensor): optional. The low precision key cache tensor. Shape is (max_bs, head_num_k, max_decode_len, head_size / 2) or
            (num_blocks, head_num_k, block_size, head_size / 2). Default is None.
        v_cache_lp (torch.Tensor): optional. The low precision value cache tensor. Shape is (max_bs, head_num_k, max_decode_len / 2, head_size) or
            (num_blocks, head_num_k, block_size / 2, head_size). Default is None.
        cache_bs_id_hp (torch.Tensor): optional. The high precision cache batch offset tensor of each batch. Shape is (batch). This tensor
            can be None, that means cache_bs_id is (0, 1, 2, 3, 4, ...). Default is None.
        cache_seq_offsets_hp (torch.Tensor): optional. The high precision cache seq_len offset tensor of each batch. Shape is (batch). Default is None.
        cache_bs_id_lp (torch.Tensor): optional. The low precision cache batch offset tensor of each batch. Shape is (batch). This tensor
            can be None, that means cache_bs_id is (0, 1, 2, 3, 4, ...). Default is None.
        cache_seq_offsets_lp (torch.Tensor): optional. The low precision cache seq_len offset tensor of each batch. Shape is (batch). Default is None.
        k_scale_hp (torch.Tensor): optional. The high precision key scale tensor that quantize key to int8. Shape is (head_num_k, head_size). Default is None.
        v_scale_hp (torch.Tensor): optional. The high precision value scale tensor that quantize value to int8. Shape is (head_num_k, head_size). Default is None.
        k_scale_lp (torch.Tensor): optional. The low precision key scale tensor that quantize key to int8. Shape is (max_bs, head_num_k, group_num). Default is None.
        v_scale_lp (torch.Tensor): optional. The low precision value scale tensor that quantize value to int8. Shape is (max_bs, head_num_k, group_num). Default is None.
        slot_mapping_hp (torch.Tensor): optional. The slot_mapping tensor of high precision cache. Shape is (batch*seq_len). Default is None.
        slot_mapping_lp (torch.Tensor): optional. The slot_mapping tensor of low precision cache. Shape is (batch*seq_len). Default is None.
        norm_type (str): The normalization type. "layernorm" and "rmsnorm" are available. Layernorm will be applied after key RoPE, and
            RMSnorm will be applied before key RoPE. Default is "layernorm".
        rope_dim_offset (int): The RoPE offset in head_size. Default is 0.
        eps (float): The layernorm eps param. Default is 1e-5.
        q_gamma (torch.Tensor): optional. The normalization gamma tensor of q. Shape is (head_size). Default is None.

    Type:
        qkv: half, bfloat16.
        k_cache_hp: same as qkv if not quant kv else int8.
        v_cache_hp: same as qkv if not quant kv else int8.
        sin_cache: same as qkv.
        cos_cache: same as qkv.
        position_id: int32.
        k_gamma: same as qkv.
        k_beta: same as qkv.
        k_cache_lp: int8(int4x2).
        v_cache_lp: int8(int4x2).
        cache_bs_id_hp: int32.
        cache_seq_offsets_hp: int32.
        cache_bs_id_lp: int32.
        cache_seq_offsets_lp: int32.
        k_scale_hp: float.
        v_scale_hp: float.
        k_scale_lp: float.
        v_scale_lp: float.
        slot_mapping_hp: int32.
        slot_mapping_lp: int32.
        q_gamma: same as qkv.

    Return:
        If both k_cache_lp and k_scale_lp are None, return a tuple of (qkv, k_cache_hp, v_cache_hp).
        If k_cache_lp exists but k_scale_lp is None, return a tuple of (qkv, k_cache_hp, v_cache_hp, k_cache_lp, v_cache_lp).
        If both k_cache_lp and k_scale_lp exist, return a tuple of (qkv, k_cache_hp, v_cache_hp, k_cache_lp, v_cache_lp, k_scale_lp, v_scale_lp).

    Note:
        1. head_size <= 128.
        2. head_size % 2 must be 0.
        3. head_num_q <= 32.
        4. head_num_k <= 32.
        5. rope_dim % 2 must be 0.
        6. rope_dim + rope_dim_offset must be less than or equal to head_size.
        7. All input tensors except sin_cache and cos_cache must be contiguous.
    """
    torch.ops.torch_mlu_ops.fused_rope(qkv, k_cache_hp, v_cache_hp, k_cache_lp, v_cache_lp,
            sin_cache, cos_cache, position_id, k_gamma, k_beta, k_scale_hp, v_scale_hp,
            k_scale_lp, v_scale_lp, cache_bs_id_hp, cache_seq_offsets_hp,
            cache_bs_id_lp, cache_seq_offsets_lp, slot_mapping_hp, slot_mapping_lp,
            norm_type, rope_dim_offset, eps, q_gamma)
    out = (qkv, k_cache_hp, v_cache_hp)
    if k_cache_lp is not None:
        out += (k_cache_lp, v_cache_lp)
        if k_scale_lp is not None:
            out += (k_scale_lp, v_scale_lp)
    return out

def update_out_and_lse(out: torch.Tensor,
                       lse: torch.Tensor,
                       block_out: torch.Tensor,
                       block_lse: torch.Tensor,
                       seq_offsets: Optional[torch.Tensor] = None,
                       cu_seqs: Optional[torch.Tensor] = None,
                       block_cu_seqs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
    """
    Update out and log-sum-exp(lse) according to block out and block lse.

    Math:
        out = out - F.sigmoid(block_lse - lse) * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
    Args:
        out (torch.Tensor): The output tensor. In pad mode, shape is (batch, max_seq_len, head_num, head_size).
                            In pack mode, shape is (total_seq_len, head_num, head_size).
        lse (torch.Tensor): The lse tensor. Shape is (batch, head_num, max_seq_len)
        block_out (torch.Tensor): The output tensor. In pad mode, shape is (batch, block_seq_len, head_num, head_size).
                                  In pack mode, shape is (total_block_seq_len, head_num, head_size).
        block_lse (torch.Tensor): The lse tensor. Shape is (batch, head_num, block_seq_len)
        seq_offsets (torch.Tensor): The seq offset of origin out and origin lse. Shape is (batch).
        cu_seqs (torch.Tensor): The cumulative sum of out seq_lens. Shape is (batch + 1).
        block_cu_seqs (torch.Tensor): The cumulative sum of block out seq_lens. Shape is (batch + 1).
    Type:
        out: half, bfloat16, float.
        lse: float.
        block_out: half, bfloat16, float.
        block_lse: float.
        seq_offsets: int32_t.
        cu_seqs: int32_t.
        block_cu_seqs: int32_t.
    Return:
        Support inplace outputs.
        Directly return the given out and lse.
    """
    torch.ops.torch_mlu_ops.update_out_and_lse(out, lse, block_out, block_lse,
                                               seq_offsets, cu_seqs, block_cu_seqs)
    return (out, lse)

def dequant_from_linear_cache(key: torch.Tensor,
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
                              quant_bit: int = 8) -> None:
    """
    De-quantizes the key and value tensors from the provided linear cache and scale.

    This function is used during inference to convert the quantized key and value tensors stored in the cache back to
    their original floating-point format. It supports both 8-bit and 4-bit quantization and different quantization modes
    (per-channel or per-head).


    Math:
        Using the key de-quantization processes as an example:
        len_i = context_lengths[i]
        start_pos_i = sum(context_lengths[:i]) if context_seq_offset is None else context_seq_offset[i]
        end_pos_i = start_pos_i + len_i
        key_i = key[start_pos_i:end_pos_i,...].transpose(1, 0)
        cache_pos = cache_bs_id[i]
        key_cache_i = key_cache[cache_pos,cache_seq_offset[i]:cache_seq_offset[i]+len_i,...]
        key_scale_i = key_scale[i,...] if quant_per_batch else key_scale
        if quant_mode == 0:
            key_scale_i = key_scale_i.reshape(head_num, -1, head_size)
        else:
            key_scale_i = key_scale_i.reshape(head_num, seq, -1)
        if quant_bit == 8:
            key_i[:] = ((key_cache_i.to(torch.float32) * key_scale_i).to(key.dtype)
        else:  # quant_bit == 4
            key_unpack_i = key_i.flatten()
            key_cache_flat_i = key_cache_i.flatten()
            key_unpack_i[0::2] = key_cache_flat_i >> 4
            key_unpack_i[1::2] = key_cache_flat_i << 4 >> 4
            key_i[:] = (key_unpack_i.reshape(key_i.shape).to(torch.float32) * key_scale_i).to(key.dtype)

    Args:
        key (torch.Tensor): The key tensor with shape (total_seqlen, head_num, head_size).
        value (torch.Tensor, optional): The value tensor with shape (total_seqlen, head_num, head_size). Default is None.
        key_cache (torch.Tensor): The key cache tensor, shape depends on the quantization bit width,
                                  where max_batch should be greater than or equal to the actual batch size.
        - For 8-bit quantization: shape is (max_batch, head_num, cache_mem_len, head_size).
        - For 4-bit quantization: shape is (max_batch, head_num, cache_mem_len, head_size//2).
        value_cache (torch.Tensor, optional): The value cache tensor, shape depends on the quantization bit width. Default is None.
        - For 8-bit quantization: shape is (max_batch, head_num, cache_mem_len, head_size).
        - For 4-bit quantization: shape is (max_batch, head_num, cache_mem_len//2, head_size).
        key_cache_quant_scale (torch.Tensor): The quantization scale tensor for the key cache. Shape depends on the quantization mode.
        - For per-channel quantization: shape is (head_num, head_size).
        - For per-token quantization: shape is (max_batch, head_num, cache_mem_len).
        value_cache_quant_scale (torch.Tensor, optional): The quantization scale tensor for the value cache, same shape as key_cache_quant_scale. Default is None.
        context_lengths (torch.Tensor): The actual lengths of the input sequences within the current context, shape (batch).
        max_context_len (int32): The maximum length of all sequences in the current context, used to determine cache size.
        context_seq_offset (torch.Tensor, optional): The starting position offset of each input sequence in the context, where cache_mem_len should be greater than or equal to max_context_len.
        If not provided, it is calculated as the cumulative sum of context_lengths. Shape (batch). Default is None.
        cache_bs_id (torch.Tensor, optional): The batch index in the cache where the key and value tensors will be placed. Shape (batch). Default is None.
        cache_seq_offset (torch.Tensor, optional): The starting position offset of each input sequence in the cache. Shape (batch). Default is None.
        quant_mode (int, optional): Quantization mode: 0 for per-channel quantization (each channel uses a different quantization scale),
                                    1 for per-token quantization (each token uses a different quantization scale). Default is 0.
        quant_bit (int, optional): Quantization bit width: 8 for 8-bit quantization, 4 for 4-bit quantization. Default is 8.

    Type:
        key: half, bfloat16
        value: half, bfloat16
        key_cache: int8
        value_cache: int8
        key_cache_quant_scale: float32
        value_cache_quant_scale: float32
        context_lengths: int32
        max_context_len: int32
        context_seq_offset: int32
        cache_bs_id: int32
        cache_seq_offset: int32
        quant_mode: int32
        quant_bit: int32

    Return:
        None.
    """

    torch.ops.torch_mlu_ops.dequant_from_linear_cache(
        key, value, key_cache, value_cache, key_cache_quant_scale, value_cache_quant_scale,
        context_lengths, max_context_len, context_seq_offset, cache_bs_id, cache_seq_offset,
        quant_mode, quant_bit
    )

def dequant_from_paged_cache(key: torch.Tensor,
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
                             quant_bit: int = 8) -> None:
    """
    Dequantize `key_cache` and `value_cache` to `key` and `value` based on quantization parameters.

    Math:
        block_num = context_lengths[i] // block_size
        rem_token_num = context_lengths[i] % block_size
        # Index key_cache based on block_tables and context_lengths
        key_cache_i = torch.concat(
            [key_cache[block_tables[i, j], ...] for j in range(block_num)] +
            ([key_cache[block_tables[i, block_num], :, :rem_token_num, :]] if rem_token_num > 0 else [])
        )
        # Get context begin and end indices
        context_begin_i = torch.cumsum(context_lengths, dims=-1)[i - 1] if context_seq_offset is None else context_seq_offset[i]
        context_end_i = context_begin_i + context_lengths[i]

        # Slice key based on context_lengths and context_seq_offset
        key_i = key[context_begin_i:context_end_i, ...]

        # Determine the scale based on quant_mode
        if quant_mode == 0:
            key_scale_i = key_cache_quant_scale[:, None, :]
        elif quant_mode == 1:
            key_scale_i = torch.concat(
                [key_cache_quant_scale[block_tables[i, j], ..., None] for j in range(block_num)] +
                ([key_cache_quant_scale[block_tables[i, block_num], :, :rem_token_num, None]] if rem_token_num > 0 else [])
            )

        # Dequantize key_cache to key
        key_i[:] = (key_cache_i * key_scale_i).transpose(1, 0)

    Args:
        key (torch.Tensor): Output tensor for dequantized key. Shape is (total_seqlens, head_num, head_size).
        value (torch.Tensor): Output tensor for dequantized value. Shape is (total_seqlens, head_num, head_size).
        key_cache (torch.Tensor): Quantized key cache tensor.
            If quant_mode is 0 or 1, Shape is (total_blocks, head_num, block_size, head_size).
            If quant_mode is 2 and quant_bit is 6, Shape is (total_blocks, head_num, block_size, head_size * 3 // 4).
        value_cache (torch.Tensor): Quantized value cache tensor.
            If quant_mode is 0 or 1, Shape is (total_blocks, head_num, block_size, head_size).
            If quant_mode is 2 and quant_bit is 6, Shape is (total_blocks, head_num, block_size, head_size * 3 // 4).
        key_cache_quant_scale (torch.Tensor): Quantization scale for key cache.
            If quant_mode is 0, Shape is (head_num, head_size).
            If quant_mode is 1, Shape is (max_batch, head_num, block_size).
            If quant_mode is 2, Shape is (max_batch, head_num, block_size, head_size // 32).
        value_cache_quant_scale (torch.Tensor): Quantization scale for value cache.
            If quant_mode is 0, Shape is (head_num, head_size).
            If quant_mode is 1, Shape is (max_batch, head_num, block_size).
            If quant_mode is 2, Shape is (max_batch, head_num, block_size // 32, head_size).
        context_lengths (torch.Tensor): Lengths of contexts for each batch. Shape is (batch).
        max_context_len (int): Maximum context length.
        context_seq_offset (torch.Tensor): Optional sequence offset for context lengths. Shape is (batch).
        block_tables (torch.Tensor): Block tables for indexing. Shape is (batch, max_block_num).
        quant_mode (int): Quantization mode. 0 for per-channel, 1 for per-token, 2 for mx. Default is 0.
        quant_bit (int): Quantization bit. Default is 8. Set to be 6 if using cnfp6 quantization.

    Type:
        key: half, bfloat16
        value: half, bfloat16
        key_cache: int8
        value_cache: int8
        key_cache_quant_scale: float32, bfloat16.
        value_cache_quant_scale: float32, bfloat16.
        context_lengths: int32
        max_context_len: int
        context_seq_offset: int32
        block_tables: int32
        quant_mode: int
        quant_bit: int

    Limitation:
        If quant_mode is 2, quant_bit must be 6, dtype of key_cache_quant_scale and value_cache_quant_scale must be bfloat16.

    Return:
        None
    """

    torch.ops.torch_mlu_ops.dequant_from_paged_cache(
        key, value, key_cache, value_cache, key_cache_quant_scale, value_cache_quant_scale,
        context_lengths, max_context_len, context_seq_offset, block_tables, quant_mode, quant_bit
    )

def dynamic_per_channel_quant(input:torch.Tensor,
                              seq_lens: Optional[torch.Tensor],
                              max_seq: int,
                              quant_out: Optional[torch.Tensor] = None,
                              quant_scale: Optional[torch.Tensor] = None,
                              quant_dtype: torch.dtype = torch.int8
                              ) -> Tuple[torch.Tensor,torch.Tensor]:

    """
    Perform dynamic per-channel quantization on input tensor with optional packed/padded format.

    Math:
        For each channel (head_num, head_size) and batch:
            1. Find absmax = max(abs(input[i])) along sequence dimension
            2. Calculate scale = absmax / qmax (127 for int8, 448 for float8_e4m3fn)
            3. Quantize: output = clip(round(input / scale), qmin, qmax)
            Pseudo-code:
                absmax = input.abs().max(dim=0)
                scale = absmax * (1.0 / qmax)
                quant_out = (input * (1.0 / scale)).round().clip(qmin, qmax)

    Args:
        input (torch.Tensor): Input tensor to be quantized.
                Shape: (batch, seq, head_num, head_size) for padded or
                       (total_seq, head_num, head_size) for packed format.
        seq_lens (torch.Tensor): Sequence lengths or cumulative lengths.
                Shape: (batch) for padded or (batch+1) for packed format.
        max_seq (int): the max length of seqs.
        quant_out (Optional[torch.Tensor]): Pre-allocated output tensor.
                Shape: Same as input.
        quant_scale (Optional[torch.Tensor]): Pre-allocated scale tensor.
                Shape: (batch, head_num, head_size).
        quant_dtype (torch.dtype): Quantization dtype (int8/float8_e4m3fn).

    Type:
        input: half, bfloat16.
        seq_lens: int32.
        max_seq: int
        quant_out: int8, float8_e4m3fn.
        quant_scale: float32.
        quant_dtype: torch.dtype.

    Return:
        Tuple[torch.Tensor,torch.Tensor]:
            quant_out: Quantized tensor with same shape as input.
            quant_scale: Per-channel scale tensor with shape (batch, head_num, head_size).
    """

    assert input.dim() in [3, 4], "input must be 3D (packed) or 4D tensor"
    packed = input.dim() == 3
    batch = seq_lens.size(0) - 1 if packed else input.size(0)
    head_num, head_size = input.size(-2), input.size(-1)

    # Create output tensors if not provided
    if quant_out is None:
        quant_out = torch.empty_like(input, dtype=quant_dtype)

    if quant_scale is None:
        quant_scale = torch.empty(batch, head_num, head_size,
                                 dtype=torch.float32, device=input.device)

    # Call MLU kernel
    torch.ops.torch_mlu_ops.dynamic_per_channel_quant(
        input,
        seq_lens,
        max_seq,
        quant_out,
        quant_scale
    )

    return quant_out, quant_scale

def reshape_from_cache(key: torch.Tensor,
                       value: Optional[torch.Tensor],
                       key_cache: torch.Tensor,
                       value_cache: Optional[torch.Tensor],
                       context_lengths: torch.Tensor,
                       max_context_len: int,
                       context_seq_offset: Optional[torch.Tensor] = None,
                       block_tables: Optional[torch.Tensor] = None,
                       cache_seq_offset: Optional[torch.Tensor] = None) -> None:
    """
    Reshape the key (and value if provided) tensors from the key_cache and value_cache based on context lengths and block tables.

    Math:
        For linear mode:
            - key_cache_i = key_cache[block_tables[i], :, cache_seq_offset[i]:cache_seq_offset[i]+context_lengths[i], :]
            - context_begin_i = cumsum(context_lengths, dims=-1)[i - 1] if context_seq_offset is None else context_seq_offset[i]
            - context_end_i = context_begin_i + context_lengths[i]
            - key[context_begin_i:context_end_i, ...] = key_cache_i
            - If value is not None and value_cache is not None:
                - value_cache_i = value_cache[block_tables[i], :, cache_seq_offset[i]:cache_seq_offset[i]+context_lengths[i], :]
                - value[context_begin_i:context_end_i, ...] = value_cache_i

        For paged mode:
            - full_num = context_lengths[i] // block_size
            - res_num = context_lengths[i] % block_size
            - key_cache_i = concat([key_cache[block_tables[i, j]:block_tables[i, j]+1, ...] for j in range(full_num)] +
                                   [key_cache[block_tables[i, full_num]:block_tables[i, full_num]+1, :, :res_num, :]] if res_num > 0 else [], dim=0)
            - context_begin_i = cumsum(context_lengths, dims=-1)[i - 1] if context_seq_offset is None else context_seq_offset[i]
            - context_end_i = context_begin_i + context_lengths[i]
            - key[context_begin_i:context_end_i, ...] = key_cache_i
            - If value is not None and value_cache is not None:
                - value_cache_i = concat([value_cache[block_tables[i, j]:block_tables[i, j]+1, ...] for j in range(full_num)] +
                                         [value_cache[block_tables[i, full_num]:block_tables[i, full_num]+1, :, :res_num, :]] if res_num > 0 else [], dim=0)
                - value[context_begin_i:context_end_i, ...] = value_cache_i

    Args:
        key (torch.Tensor): The target tensor to store reshaped key values. Shape is (total_length, head_num, head_size).
        value (Optional[torch.Tensor]): The target tensor to store reshaped value values. Should be provided if value_cache is also provided.
                                        Shape is (total_length, head_num, head_size) if present.
        key_cache (torch.Tensor): The source tensor containing cached key values.
                                  Shape is (max_batch_size, head_num, cache_mem_len, head_size) for linear mode or
                                  (total_blocks, head_num, block_size, head_size) for paged mode.
        value_cache (Optional[torch.Tensor]): The source tensor containing cached value values. Should be provided if value is also provided.
                                              Shape is (max_batch_size, head_num, cache_mem_len, head_size) for linear mode or
                                              (total_blocks, head_num, block_size, head_size) for paged mode if present.
        context_lengths (torch.Tensor): A 1D tensor representing the lengths of each batch context. Shape is (batch_size).
        max_context_len (int): The maximum length of the context that can be processed at once.
        context_seq_offset (Optional[torch.Tensor]): A 1D tensor representing the sequence offsets for each context.
                                                     Provides a shift offset for context begin if not None. Shape is (batch_size) if present.
                                                     Default is None.
        block_tables (Optional[torch.Tensor]): A tensor containing the block indices for each batch. Shape is (batch, 1) in linear mode, and
                                               shape is (batch_size, max_blocks) in paged mode. Default is None (linear mode).
        cache_seq_offset (Optional[torch.Tensor]): A 1D tensor representing the sequence offsets where the cache data starts for each batch.
                                                   Used for slicing the key and value cache. Shape is (batch_size) if present. Default is None.

    Type:
        key: float32, half, bfloat16, int8
        value: float32, half, bfloat16, int8
        key_cache: float32, half, bfloat16, int8
        value_cache: float32, half, bfloat16, int8
        context_lengths: int32
        context_seq_offset: int32
        block_tables: int32
        cache_seq_offset: int32

    Return:
        None
    """
    torch.ops.torch_mlu_ops.reshape_from_cache(
        key, value, key_cache, value_cache, context_lengths, max_context_len, context_seq_offset,
        block_tables, cache_seq_offset
    )

def fused_indexer_q(q: torch.Tensor,
                    w_q: torch.Tensor,
                    sin: torch.Tensor,
                    cos: torch.Tensor,
                    position_id: torch.Tensor,
                    output: Optional[torch.Tensor] = None,
                    hadamard_matrix: Optional[torch.Tensor] = None,
                    w_q_scale: Optional[torch.Tensor] = None,
                    output_quant_mode: str = 'none',
                    output_scale: Optional[torch.Tensor] = None,
                    interleaved: bool = True,
                    rope_at_front: bool = True):
    """
    This function fuses the query projection(Matmul), Rotary Position Embedding (RoPE), and an optional
    Hadamard transformation(Matmul) into a single high-performance kernel.

    Math:
        if w_q_scale is not None:
            input, input_scale = per_token_quant(input)
            q_proj = matmul(input, w_q.T) * (input_scale.unsqueeze(1) * w_q_scale.unsqueeze(0))
        else:
            q_proj = matmul(input, w_q.T)
        q_proj.reshape(token_num, head_num, head_size)
        if rope_at_front:
            q_pe = q_proj[..., :rotary_dim]
            q_no_pe = q_proj[..., rotary_dim:]
            q_pe_rotated = apply_rotary_embedding(q_pe, sin, cos, position_id, interleaved)
            q_rotated = cat((q_pe_rotated, q_no_pe), dim=-1)
        else:
            q_no_pe = q_proj[..., :-rotary_dim]
            q_pe = q_proj[..., -rotary_dim:]
            q_pe_rotated = apply_rotary_embedding(q_pe, sin, cos, position_id, interleaved)
            q_rotated = cat((q_no_pe, q_pe_rotated), dim=-1)
        if hadamard_matrix is not None:
            out = matmul(q_rotated.reshape(-1, head_size), hadamard_matrix.T)
        else:
            out = q_rotated
        out.reshape(token_num, head_num, head_size)
        if output_quant_mode == 'dynamic_per_token':
            output, output_scale = per_token_quant(out)
        else:
            output = out

    Args:
        q (torch.Tensor):
            The input tensor for query projection. Shape is (token_num, input_dim).
        w_q (torch.Tensor):
            The weight tensor for query projection. Shape is (head_num, head_size, input_dim).
        sin (torch.Tensor):
            A pre-computed tensor containing sine values for RoPE. Shape is (rotary_seq, rotary_dim).
        cos (torch.Tensor):
            A pre-computed tensor containing cosine values for RoPE. Shape is (rotary_seq, rotary_dim).
        position_id (torch.Tensor):
            A tensor indicating the position index for each token, used to gather from sin/cos tables.
            Shape is (token_num,).
        output (torch.Tensor):
            An output tensor to store the final result. The result of the operation will be written into this tensor
            in-place. Shape must be (token_num, head_num, head_size).
        hadamard_matrix (Optional[torch.Tensor]):
            An optional weight tensor for the Hadamard transformation. If provided, a matrix multiplication is
            performed after RoPE. Shape must be (head_size, head_size). Defaults to None.
        w_q_scale (Optional[torch.Tensor]):
            The scale tensor for the `w_q` weight, used for per-channel quantized matrix multiplication.
            If `w_q` is quantized (e.g., int8), this tensor is required. Shape is (head_num, head_size).
            Defaults to None.
        output_quant_mode (str, optional):
            Specifies the quantization mode for the output. Can be "none" or "dynamic_per_token".
            If "dynamic_per_token", the output will be quantized per token per head. Defaults to "none".
        output_scale (Optional[torch.Tensor]):
            An optional output tensor to store the quantization scales if `output_quant_mode` is
            "dynamic_per_token". If provided, the scales will be written into this tensor in-place.
            Shape must be (token_num, head_num). Defaults to None.
        interleaved (bool, optional):
            Control the Rotary Position Embedding mode.
            - If True: Uses interleaved mode (rotates adjacent pairs [x, x+1]).
            - If False: Uses folded/half-half mode (rotates pairs [x, x + rotary_dim/2]).
            Defaults to True.
        rope_at_front (bool, optional):
            Controls the position of RoPE within the head_size dimension.
            - If True: RoPE is applied to the first [0, rotary_dim) channels.
            - If False: RoPE is applied to the last [head_size - rotary_dim, head_size) channels.
            Defaults to True.

    Type:
        q: half, bfloat16.
        w_q: half, bfloat16.
        sin: same as q.
        cos: same as q.
        position_id: int32.
        output: same as input, or int8 if output is quantized.
        hadamard_matrix: same as q.
        w_q_scale: float32.
        output_quant_mode: str.
        output_scale: float32.
        interleaved: bool.

    Return:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            The return value depends on `output_quant_mode`.
            - if `output_quant_mode` is 'none': `output` (the same tensor passed as q)
            - if `output_quant_mode` is 'dynamic_per_token': `(output, output_scale)`

    Note:
        1. This is an in-place operation on the `output` and `output_scale` tensors.
        2. The `head_size` dimension of the `w_q` tensor must be a multiple of 128.
        3. The combination of `hadamard_matrix=None` and `output_quant_mode='dynamic_per_token'` is not supported.
           Output quantization requires the Hadamard transformation to be performed.
        4. All input tensors are expected to be contiguous except sin/cos.
        5. The rotary dimension (`rotary_dim`) is implicitly defined as `sin.shape[-1]`.
           It is expected to be `head_size / 2`.
        6. Input quantization (using `w_q_scale`) is not supported in the current version.
            `w_q` must be a `bfloat16` or `half` tensor and `w_q_scale` must be `None`.
        7. The Rotary Position Embedding (RoPE) supports both **interleaved** and **folded** modes via the `interleaved` parameter.
        8. The RoPE can be applied to either the beginning or the end of the head dimension
           controlled by the `rope_at_front` parameter.
    """
    output_quant = output_quant_mode == 'dynamic_per_token'
    token_num = q.size(0)
    head_num = w_q.size(0)
    head_size = w_q.size(1)
    if output is None:
        output_dtype = torch.int8 if output_quant else q.dtype
        output = torch.empty((token_num, head_num, head_size), dtype=output_dtype, device=q.device)
    if output_quant and output_scale is None:
        output_scale = torch.empty(token_num, head_num, dtype=torch.float32, device=q.device)
    torch.ops.torch_mlu_ops.fused_indexer_q(q, output, output_scale, w_q, w_q_scale, hadamard_matrix,
                                            sin, cos, position_id, output_quant_mode, interleaved, rope_at_front)
    if output_quant:
        return (output, output_scale)
    else:
        return output

def fused_mla_q(q: torch.Tensor,
                gamma: torch.Tensor,
                smooth_quant_scale: Optional[torch.Tensor],
                weight_b: torch.Tensor,
                weight_b_scale: torch.Tensor,
                weight_c: torch.Tensor,
                sin: torch.Tensor,
                cos: torch.Tensor,
                position_id: torch.Tensor,
                output: Optional[torch.Tensor] = None,
                eps: float = 1e-6,
                interleaved: bool = True,
                output_quant_mode: str = 'none',
                output_scale: Optional[torch.Tensor] = None,
                store_norm: bool = False,
                output_norm: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    This function fuses the calculations of the 'nope' and 'pe' parts in the MLA operation specifically
    for the input tensor 'q'.

    Math:
        norm_out = rmsnorm(q, gamma, eps)
        quanted_norm_out, input_scale = per_token_smooth_quant(norm_out, smooth_quant_scale)
        sqmm_out = smooth_quant_matmul(quanted_norm_out, input_scale, weight_b, weight_b_scale)
        sqmm_out.reshape(batch*seq, head_num, (nope_dim+pe_dim))
        q_nope = sqmm_out[..., :nope_dim]
        q_pe = sqmm_out[..., nope_dim:]
        bmm_out = bmm(q_nope.transpose(1, 0), weight_c.permute(0,2,1)).reshape(batch, seq, head_num, -1)
        q_pe_out = apply_rotary_embedding(q_pe, sin, cos, position_id, interleaved).reshape(batch, seq, head_num, pe_dim)
        out = cat((bmm_out, q_pe_out), dim=-1)

    Args:
        q (torch.Tensor):
            The query tensor. Shape is (batch, seq, input_size).
        gamma (torch.Tensor):
            The scaling parameter in the Root Mean Square Normalization (RMSNorm) operation. Shape is (input_size).
        smooth_quant_scale (torch.Tensor):
            A tensor representing the scale of input for smooth quantization. It can be set to None if not needed.
            otherwise, Its shape must be (input_size).
        weight_b (torch.Tensor):
            The weight tensor of matmul operation. Shape is (head_num * (nope_dim + pe_dim), input_size).
        weight_b_scale (torch.Tensor):
            The scale tensor for the `weight_b`. It is used to scale the
            weight_b tensor during quantization. Shape is (head_num * (nope_dim + pe_dim)).
        weight_c (torch.Tensor):
            The weight tensor of bmm. Shape is (head_num, head_size - pe_dim, nope_dim).
        sin (torch.Tensor):
            A tensor containing sin table. Shape is (rotary_seq, pe_dim).
        cos (torch.Tensor):
            A tensor containing cos table. Shape is same as sin.
        position_id (torch.Tensor):
            A tensor representing rotary seq_len offset of each batch. Shape is (batch).
        output (Optional[torch.Tensor]):
            An optional output tensor. If provided, the result of the operation will be stored in this tensor.
            Shape is (batch, seq, head_num, head_size).
        eps (float):
            A small constant used in the Root Mean Square Normalization (RMSNorm) operation for numerical stability.
        interleaved (bool):
            A bool parameter indicates rotary embedding mode. If interleaved is True, apply cross rotary embedding,
            otherwise apply fold rotary embedding.
        output_quant_mode(str, optional): quantize mode, which can be "none", "dynamic_per_token", "fp8e4-s1"
            and "mx-int8-s1". Default to "none".
        output_scale (Optional[torch.Tensor]):
            An optional output tensor. If provided, the quant scales of output will be stored in this tensor.
            Shape is (batch, seq, head_num) or (batch, seq, head_num, head_size / 32) if output_quant_mode is "fp8e4-s1"/"mx-int8-s1".
        store_norm (bool, optional):
            If True, the intermediate RMSNorm result will be computed and returned. Defaults to False.
        output_norm (Optional[torch.Tensor]):
            An optional output tensor to store the intermediate RMSNorm result.
            This is only used if `store_norm` is True. If `store_norm` is True and this is None,
            a new tensor will be allocated. Its shape must be (batch, seq, input_size).

    Type:
        q: half, bfloat16.
        gamma: same as q.
        smooth_quant_scale: float32.
        weight_b: int8, float8_e4m3fn.
        weight_b_scale: float32.
        weight_c: same as q.
        sin: same as q.
        cos: same as q.
        position_id: int32.
        output: same as q, int8, float8_e4m3fn, mxint8.
        eps: float32.
        interleaved: bool.
        output_quant_mode: str.
        output_scale: float, float8_e8m0fnu.
        output_norm: same as q.

    Return:
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
            The return value depends on `output_quant_mode` and `store_norm`.
            - if `output_quant_mode` is 'none' and `store_norm` is False: `output`
            - if `output_quant_mode` is not 'none' and `store_norm` is False: `(output, output_scale)`
            - if `store_norm` is True, `output_norm` is appended to the returned tuple.
              e.g., `(output, output_scale, output_norm)`

    Note:
        1. This is an inference operator, nan/inf behavior is undefined.
        2. The last dimension of q, sin and cos must be contiguous.
        3. Output must be contiguous if it is not None.
        4. Output_scale must be contiguous if it is not None.
        5. Gamma, smooth_quant_scale, weight_b, weight_b_scale, weight_c, position_id must be contiguous.
        6. pe_dim must be even (pe_dim % 2 == 0).
        7. All input tensors must be on the same MLU device.
    """
    output_quant = output_quant_mode != 'none' and output_quant_mode in ('dynamic_per_token', 'fp8e4-s1', 'mx-int8-s1')
    if (output is None) or (output_quant and (output_scale is None)):
        batch, seq = q.size(0), q.size(1)
        head_num, kv_lora_rank, nope_dim = weight_c.size(0), weight_c.size(1), weight_c.size(2)
        sum_nope_pe_dim = weight_b.size(0) // head_num
        pe_dim = sum_nope_pe_dim - nope_dim
        head_size = kv_lora_rank + pe_dim
        output_dtype = weight_b.dtype if output_quant else q.dtype
        if output is None:
            if output_quant_mode == 'none':
                output_dtype = q.dtype
            elif output_quant_mode == 'dynamic_per_token':
                output_dtype = weight_b.dtype
            elif output_quant_mode == 'fp8e4-s1':
                output_dtype = torch.float8_e4m3fn
            elif output_quant_mode == 'mx-int8-s1':
                output_dtype = torch.int8
            output = torch.empty((batch, seq, head_num, head_size), device="mlu", dtype = output_dtype)
        if (output_quant and output_scale is None):
            if output_quant_mode == 'dynamic_per_token':
                scale_shape = (batch, seq, head_num)
                scale_dtype = torch.float
            elif output_quant_mode in ('fp8e4-s1', 'mx-int8-s1'):
                # For MX Quant, shape is [B, S, H, head_size/32]
                # The dtype for mx_scale_s1_t is represented as float8_e8m0fnu in PyTorch.
                scale_shape = (batch, seq, head_num, (head_size + 31) // 32)
                scale_dtype = torch.float8_e8m0fnu
            output_scale = torch.empty(scale_shape, device="mlu", dtype=scale_dtype)
    if store_norm and output_norm is None:
        output_norm = torch.empty_like(q)
    final_output_norm_for_kernel = output_norm if store_norm else None
    torch.ops.torch_mlu_ops.fused_mla_q(q, output, output_scale, final_output_norm_for_kernel, gamma, smooth_quant_scale,
                                        weight_b, weight_b_scale, weight_c,
                                        sin, cos, position_id, output_quant_mode, eps, interleaved)
    outputs = [output]
    if output_quant:
        outputs.append(output_scale)
    if store_norm:
        outputs.append(output_norm)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

def fused_mla_kv(
  kv: torch.Tensor,
  sin: torch.Tensor,
  cos: torch.Tensor,
  position_id: torch.Tensor,
  gamma: torch.Tensor,
  kv_cache: torch.Tensor,
  kv_cache_scale: Optional[torch.Tensor],
  slot_mapping: Optional[torch.Tensor],
  cache_bs_id: Optional[torch.Tensor] = None,
  cache_seq_offset: Optional[torch.Tensor] = None,
  is_paged_cache: bool = True,
  eps: float = 1e-5,
  interleaved: bool = True,
  quant_mode: str = 'dynamic_per_token'):
    """
    Perform rope + rmsnorm + static per-channel quant or dynamic per-token quant to paged/linear kv_cache.

    Math:
        norm = kv[..., 0:norm_dim].rmsnorm()
        rope = kv[..., norm_dim:].rotary_embedding(interleaved)
        kv_out = concat((norm, rope), dim=-1)
        if quant:
            scaled = kv_out / kv_cache_scale
            kv_out = torch.round(scaled).clip(-128, 127).to(torch.int8)
        if is_paged_cache:
            for i in range(num_tokens):
                block_id = slot_mapping[i] // block_size
                block_offset = slot_mapping[i] % block_size
                kv_cache[block_id, :, block_offset, :] = kv_out[i]
        else:
            for i in range(batch):
                block_id = cache_bs_id[i]
                block_offset = cache_seq_offset[i]
                kv_cache[block_id, :, block_offset, :] = kv_out[i]

    Args:
        kv (torch.Tensor): Shape is (batch, seq, head_num, head_size).
        sin (torch.Tensor):  The rotary sin table tensor. Shape is (rotary_seq, rotary_dim).
        cos (torch.Tensor):  The rotary cos table tensor. Shape is (rotary_seq, rotary_dim).
        position_id (torch.Tensor): The rotary seq_len offset of each batch. Shape is (batch).
        gamma (torch.Tensor): The weight of layernorm. Shape is (norm_dim).
        kv_cache (torch.Tensor): The cache tensor. Shape is (num_blocks, num_heads, block_size, head_size).
        kv_cache_scale (torch.Tensor): Shape of (head_num, head_size) for the static per-channel quantization, which is an input tensor.
                                       Shape of (batch, head_num, head_size) for the static per-channel quantization, each batch has the individual scale.
                                       Shape of (num_blocks, head_num, block_size) for the dynamic per-token quantization, which is a output tensor.
        slot_mapping (torch.Tensor, optional): The slot_mapping tensor. Shape is (batch, seq).
        cache_bs_id (torch.Tensor, optional): The batch index in the cache where the kv tensors will be placed. Shape is (batch).
        cache_seq_offset (torch.Tensor, optional): A 1D tensor representing the sequence offsets where the cache data starts for each batch. Shape is (batch).
        is_paged_cache(bool): Describing the cache style. slot_mapping must exist if True.
        eps (float): The layernorm eps param.
        interleaved (bool): Describing the rope mode, Cross mode if True else fold mode.
        quant_mode(str, optional): quantize mode, which can be "dynamic_per_token", "static_per_channel" and "none". Default to "dynamic_per_token".

    Type:
        kv: half, bfloat16
        sin: same as kv
        cos: same as kv
        gamma: same as kv
        position_id: int32
        kv_cache: half, bfloat16, int8, float8_e4m3fn
        kv_cache_scale: float32
        slot_mapping: int32
        cache_bs_id: int32
        cache_seq_offset: int32

    Return:
        Support inplace outputs.
        Return kv_cache or (kv_cacke, kv_cache_scale).
    """
    is_dynamic_quant = quant_mode == "dynamic_per_token"
    torch.ops.torch_mlu_ops.fused_mla_kv(kv, sin, cos, position_id, gamma, kv_cache,
                                         kv_cache_scale, slot_mapping, cache_bs_id, cache_seq_offset,
                                         quant_mode, is_paged_cache, eps, interleaved)
    return (kv_cache, kv_cache_scale) if kv_cache_scale is not None and is_dynamic_quant else kv_cache

def scaled_matmul(a: torch.Tensor,
                  b: torch.Tensor,
                  a_scale: Optional[torch.Tensor],
                  b_scale: torch.Tensor,
                  output_dtype: torch.dtype,
                  bias: torch.Tensor = None,
                  c: torch.Tensor = None,
                  act_mode: str = "none",
                  quant_bit_size: int = 8,
                  alpha: float = 1.0,
                  beta: float = 1.0,
                  use_hp_active: bool = False,
                  a_quant_bit_size: int = -1,
                  a_calib: torch.Tensor = None,
                  b_calib: torch.Tensor = None,
                  output: torch.Tensor = None,
                  tile_config: Optional[Dict[str, int]] = None):
    """
    Perform quantized matrix multiplication on tensor a and b.

    Args:
        a (torch.Tensor): If a_quant_bit_size != 4, shape is (M, K).
                          If a_quant_bit_size = 4, shape is (M, K//2).
        b (torch.Tensor): If quant_bit_size != 4, shape is (N, K).
                          If quant_bit_size = 4, shape is (N, K//2).
        a_scale (Optional[torch.Tensor]): Shape can be (1) for per_tensor quantization or (M) for per_token quantization or (M, group_num) for groupwise quantization.
            groupsize should satisfied K % groupsize == 0. groupsize only support 32 when a is MX format.
        b_scale (torch.Tensor): Shape can be (N) when use per_channel quantization, (N, group_num) when use groupwise quantization,
            and (block_num, group_num) when use per_block groupwise quantization, only support fp8 and blocksize=groupsize=128.
            If quantization algorithm is weight_only and b use groupwise quantization, b_scale dtype could be same as a or float.
            If a/b both use groupwise quantization, a_scale/b_scale must have the same groupwise.
            Else if only b use groupwise quantization, groupsize support [64, 128, 256, 512, 1024].
            If a use per_tensor quantization, can set a_scale = None, and b_scale = b_scale * s_scale.
        output_dtype (torch.dtype): Specify the data type of output, must be torch.half or torch.bfloat16.
        bias (torch.Tensor, optional): Shape is (N).
        c (torch.Tensor, optional): Shape is (M, N).
        act_mode (str, optional): Choose the activation algorithm, must be 'silu', 'gelu', 'relu' or 'none'. act_mode must be 'none' when c is not None.
        quant_bit_size (int, optional): The data format of b. Defaults to 8. 4 for int4, 8 for int8 or float8.
        alpha (float, optional): coefficient of acted. Defaults to 1.0.
        beta (float, optional): coefficient of c. Defaults to 1.0.
        use_hp_active (bool, optional): Describing the algorithm that used in the implementation of the activation function.
             When the value is true, use the high-precision algorithm, otherwise use the fastest algorithm of activation.
             Defaults to False.
        a_quant_bit_size (int, optional): The data format of a. Defaults to -1. 4 for int4, 8 for int8 or float8, -1 for half and bf16.
        a_calib(Optional[torch.Tensor]): Calibration of a.
                                        If use flat_quant, a_calib could be None, or tensor with shape of (M, 2) and dtype of float.
                                        If use svd_quant, a_calib must be a tensor with shape (M, 32) or (M, 64), dtype be same as output.
        b_calib(Optional[torch.Tensor]): Calibration of b.
                                        If use flat_quant, b_calib could be None, or tensor with shape of (N, 2) and dtype of float.
                                        If use svd_quant, b_calib must be a tensor with shape (N, 32) or (N, 64), dtype be same as output.
        tile_config (Dict[str, int]): optional. Tile config, only supports smooth_quant, svd_quant, flat_quant quantization modes. Not support when a/b is 3-dim. including TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, WARP_SCHEDULER, SPLIT_K, NUM_STAGE, SWIZZLE_SIZE, DIRECTION
            tile_config = {'TILE_SIZE_M': 32, 'TILE_SIZE_N': 256, 'TILE_SIZE_K': 128, 'WARP_SCHEDULER': 4, 'SPILT_K': 1, 'NUM_STAGE': 1, 'SWIZZLE_SIZE': 2, 'DIRECTION': 2}.
            SWIZZLE_SIZE must be divisor of the cluster number of the device, DIRECTION support {1, 2}, NUM_STAGE only support 1 current.

    Type:
        a: int8, half, bfloat16, float8_e4m3fn, int4x2, float4_e2m1fn_x2
        a_scale: float, float8_e8m0fnu, bfloat16
        b: int8, float8_e4m3fn, int4x2, float4_e2m1fn_x2
        b_scale: the same as a_scale, or the same as a
        bias: same as output
        c: same as output
        output: specified by output_dtype, can be half and bfloat16
        a_calib: float, half, bfloat16
        b_calib: float, half, bfloat16

    Returns:
        A tensor with the shape of (M, N).
    """
    gemm_output_scale, quant_algo, a_quant_layout, b_quant_layout = None, 'none', 'none', 'none'
    if output is None:
        output = torch.empty((a.size(0), b.size(0)), dtype=output_dtype, device=a.device)
    if tile_config is None or tile_config == {}:
        torch.ops.torch_mlu_ops.scaled_matmul(output, a, b, a_scale, None, a_calib, b_scale, None, b_calib, bias, c, None,
                    None, gemm_output_scale, None, quant_algo, a_quant_layout, b_quant_layout, a_quant_bit_size,
                    quant_bit_size, act_mode, use_hp_active, 1.0, alpha, beta, False, True)
    else:
        torch.ops.torch_mlu_ops.scaled_matmul_tile(output, a, b, a_scale, None, a_calib, b_scale, None, b_calib, bias, c, None,
                None, gemm_output_scale, None, quant_algo, a_quant_layout, b_quant_layout, a_quant_bit_size,
                quant_bit_size, act_mode, use_hp_active, 1.0, alpha, beta, False, True, tile_config)
    return output

def scaled_quantize(x: torch.Tensor,
                    scale: Optional[torch.Tensor] = None,
                    zero:Optional[torch.Tensor] = None,
                    scale_ub:Optional[torch.Tensor] = None,
                    quant_type = torch.int8,
                    quant_mode: str = "dynamic_per_token",
                    act_mode: str = "none",
                    active_coef: float = 1.0,
                    is_gated: bool = False,
                    quant_bit_size: int = 8,
                    scale_type = torch.float32,
                    need_output_scale_trans: bool = False,
                    output_reduced: Optional[torch.Tensor] = None,
                    group_size: int = 1
                    )-> Tuple[torch.Tensor]:
    """
    Apply activation and quantization to the input tensor x.

    Args:
        x (torch.Tensor): The tensor to be quantized, shape is (..., C), must be continuous between 0 and -2 dimensions.
        scale (Optional[torch.Tensor], optional): The scale multipled to the input tensor.  Shape is (C) or (1).
        zero (Optional[torch.Tensor], optional):  Not supported, must pass None.
        scale_ub (Optional[torch.Tensor], optional): The output_scale upper bound.
            Take effect only if quant_type == torch.float8_e4m3fn and quant_mode == "dynamic_per_token".
        quant_type (optional): Output data type, can be torch.int8, torch.float8_e4m3fn. Defaults to torch.int8.
        quant_mode (str, optional): quantize mode, which can be "dynamic_per_token", "dynamic_per_tensor", "static_per_tensor"
            and "static_per_channel". Defaults to "dynamic_per_token".
        act_mode (str): The mode of activation, must be "none", "gelu", "silu", "swish".
        active_coef(float): The coefficient used in the swish activation. Default is 1.0.
        is_gated (bool): A boolean parameter that indicates whether a gating mechanism is applied. It only
                         takes effect when act_mode is not "none".
        quant_bit_size: Quantization bit width. Setting to 4 with int8 dtype enables int4x2 packing (two int4 in one int8).
        scale_type: Specifies the quantization scaling method. Set to 'float8_e8m0' or 'torch.bfloat16' in dynamic_quant mode to enable mx Quantization.
        need_output_scale_trans: A boolean parameter that indicates whether output scale transformation should be applied.
                                 It only effective in mx Quantization.
        output_reduced: An optional tensor to store the reduced output scale. Shape is (..., C // group_size),
                        or (..., C // (2 * group_size)) when is_gated.
        group_size: An integer parameter that indicates the group size for quantization. It only effective when output_reduced is not None,
                    group_size must be 128, 256, 512, 1024.

    Type:
        x: float, half, bfloat16.
        scale: float.
        scale_ub: float.
        act_mode: str
        active_coef: float
        is_gated: bool
        output: int8, float8_e4m3fn or float8_e5m2, float4_e2m1fn_x2.
        output_scale: float or foat8_e8m0, or bfloat16.
        output_reduced: float, half, bfloat16.
        group_size: int

    Returns:
        Tuple[torch.Tensor]: Returns (output, output_scale) if quant_mode is "dynamic_per_token" or "dynamic_per_tensor",
        otherwise returns output only.
    """
    is_dynamic_mode = quant_mode in ["dynamic_per_token", "dynamic_per_tensor"]
    if scale_type in (torch.float32,):
        if quant_type == torch.int8 and quant_bit_size == 4:
            # int4x2量化下的输出形状
            output = torch.empty(x.size()[:-1] + (x.size(-1) // 2 // (1+is_gated),), dtype=quant_type, device=x.device)
        else:
            # 非int4x2量化下的输出形状
           output = torch.empty(x.size()[:-1] + (x.size(-1) // (1+is_gated),), dtype=quant_type, device=x.device)

    if quant_mode == "dynamic_per_tensor":
        output_scale = torch.empty(1, dtype=torch.float32, device=x.device)
    if quant_mode == "dynamic_per_token" and scale_type in (torch.float32,):
        output_scale = torch.empty(x.size()[:-1], dtype=torch.float32, device=x.device)
    elif quant_mode == "dynamic_per_token" and scale_type in (torch.float8_e8m0fnu, torch.bfloat16 ):
        # sacle
        if need_output_scale_trans:
            output_scale = torch.empty((x.size(-1) // (1+is_gated) // 32,) + x.size()[:-1], dtype=scale_type, device=x.device)
        else:
            output_scale = torch.empty(x.size()[:-1] + (x.size(-1) // (1+is_gated) // 32,), dtype=scale_type, device=x.device)
        # output
        if quant_type in (torch.float4_e2m1fn_x2, ):
            # fp4x2量化下的输出形状
            output = torch.empty(x.size()[:-1] + (x.size(-1) // 2 // (1+is_gated),), dtype=quant_type, device=x.device)
        else:
            # 非fp4x2量化下的输出形状
            output = torch.empty(x.size()[:-1] + (x.size(-1) // (1+is_gated),), dtype=quant_type, device=x.device)
    if not is_dynamic_mode:
        output_scale = None
    torch.ops.torch_mlu_ops.scaled_quantize(x, output, output_scale, scale, zero, None, None, None, scale_ub, quant_mode, act_mode, active_coef, is_gated, quant_bit_size, need_output_scale_trans, output_reduced, group_size)
    return (output, output_scale) if is_dynamic_mode else output

def quant_per_block(q: torch.Tensor,
                    k: Optional[torch.Tensor],
                    seq_lens_q: Optional[torch.Tensor],
                    seq_lens_k: Optional[torch.Tensor],
                    max_seq_q: int,
                    max_seq_k: int,
                    block_size_q: int,
                    block_size_k: int,
                    smooth_k: bool = True,
                    quant_q: Optional[torch.Tensor] = None,
                    q_scale: Optional[torch.Tensor] = None,
                    quant_k: Optional[torch.Tensor] = None,
                    k_scale: Optional[torch.Tensor] = None,
                    quant_dtype: torch.dtype = torch.int8,
                    v: Optional[torch.Tensor] = None,
                    quant_v: Optional[torch.Tensor] = None,
                    v_scale: Optional[torch.Tensor] = None,
                    seq_lens_v: Optional[torch.Tensor] = None,
                    max_seq_v: int = 0) -> Tuple[torch.Tensor]:
    """
    Perform per_block quantization on q and k, and per_channel quantization on v.

    Math:
        Given q with shape of (seq, head_num, head_size):
        q_chunks = q.split(block_size_q, dim=0)
        [(quant_qi, scale_qi) = per_tensor_quantize(qi) for qi in q_chunks]
        quant_q = torch.cat([quant_qi], dim=0)
        q_scale = torch.cat([scale_qi], dim=0)

        Given k with shape of (seq, head_num, head_size):
        k_smooth = k - k.mean(dim=0)
        Then doing the quantization the same as q.

        Given v with shape of (seq, head_num, head_size):
        v_scale = v.abs().max(dim=0) / 127.0
        quant_v = (v / v_scale).round().clip(-128.0, 127.0).to(torch.int8)

    Args:
        q (torch.Tensor): The input query tensor.
                Shape must be (batch, max_seq_q, head_num_q, head_size_qk) or (total_seq_q, head_num_q, head_size_qk).
        k (Optional[torch.Tensor]): The input key tensor.
                Shape must be (batch, max_seq_kv, head_num_kv, head_size_qk) or (total_seq_kv, head_num_kv, head_size_qk).
        seq_lens_q (torch.Tensor): The sequence lengths or cumulative sequence lengths of query.
                Shape must be (batch+1) if the q is packed, otherwise (batch).
                Could be None, which indicates that sequence length of each batch is max_seq_q.
        seq_lens_k (torch.Tensor): The sequence lengths or cumulative sequence lengths of key.
                Shape must be (batch+1) if the k is packed, otherwise (batch).
                Could be None, which indicates that sequence length of each batch is max_seq_k.
        max_seq_q (int): The maximum sequence length of q.
        max_seq_k (int): The maximum sequence length of k.
        block_size_q (int): The quantized block size of query.
        block_size_k (int): The quantized block size of key.
        smooth_k (bool): If subtracting the mean data before doing quantization for key.
        quant_q (torch.Tensor): The output quantized query.
                Shape must be the same as q.
        q_scale (torch.Tensor): The quantized scale of query.
                Shape must be (batch, head_num_q, ceil(max_seq_q/block_size_q), 1).
        quant_k (Optional[torch.Tensor]): The output quantized key. Shape must be the same as k.
        k_scale (torch.Tensor): The quantized scale of key.
                Shape must be (batch, head_num_kv, ceil(max_seq_kv/block_size_k), 1).
        quant_dtype (torch.dtype): The data type after quantization. Only torch.int8 and torch.float8_e4m3fn are supported.
        v (Optional[torch.Tensor]): The input value tensor.
                Shape must be (batch, max_seq_v, head_num_kv, head_size_v) or (total_seq_kv, head_num_kv, head_size_v).
        quant_v (Optional[torch.Tensor]): The output quantized value.
                Shape must be the same as v.
        v_scale (Optional[torch.Tensor]): The per_channel quantized scale of value.
                Shape must be (batch, head_num_kv, head_size_v).
        seq_lens_v (torch.Tensor): The sequence lengths or cumulative sequence lengths of value.
                Shape must be (batch+1) if the v is packed, otherwise (batch).
                Could be None, which indicates that sequence length of each batch is max_seq_v.
        max_seq_v (int): The maximum sequence length of v.

    Type:
        q: float, half, bfloat16.
        k: same as q.
        seq_lens_q: int32.
        seq_lens_k: int32.
        quant_q: int8, float8_e4m3fn.
        q_scale: float.
        quant_k: same as quant_q.
        k_scale: float.
        v: same as q.
        quant_v: same as quant_q.
        v_scale: float.
        seq_lens_v: int32.

    Return:
        A tuple of (quant_q, q_scale, quant_k, k_scale, quant_v, v_scale),
        which (quant_k, k_scale) and (quant_v, v_scale) maybe absent according to the inputs.
    """
    assert q.dim() == 3 or q.dim() == 4, "dim of q must be equal to 3 or 4"
    packed = q.dim() == 3
    if packed:
        assert seq_lens_q is not None and seq_lens_q.dim() == 1, "dim of seq_lens_q must be equal to 1"
    batch = seq_lens_q.size(0) - 1 if packed else q.size(0)
    head_num_q = q.size(-2)
    if quant_q is None:
        quant_q = torch.empty(*q.shape, dtype=quant_dtype, device=q.device)
    if q_scale is None:
        max_block_num_q = (max_seq_q + block_size_q - 1) // block_size_q
        q_scale = torch.empty(batch, head_num_q, max_block_num_q, 1, dtype=torch.float, device=q.device)
    if k is not None:
        assert k.dim() == q.dim(), "k.dim() must be equal to q.dim()"
        head_num_k = k.size(-2)
        max_block_num_k = (max_seq_k + block_size_k - 1) // block_size_k
        if quant_k is None:
            quant_k = torch.empty(*k.shape, dtype=quant_dtype, device=k.device)
        if k_scale is None:
            k_scale = torch.empty(batch, head_num_k, max_block_num_k, 1, dtype=torch.float, device=k.device)
    if v is not None:
        assert v.dim() == q.dim(), "v.dim() must be equal to q.dim()"
        head_num_v = v.size(-2)
        if k is not None:
            assert head_num_v == head_num_k, "head_num_v must be equal to head_num_k"
        head_size_v = v.size(-1)
        if quant_v is None:
            quant_v = torch.empty(*v.shape, dtype=quant_dtype, device=v.device)
        if v_scale is None:
            v_scale = torch.empty(batch, head_num_v, head_size_v, dtype=torch.float, device=v.device)

    torch.ops.torch_mlu_ops.quant_per_block(
        q, k, v, seq_lens_q, seq_lens_k, seq_lens_v, max_seq_q, max_seq_k, max_seq_v,
        block_size_q, block_size_k, smooth_k, quant_q, q_scale, quant_k, k_scale, quant_v, v_scale,
        None)

    outputs = [quant_q, q_scale]
    if k is not None:
        outputs.append(quant_k)
        outputs.append(k_scale)
    if v is not None:
        outputs.append(quant_v)
        outputs.append(v_scale)
    return tuple(outputs)

torch.ops.torch_mlu_ops.quant_mx_qkv.default.mutates_args = (3, 4, 5, 6, 7, 8)
def quant_mx_qkv(q: torch.Tensor,
                 k: Optional[torch.Tensor] = None,
                 v: Optional[torch.Tensor] = None,
                 quant_q: Optional[torch.Tensor] = None,
                 q_scale: Optional[torch.Tensor] = None,
                 quant_k: Optional[torch.Tensor] = None,
                 k_scale: Optional[torch.Tensor] = None,
                 quant_v: Optional[torch.Tensor] = None,
                 v_scale: Optional[torch.Tensor] = None,
                 cu_seq_lens_q: Optional[torch.Tensor] = None,
                 cu_seq_lens_kv: Optional[torch.Tensor] = None,
                 quant_dtype: torch.dtype = torch.float8_e4m3fn,
                 max_seq_q: int = -1,
                 max_seq_kv: int = -1,
                 smooth_k: bool = True,
                 trans_v: bool = True,
                 scale_dtype: torch.dtype = torch.float8_e8m0fnu
                 ) -> Tuple[torch.Tensor]:
    """
    Perform per_block quantization on q and k, and per_channel quantization on v.

    Math:
        Given q with shape of (batch*seq, head_num, head_size):
        q_chunks = q.split(mx_group_size, dim=-1)
        [(quant_qi, scale_qi) = per_tensor_quantize(qi) for qi in q_chunks]
        quant_q = torch.cat([quant_qi], dim=-1)
        q_scale = torch.cat([scale_qi], dim=-1)

        Given k with shape of (batch*seq, head_num, head_size):
        k_smooth = k - k.mean(dim=0)
        Then doing the quantization the same as q.

        Given v with shape of (batch*seq, head_num, head_size):
        If not trans_v, doing the quantization the same as q.
        Else v = v.permute(1, 2, 0), doing the quantization the same as q.

    Args:
        q (torch.Tensor): The input query tensor.
                Shape must be (batch, max_seq_q, head_num_q, head_size_qk) or (total_seq_q, head_num_q, head_size_qk).
        k (Optional[torch.Tensor]): The input key tensor.
                Shape must be (batch, max_seq_kv, head_num_kv, head_size_qk) or (total_seq_kv, head_num_kv, head_size_qk).
        v (Optional[torch.Tensor]): The input key tensor.
                Shape must be (batch, max_seq_kv, head_num_kv, head_size_qk) or (total_seq_kv, head_num_kv, head_size_v).
        quant_q (torch.Tensor): The output quantized query. Shape is (head_num_q, batch, max_seq_q, head_size_qk) or (head_num_q, total_seq_q, head_size_qk)
        q_scale (torch.Tensor): The quantized scale of query.
                Shape is (head_num_q, batch, max_seq_q, ceil(head_size_qk/mx_group_size)) or (head_num_q, total_seq_q, ceil(head_size_qk/mx_group_size)).
        quant_k (Optional[torch.Tensor]): The output quantized key. Shape is (head_num_kv, batch, max_seq_kv, head_size_qk) or (head_num_kv, total_seq_kv, head_size_qk).
        k_scale (torch.Tensor): The quantized scale of key.
                Shape is (head_num_kv, batch, max_seq_kv, ceil(head_size_qk/mx_group_size)) or (head_num_kv, total_seq_kv, ceil(head_size_qk/mx_group_size)).
        quant_v (Optional[torch.Tensor]): The output quantized value. If trans_v, shape is (head_num_kv, head_size_v, (ceil(head_size_v / mx_group_size) + batch) * mx_group_size),
                Else shape is (head_num_kv, batch, max_seq_v, head_size_v) or (head_num_kv, total_seq_kv, head_size_v).
        v_scale (Optional[torch.Tensor]): The per_channel quantized scale of value. If trans_v, shape is (head_num_kv, head_size_v, pad(head_size_v, mx_group_size) // mx_group_size + batch)
                Else shape is (head_num_kv, batch, max_seq_v, pad(head_size_v, mx_group_size) // mx_group_size) or (head_num_kv, total_seq_kv, pad(head_size_v, mx_group_size) // mx_group_size).
        cu_seq_lens_q (torch.Tensor): The sequence lengths or cumulative sequence lengths of query.
                Shape must be (batch+1) if the q is packed, otherwise (batch).
                Could be None, which indicates that sequence length of each batch is max_seq_q.
        cu_seq_lens_kv (torch.Tensor): The sequence lengths or cumulative sequence lengths of key.
                Shape must be (batch+1) if the k is packed, otherwise (batch).
                Could be None, which indicates that sequence length of each batch is max_seq_k.
        quant_dtype (torch.dtype): The data type after quantization. Only torch.int8 and torch.float8_e4m3fn are supported.
        max_seq_q (int): The maximum sequence length of q.
        max_seq_kv (int): The maximum sequence length of k/v.
        smooth_k (bool): If subtracting the mean data before doing quantization for key.
        trans_v (bool): If transpose quant_v and v_scale.
        scale_dtype (torch.dtype): The data type of scale. Only torch.float8_e8m0fnu and torch.bfloat16 are supported.

    Type:
        q: half, bfloat16.
        k: same as q.
        v: same as q.
        quant_q: mx-int8, float8_e4m3fn.
        q_scale: float8_e8m0fnu, bfloat16.
        quant_k: same as quant_q.
        k_scale: float8_e8m0fnu, bfloat16.
        quant_v: same as quant_q.
        v_scale: float8_e8m0fnu, bfloat16.
        cu_seq_lens_q: int32.
    Return:
        A tuple of (quant_q, q_scale, quant_k, k_scale, quant_v, v_scale),
        which (quant_k, k_scale) and (quant_v, v_scale) maybe absent according to the inputs.
    """
    mx_group_size = 32
    assert q.dim() == 3 or q.dim() == 4, "dim of q must be equal to 3 or 4"
    packed = q.dim() == 3
    if packed:
        assert cu_seq_lens_q is not None and cu_seq_lens_q.dim() == 1, "dim of cu_seq_lens_q must be equal to 1"
    else:
        max_seq_q = q.size(1)
        max_seq_kv = k.size(1) if k is not None else (v.size(1) if v is not None else -1)

    batch = cu_seq_lens_q.size(0) - 1 if packed else q.size(0)
    head_num_q, head_size = q.size(-2), q.size(-1)
    block_num_qk = (head_size + mx_group_size - 1) // mx_group_size
    if quant_q is None:
        quant_q_shape = (head_num_q, q.size(0), head_size) if packed else (head_num_q, batch, q.size(1), head_size)
        quant_q = torch.empty(quant_q_shape, dtype=quant_dtype, device=q.device)
    if q_scale is None:
        scale_q_shape = (head_num_q, q.size(0), block_num_qk) if packed else (head_num_q, batch, max_seq_q, block_num_qk)
        q_scale = torch.empty(scale_q_shape, dtype=scale_dtype, device=q.device)
    if k is not None:
        assert k.dim() == q.dim(), "k.dim() must be equal to q.dim()"
        head_num_k = k.size(-2)
        quant_k_shape = (head_num_k, k.size(0), head_size) if packed else (head_num_k, batch, k.size(1), head_size)
        scale_k_shape = (head_num_k, k.size(0), block_num_qk) if packed else (head_num_k, batch, k.size(1), block_num_qk)
        if quant_k is None:
            quant_k = torch.empty(quant_k_shape, dtype=quant_dtype, device=k.device)
        if k_scale is None:
            k_scale = torch.empty(scale_k_shape, dtype=scale_dtype, device=k.device)
    if v is not None:
        assert v.dim() == q.dim(), "v.dim() must be equal to q.dim()"
        head_num_v = v.size(-2)
        if k is not None:
            assert head_num_v == head_num_k, "head_num_v must be equal to head_num_k"
        total_seq_v = v.size(0) if packed else v.size(0) * v.size(1)
        head_size_v = v.size(-1)
        block_num_v = (head_size_v + mx_group_size - 1) // mx_group_size
        quant_v_shape = (head_num_v, v.size(0), head_size_v) if packed else (head_num_v, v.size(0), v.size(1), head_size_v)
        scale_v_shape = (head_num_v, v.size(0), block_num_v) if packed else (head_num_v, v.size(0), v.size(1), block_num_v)
        if trans_v:
            if packed:
                block_num_v = (total_seq_v + mx_group_size - 1) // mx_group_size + batch
                total_v_pad = block_num_v * mx_group_size
                quant_v_shape = (head_num_v, head_size_v, total_v_pad)
                scale_v_shape = (head_num_v, head_size_v, block_num_v)
            else:
                seq_v_sn = (v.size(1) + mx_group_size - 1) // mx_group_size
                seq_v_pad = seq_v_sn * mx_group_size
                quant_v_shape = (head_num_v, head_size_v, batch * seq_v_pad)
                scale_v_shape = (head_num_v, head_size_v, batch * seq_v_sn)

        if quant_v is None:
            quant_v = torch.empty(quant_v_shape, dtype=quant_dtype, device=v.device)
        if v_scale is None:
            v_scale = torch.empty(scale_v_shape, dtype=scale_dtype, device=v.device)

    torch.ops.torch_mlu_ops.quant_mx_qkv(q, k, v, quant_q, q_scale, quant_k, k_scale, quant_v, v_scale, None, cu_seq_lens_q,
            cu_seq_lens_kv, max_seq_q, max_seq_kv, smooth_k, trans_v)
    outputs = [quant_q, q_scale]
    if k is not None:
        outputs.append(quant_k)
        outputs.append(k_scale)
    if v is not None:
        outputs.append(quant_v)
        outputs.append(v_scale)
    return tuple(outputs)

def quant_conv2d(input: torch.Tensor,
                 weight: torch.Tensor,
                 input_scale: torch.Tensor,
                 weight_scale: torch.Tensor,
                 stride: Optional[Tuple[int]] = (1,1),
                 padding: Optional[Tuple[int]] = (0,0),
                 dilation: Optional[Tuple[int]] = (1,1),
                 groups: Optional[int] = 1,
                 output_dtype: Optional[torch.dtype] = torch.bfloat16,
                 compute_dtype: Optional[torch.dtype] = torch.float,
                 bias: Optional[torch.Tensor] = None,
                 output: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Perform a quantized 2D convolution operation.

    Math:
        input_fp = torch.mul(input, input_scale).to(output_dtype)
        weight_fp = torch.mul(weight, weight_scale).to(output_dtype)
        has_bias = bias is not None
        conv2d = nn.Conv2d(input_fp.size(1), weight.size(0), (weight.size(2), weight.size(3)),
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=has_bias).mlu()
        conv2d.weight.data = weight_fp
        if has_bias:
            conv2d.bias.data = bias
        output = conv2d(input_fp)

    Args:
        input (torch.Tensor): The input tensor for the 2D convolution operation.
                Shape is (N, CI, HI, WI).
        weight (torch.Tensor): The weight tensor for the 2D convolution operation.
                Shape is (CO, CI/groups, kernel_size[0], kernel_size[1]).
        input_scale (torch.Tensor): The quantization scale of input.
                Only support per_tensor and per_pixel quantization, shape is (1) if per_tensor, else (N, H, W)
        weight_scale (torch.Tensor): The quantization scale of weight.
                Only support per_tensor and per_channel quantization, shape is (1) if per_tensor, else (CO).
        stride (Optional[Tuple[int]]): The stride of the convolution in each dimension.
                Default is (1, 1). The tuple should have a length of 2, the first int is used for the height dimension
                and the second int for the width dimension.
        padding (Optional[Tuple[int]]): The padding added to all four sides of the input.
                Default is (0, 0). The tuple should have a length of 2, representing padding in each spatial dimension.
                The first int is used for the height dimension and the second int for the width dimension.
        dilation (Optional[Tuple[int]]): The spacing between kernel points.
                Default is (1, 1). The tuple should have a length of 2, representing dilations in each spatial dimension.
                The first int is used for the height dimension and the second int for the width dimension.
        groups (Optional[int]): The number of blocked connections from input channels to output channels.
                Default is 1. It should be a positive integer. groups = CO / CI.
        output_dtype (Optional[torch.dtype]): The data type of the output tensor. Default is torch.bfloat16.
        compute_dtype (Optional[torch.dtype]): The data type used for computation. Default is torch.float.
        bias (Optional[torch.Tensor]): The optional bias tensor to be added to the output.
                Its shape should be compatible with the output of the convolution. Shape is (CO).
        output (Optional[torch.Tensor]): The optional output tensor to store the result.
                Shape is (N, CO, HO, WO).

    Type:
        input: int8.
        weight: int8.
        input_scale: fp32.
        weight_scale: fp32.
        stride: Tuple[int32].
        padding: Tuple[int32].
        dilation: Tuple[int32].
        groups: int32.
        output_dtype: torch.dtype.
        compute_dtype: torch.dtype.
        bias: bf16, half, float.
        output: bf16, half, float.

    Return:
        Covolution result tensor. Shape is (N, CO, HO, WO).

    """
    if output is None:
        # caculate output shape
        N = input.size(0)
        CO = weight.size(0)
        HO = (input.size(2) + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) // stride[0] + 1
        WO = (input.size(3) + 2 * padding[1] - dilation[1] * (weight.size(3) - 1) - 1) // stride[1] + 1
        output_size = (N, CO, HO, WO)
        output = torch.empty(output_size, dtype=output_dtype, device="mlu", memory_format=torch.channels_last)
    torch.ops.torch_mlu_ops.quant_conv3d(input, weight, bias, input_scale, weight_scale, output,
                                        stride, padding, dilation, groups, _torchDtype2Str(compute_dtype))
    return output

def quant_conv3d(input: torch.Tensor,
                 weight: torch.Tensor,
                 input_scale: torch.Tensor,
                 weight_scale: torch.Tensor,
                 stride: Optional[Tuple[int]] = (1,1,1),
                 padding: Optional[Tuple[int]] = (0,0,0),
                 dilation: Optional[Tuple[int]] = (1,1,1),
                 groups: Optional[int] = 1,
                 output_dtype: Optional[torch.dtype] = torch.bfloat16,
                 compute_dtype: Optional[torch.dtype] = torch.float,
                 bias: Optional[torch.Tensor] = None,
                 output: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Perform a quantized 3D convolution operation.

    Math:
        input_fp = torch.mul(input, input_scale).to(output_dtype)
        weight_fp = torch.mul(weight, weight_scale).to(output_dtype)
        has_bias = bias is not None
        conv3d = nn.Conv3d(input_fp.size(1), weight.size(0), (weight.size(2), weight.size(3), weight.size(4)),
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=has_bias).mlu()
        conv3d.weight.data = weight_fp
        if has_bias:
            conv3d.bias.data = bias
        output = conv3d(input_fp)

    Args:
        input (torch.Tensor): The input tensor for the 3D convolution operation.
                Shape is (N, CI, DI, HI, WI).
        weight (torch.Tensor): The weight tensor for the 3D convolution operation.
                Shape is (CO, CI/groups, kernel_size[0], kernel_size[1], kernel_size[2]).
        input_scale (torch.Tensor): The quantization scale of input.
                Only support per_tensor quantization, shape is (1). scale = f_max / i_max.
        weight_scale (torch.Tensor): The quantization scale of weight.
                Only support per_tensor quantization, shape is (1). scale = f_max / i_max.
        stride (Optional[Tuple[int]]): The stride of the convolution in each dimension.
                Default is (1, 1, 1). The tuple should have a length of 3, the first int is used for the depth dimension,
                the second int for the height dimension and the third int for the width dimension.
        padding (Optional[Tuple[int]]): The padding added to all six sides of the input.
                Default is (0, 0, 0). The tuple should have a length of 3, representing padding in each spatial dimension.
                The first int is used for the depth dimension, the second int for the height dimension and the third int
                for the width dimension.
        dilation (Optional[Tuple[int]]): The spacing between kernel points.
                Default is (1, 1, 1). The tuple should have a length of 3, representing dilations in each spatial dimension.
                The first int is used for the depth dimension, the second int for the height dimension and the third
                int for the width dimension.
        groups (Optional[int]): The number of blocked connections from input channels to output channels.
                Default is 1. It should be a positive integer. groups = CO / CI.
        output_dtype (Optional[torch.dtype]): The data type of the output tensor. Default is torch.bfloat16.
        compute_dtype (Optional[torch.dtype]): The data type used for computation. Default is torch.float.
        bias (Optional[torch.Tensor]): The optional bias tensor to be added to the output.
                Its shape should be compatible with the output of the convolution. Shape is (CO).
        output (Optional[torch.Tensor]): The optional output tensor to store the result.
                Shape is (N, CO, DO, HO, WO).

    Type:
        input: int8.
        weight: int8.
        input_scale: fp32.
        weight_scale: fp32.
        stride: Tuple[int32].
        padding: Tuple[int32].
        dilation: Tuple[int32].
        groups: int32.
        output_dtype: torch.dtype.
        compute_dtype: torch.dtype.
        bias: bf16, half, float.
        output: bf16, half, float.

    Return:
        Covolution result tensor. Shape is (N, CO, DO, HO, WO).

    """
    if output is None:
        # caculate output shape
        N = input.size(0)
        CO = weight.size(0)
        DO = (input.size(2) + 2 * padding[0] - dilation[0] * (weight.size(2) - 1) - 1) // stride[0] + 1
        HO = (input.size(3) + 2 * padding[1] - dilation[1] * (weight.size(3) - 1) - 1) // stride[1] + 1
        WO = (input.size(4) + 2 * padding[2] - dilation[2] * (weight.size(4) - 1) - 1) // stride[2] + 1
        output_size = (N, CO, DO, HO, WO)
        output = torch.empty(output_size, dtype=output_dtype, device="mlu", memory_format=torch.channels_last_3d)
    torch.ops.torch_mlu_ops.quant_conv3d(input, weight, bias, input_scale, weight_scale, output, stride,
                                         padding, dilation, groups, _torchDtype2Str(compute_dtype))
    return output

def masked_dot_select_sparse_paged_kv(q_low_rank: torch.Tensor,
                                      label_cache: torch.Tensor,
                                      context_lens: torch.Tensor,
                                      label_cache_block_table: torch.Tensor,
                                      kv_cache_block_table: torch.Tensor,
                                      recent_window: int,
                                      kv_cache_block_size: int,
                                      sparse_kv_length: int,
                                      sparse_block_table: torch.Tensor,
                                      sparse_context_lens: torch.Tensor):
    """
    Generates the sparse block_table by the attention logits of q_low_rank and label_cache.

    Math:
        Let i be the batch index. For details please check tests/ops_pytest/test_masked_dot_select_sparse_paged_kv.py generate_sparsed_block_table
        Select ki from sparse label_cache.
        ki = label_base[label_cache_block_table[i][:blkn_k_i]]
        ki = ki.transpose(0, 1)
        ki = ki.reshape(head_num, -1, head_size)
        ki = ki[:, :seq_k_i, :]
        qi = q_low_rank[i].permute(1, 2, 0)
        Compute the score.
        logit_i = torch.bmm(ki, qi, ).permute(0, 2, 1)
        Do reduce sum.
        reduced_logit_i = logit_i.reshape(head_num, seq_q, seq_k_i//kv_cache_blk_size, kv_cache_blk_size).sum(dim=-1,keepdim=False)
        For each head_num, seq_q do topk for each slice, concat recent window index with topk index.

    Args:
        q_low_rank(torch.Tensor): Low-rank query tensor. Shape is [batch, seq_q, head_num, head_size].
        label_cache(torch.Tensor): Low-rank paged cache of key. Shape is [total_label_cache_blkn, head_num, label_cache_blks, head_size].
        context_lens(torch.Tensor): Length of low-rank Key sequences per batch. Shape is [batch].
        label_cache_block_table(torch.Tensor): Indices for label_cache’s first dimension. Shape is [batch, label_cache_max_blkn].
        kv_cache_block_table(torch.Tensor): Indices for the actual KV cache blocks. Shape is [batch, kv_cache_max_blkn].
        recent_window(int): Number of nearby tokens (in KV cache) that are always selected.
        kv_cache_block_size(int): Size of each block in kv_cache_block_table.
        sparse_kv_length(int): The number of Top-K tokens selected by attention score.
        sparse_block_table(torch.Tensor): Selected block indices (for KV cache lookup). Shape is [batch, seq_q, head_num, kv_cache_max_blkn].
        sparse_context_lens(torch.Tensor): Valid lengths of selected blocks. Shape is [batch, seq_q].

    Type:
        q_low_rank: FP16, BF16
        label_cache: FP16, BF16
        context_lens: INT32
        label_cache_block_table: INT32
        kv_cache_block_table: INT32
        recent_window: INT32
        kv_cache_block_size: INT32
        sparse_kv_length: INT32
        sparse_block_table: INT32
        sparse_context_lens: INT32

    Return:
        sparse_block_table and sparse_context_lens are written inplace.
    """
    torch.ops.torch_mlu_ops.masked_dot_select_sparse_paged_kv(q_low_rank, label_cache, context_lens, label_cache_block_table,
                                         kv_cache_block_table, recent_window, kv_cache_block_size, sparse_kv_length, sparse_block_table, sparse_context_lens)
    return (sparse_block_table,  sparse_context_lens)

def apply_topkp(logits: torch.Tensor,
                index_in: torch.Tensor,
                per_slice_k: List,
                per_slice_p: List,
                logits_out: Optional[torch.Tensor] = None,
                sorted_logits_out: Optional[torch.Tensor] = None,
                index_out: Optional[torch.Tensor] = None,
                true_select_len: Optional[torch.Tensor] = None):
    """
    Apply topk filter and topp filter.

    k value in per_slice_k should be less than 20000.

    Math:
        sorted_logits,sorted_indices = logits.topk(per_slice_k[i])
        mask = sorted_logits.softmax().cumsum() > per_slice_p[i]
        mask_new[..., 1:] = mask[..., :-1].clone()
        mask_new[..., 0] = 0
        logits_out.scatter_(-1, sorted_indices, sorted_logits.masked_fill(mask, float('-inf')))

    Args
        logits(torch.Tensor): logits for sampling. Shape is [batch, vocab_size].
        index_in(torch.Tensor): Indices from 0 to vocab_size - 1. Shape is [vocab_size].
        per_slice_k(List)：k for each batch.
        per_slice_p(List)：p for each batch.
        logits_out(torch.Tensor): logits after sampling, set the unselected value to -inf. Shape is [batch, vocab_size].
        sorted_logits_out(torch.Tensor)：selected logits. Shape is [batch, vocab_size].
        index_out(torch.Tensor): selected indices. Shape is[batch, vocab_size].
        true_select_len(torch.Tensor): The number selected. Shape is [batch].

    Type:
        logits: BF16、FP32
        index_in: INT32
        per_slice_k: INT32
        per_slice_p: FP32
        logits_out: FP32
        sorted_logits_out: FP32
        index_out: INT32
        true_select_len: INT32

    Return:
        logits_out
        sorted_logits_out
        index_out
        true_select_len
    """
    #breakpoint()
    if logits_out is None:
        logits_out = torch.empty(logits.shape, dtype=torch.float, device=logits.device)
    if sorted_logits_out is None:
        sorted_logits_out = torch.empty(logits.shape, dtype=torch.float, device=logits.device)
    if index_out is None:
        index_out = torch.empty(logits.shape, dtype=torch.int32, device=logits.device)
    if true_select_len is None:
        true_select_len = torch.empty(logits.size(0), dtype=torch.int32, device=logits.device)

    #index_in = torch.arange(logits.shape[-1], dtype=torch.int32, device=logits.device)
    #sorted_logits_out.fill_(float('-inf'))
    logits_contiguous = logits
    if (logits.dim() != 2 or logits.stride(1) != 1 or logits.stride(0) < logits.size(1)) :
        logits_contiguous = logits.contiguous()
    torch.ops.torch_mlu_ops.apply_topkp(logits_contiguous, index_in, per_slice_k, per_slice_p, logits_out, sorted_logits_out, index_out, true_select_len)

    #breakpoint()
    #logits_out, sorted_topk_logits, sorted_topk_indices = apply_comebine_topk_topp(logits, per_slice_k, per_slice_p)
    return (logits_out, sorted_logits_out, index_out, true_select_len)

def random_sample(
    probs: torch.Tensor,
    is_gumbel_max: bool,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """
    Randomly sample form the probabilities

    Math:
        output = probs.div(exponential()).argmax(dim=-1).view(-1)

    Args:
        probs (torch.Tensor): The probability of sampling. Shape is (batch, vocab_size).
        is_gumbel_max (bool): Decide whether to sample the gumbel_max algorithm for sampling.
        generators (dict[int, torch.Generator]): Generators for random.

    Type:
        probs: float
        is_gumbel_max: bool

    Return:
        output
    """
    output = torch.empty((probs.size(0), 1), dtype=torch.int64, device=probs.device)
    if len(generators) == probs.size(0):
        for i, generator in generators.items():
            torch.ops.torch_mlu_ops.random_sample(probs[i].unsqueeze(0), output[i].unsqueeze(0),
                is_gumbel_max, generator)
    else:
        torch.ops.torch_mlu_ops.random_sample(probs, output, is_gumbel_max, None)
    return output

def rejection_sample(draft_token_ids: torch.Tensor,
                             num_draft_tokens: torch.Tensor,
                             cu_num_draft_tokens: torch.Tensor,
                             draft_probs: Optional[torch.Tensor],
                             target_probs: torch.Tensor,
                             bonus_token_ids: torch.Tensor,
                             uniform_rand: torch.Tensor,
                             uniform_probs: torch.Tensor,
                             max_spec_len: int,
                             high_acc: bool = True) -> torch.Tensor:
    """
    By comparing the probability distribution of the draft model and the target model, decide whether to accept the candidate token generated by the draft model.
    If rejected, resample from the recovered distribution. The final output contains the accepted token and possible bonus tokens.

    Math:
        selected_draft_probs = draft_probs[batch_indices, probs_indices, draft_token_ids]
        selected_target_probs = target_probs[batch_indices, probs_indices, draft_token_ids]
        accepted = uniform_rand < torch.minimum(selected_target_probs / selected_draft_probs, torch.full((1, ), 1)

        recovered_token_ids = torch.clamp(target_probs - draft_probs, min=0).div_(uniform_probs).argmax(dim=-1)

        for each batch:
            if index < first_0_index in accepted:
                output_token_ids = draft_token_ids
            else if index == first_0_index in accepted:
                output_token_ids = recovered_token_ids
            else:
                output_token_ids = -1
            if accepted is all true:
                output_token_ids += bonus_token_ids

    Args
        draft_token_ids(torch.Tensor): The draft token index which will be selected. Shape is [num_tokens], which num_tokens equals the cu_num_draft_tokens[batch_size - 1].
        num_draft_tokens(torch.Tensor): The number of draft token in each batch. Shape is [batch_size], which batch_size equals the len(num_draft_tokens).
        cu_num_draft_tokens(torch.Tensor): The accumulated number of draft token in each batch. Shape is [batch_size].
        draft_probs(torch.Tensor): The probility distributions of draft model. Shape is [num_tokens, vocab_size], which vocab_size means the length of tokens.
        target_probs(torch.Tensor): The probility distributions of target model. Shape is [num_tokens, vocab_size], which vocab_size means the length of tokens.
        bonus_token_ids(torch.Tensor): The bunus token index which will be selected when all draft_token_ids have been accepted. Shape is [batch_size].
        uniform_rand(torch.Tensor): The random probability which will be compared with the probility selected by draft_token_ids. Shape is [num_tokens].
        uniform_probs(torch.Tensor): The random probability which will be calculated to get the recovered probability. Shape is [num_tokens, vocab_size].
        max_spec_len(int): The maxmium value in num_draft_tokens.
        high_acc(bool): High-precision parameter, the default value is true. If it is set to false, it means that the bit width will not be increased for low-bit width input. Not enabled now.

    Type:
        draft_token_ids: INT32
        num_draft_tokens: INT32
        cu_num_draft_tokens: INT32
        draft_probs: FP32，FP16，BF16
        target_probs: FP32，FP16，BF16
        bonus_token_ids: INT32
        uniform_rand: FP32
        uniform_probs: FP32
        max_spec_len：INT32
        high_acc：BOOL

    Return:
        output_token_ids
    """
    output_token_ids = torch.empty(num_draft_tokens.numel() + draft_token_ids.numel(), dtype=torch.int32, device=draft_token_ids.device)
    torch.ops.torch_mlu_ops.rejection_sample(output_token_ids, draft_token_ids, num_draft_tokens, cu_num_draft_tokens, draft_probs,
                                            target_probs, bonus_token_ids, uniform_rand, uniform_probs, max_spec_len, high_acc)
    return output_token_ids

def index_selected_rotary_embedding(
    input: torch.Tensor,
    sin_table: torch.Tensor,
    cos_table: torch.Tensor,
    ids: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    input_discrete_only: bool = False
) -> torch.Tensor:
    """
    Apply index-selected cross rotary embedding.

    Math:
        selected_input = input[ids]
        if input_discrete_only:
            ids_len = ids.size(0)
            selected_sin = sin_table[:ids_len]
            selected_cos = cos_table[:ids_len]
        else:
            selected_sin = sin_table[ids]
            selected_cos = cos_table[ids]
        rotary_dim = selected_sin.shape[-1]
        input_rot = selected_input[..., :rotary_dim]
        x1, x2 = input_rot[..., ::2], input_rot[..., 1::2]
        input_rot[..., ::2], input_rot[..., 1::2] = -x2, x1
        output = input_rot * selected_sin + selected_input * selected_cos
        return output

    Args:
        input (torch.Tensor): Shape is (total_seq, head_num, head_size).
        sin_table (torch.Tensor): Shape must be (rotary_seq_len, rotary_dim).
        cos_table (torch.Tensor): Shape must be (rotary_seq_len, rotary_dim).
        ids (torch.Tensor): The index of input tokens. Shape is (seq_len).
        output (torch.Tensor) optional: Shape is (seq_len, head_num, head_size).
        input_discrete_only (bool): The default value is false. If false, it means that both input and sin/cos table should index select, else only input need index select.

    Type:
        input: half, bfloat16.
        sin_table: same as input.
        cos_table: same as input.
        ids: int32.
        output: same as input.

    Return:
        Return the output tensor.
    """
    if output is None:
        seq_len = ids.size(0)
        head_num = input.shape[-2]
        head_size = input.shape[-1]
        output = torch.empty((seq_len, head_num, head_size), device=input.device, dtype=input.dtype)
    torch.ops.torch_mlu_ops.index_selected_rope(input, output, sin_table, cos_table, ids, input_discrete_only)
    return output

def apply_topkp_v2(logits: torch.Tensor,
                   index_in: torch.Tensor,
                   temperature_list: torch.Tensor,
                   minp_list: torch.Tensor,
                   topk_list: torch.Tensor,
                   topp_list: torch.Tensor,
                   logits_out: Optional[torch.Tensor] = None,
                   sorted_logits_out: Optional[torch.Tensor] = None,
                   index_out: Optional[torch.Tensor] = None,
                   true_select_len: Optional[torch.Tensor] = None):
    """
    Apply topk filter and topp filter.

    k value in topk_list should be less than 10000.

    Math:
        logits = logits / temperature_list.unsqueeze(1)
        sorted_logits,sorted_indices = logits.topk(topk_list[i])
        mask = sorted_logits.softmax().cumsum() > topp_list[i]
        mask_new[..., 1:] = mask[..., :-1].clone()
        mask_new[..., 0] = 0
        logits_out.scatter_(-1, sorted_indices, sorted_logits.masked_fill(mask, float('-inf')))

    Args
        logits(torch.Tensor): logits for sampling. Shape is [batch, vocab_size].
        index_in(torch.Tensor): Indices from 0 to vocab_size - 1. Shape is [vocab_size].
        temperature_list(torch.Tensor): temperature for each batch. Shape is [batch].
        minp_list(torch.Tensor): minp for each batch. Shape is [batch].
        topk_list(torch.Tensor): topk for each batch. Shape is [batch].
        topp_list(torch.Tensor): topp for each batch. Shape is [batch].
        logits_out(torch.Tensor): logits after sampling, set the unselected value to -inf. Shape is [batch, vocab_size].
        sorted_logits_out(torch.Tensor): selected logits. Shape is [batch, vocab_size].
        index_out(torch.Tensor): selected indices. Shape is[batch, vocab_size].
        true_select_len(torch.Tensor): The number selected. Shape is [batch].

    Type:
        logits: FP32
        index_in: INT32
        temperature_list: FP32
        minp_list: FP32
        topk_list:INT32
        topp_list: FP32
        logits_out: FP32
        sorted_logits_out: FP32
        index_out: INT32
        true_select_len: INT32

    Return:
        logits_out
        sorted_logits_out
        index_out
        true_select_len
    """
    if logits_out is None:
        logits_out = torch.empty(logits.shape, dtype=torch.float, device=logits.device)
    if sorted_logits_out is None:
        sorted_logits_out = torch.empty(logits.shape, dtype=torch.float, device=logits.device)
    if index_out is None:
        index_out = torch.empty(logits.shape, dtype=torch.int32, device=logits.device)
    if true_select_len is None:
        true_select_len = torch.empty(logits.size(0), dtype=torch.int32, device=logits.device)

    if temperature_list is not None and minp_list is None and topk_list is None and topp_list is not None:
        logits, index_out = torch.topk(logits, logits.size(1))
        index_out = index_out.to(dtype=torch.int32)

    torch.ops.torch_mlu_ops.apply_topkp_v2(logits, index_in,
                                           temperature_list, minp_list, topk_list, topp_list,
                                           logits_out, sorted_logits_out, index_out, true_select_len)
    return (logits_out, sorted_logits_out, index_out, true_select_len)

def gather_split(input: torch.Tensor,
                 gather_index: torch.Tensor,
                 valid_token_num: torch.Tensor,
                 output1: torch.Tensor,
                 output2: Optional[torch.Tensor] = None,
                 output3: Optional[torch.Tensor] = None):
    """
    Gather and split valid tokens from input tensor.

    Math:
        size_1 = output1.size(-1)
        valid_token_num = valid_token_num[0].item()
        valid_index = gather_index[:valid_token_num]
        output = input[valid_index]
        output1[:valid_token_num, :] = output[..., :size_1].contiguous()
        if output2 is not None:
            size_2 = output2.size(-1)
            output2[:valid_token_num, :] = output[..., size_1:size_1+size_2].contiguous()
        if output3 is not None:
            size_3 = output3.size(-1)
            output3[:valid_token_num, :] = output[..., input.size(-1)-size_3:].contiguous()

    Args:
        input (torch.Tensor): Shape is (token_num, input_size). input_size = size_1 + size_2 + size_3.
        gather_index (torch.Tensor): Shape is (token_num).
        valid_token_num (torch.Tensor): Shape is (1).
        output1 (torch.Tensor): Shape is (token_num, size_1). Size_1 must be less than or equal to input_size.
        output2 (torch.Tensor) optional: Shape is (token_num, size_2). 0 <= size_2 <= 4.
        output3 (torch.Tensor) optional: Shape is (token_num, size_3). 0 <= size_3 <= 4.

    Type:
        input: int8, float, half, bfloat16.
        gather_index: int32.
        valid_token_num: int32.
        output1: same as input.
        output2: same as input.
        output3: same as input.

    Return:
        None.
    """
    torch.ops.torch_mlu_ops.gather_split(output1, output2, input, gather_index, valid_token_num, output3)

def hshare(block_table: torch.Tensor,
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
           kv_head_num: int):
    """
    Apply head-wise key-value block sharing (HShare) transformation.

    This function reorganizes the key-value block index table using a front/mid/back structure based on the
    given `ratios`, enabling efficient memory reuse across transformer layers and heads. It optionally skips
    specified layers via `disable_hshare_layer`. Mid-blocks are randomly selected using precomputed
    `indices_cache`.

    Math:
        For each batch b in actual_batch_size:
            kv_len = kv_len_after_store[b]
            [origin_total_blocks, front_blocks, back_blocks, mid_blocks] = block_num_cache[b]

            If all > 0:
                mid_range = block_table[b, front_blocks : origin_total_blocks - back_blocks]
                selected_mid = mid_range[indices_cache[indices_cache_offset[b] : ...]]

                hshare_block_table[b, :front_blocks] = front_blocks
                hshare_block_table[b, front_blocks:front_blocks+mid_blocks] = selected_mid
                hshare_block_table[b, front_blocks+mid_blocks:front_blocks+mid_blocks+back_blocks] = back_blocks

                kv_len_new = (front_blocks + mid_blocks + back_blocks) * block_size (+ remainder)
                kv_len_after_store[b] = kv_len_new

        For disabled layers:
            Copy original block_table and kv_len_after_store

    Args:
        block_table (torch.Tensor): Original block table.
            Shape: [max_batch_size, max_seq_len // block_size]. Type: int32.
        kv_len_after_store (torch.Tensor): KV length before HShare.
            Shape: [max_batch_size]. Type: int32.
        disable_hshare_layer (Optional[torch.Tensor]): Indices of layers not using HShare.
            Shape: [disable_hshare_layer_num]. Type: int32.
        ratios (torch.Tensor): Proportion of front, mid, and back blocks.
            Shape: [3]. Type: float32.
        indices_cache (torch.Tensor): Precomputed mid indices for each kv_len.
            Shape: [sum(indices_cache_offset)]. Type: int32.
        indices_cache_offset (torch.Tensor): Offset into `indices_cache` per sample.
            Shape: [max_batch_size]. Type: int32.
        block_num_cache (torch.Tensor): Block counts [origin_total, front, back, mid] per batch.
            Shape: [max_batch_size, 4]. Type: int32.
        actual_batch_size (int): Number of valid batches.
        max_seq_len (int): Maximum sequence length.
        block_size (int): Size of one block (in tokens).
        layer_num (int): Total transformer layer count.
        kv_head_num (int): Number of KV heads per layer.

    Type:
        block_table: int32
        kv_len_after_store: int32
        disable_hshare_layer: int32
        ratios: float32
        indices_cache: int32
        indices_cache_offset: int32
        block_num_cache: int32
        hshare_block_tables: int32
        hshare_kv_len_after_store: int32

    Return:
    Tuple of:
        hshare_block_tables (torch.Tensor): Block table after HShare.
            Shape: [layer_num, max_batch_size, kv_head_num, max_blocks]. Type: int32.
        hshare_kv_len_after_store (torch.Tensor): Updated kv_len after block replacement.
            Shape: [layer_num, max_batch_size]. Type: int32.
    """
    max_batch_size = block_table.shape[0]

    hshare_block_tables = block_table.unsqueeze(-1).repeat(1, 1, layer_num * kv_head_num)
    hshare_block_tables = hshare_block_tables.view(max_batch_size, -1, layer_num, kv_head_num).permute(2, 0, 3, 1)
    hshare_block_tables = hshare_block_tables.contiguous().mlu()

    hshare_kv_len_after_store = torch.empty(
        (layer_num, kv_len_after_store.shape[0]),
        dtype=torch.int32,
    ).mlu()
    torch.ops.torch_mlu_ops.hshare(hshare_block_tables, hshare_kv_len_after_store, block_table,
                                   kv_len_after_store, disable_hshare_layer, ratios, indices_cache,
                                   indices_cache_offset, block_num_cache, actual_batch_size, max_seq_len, block_size, layer_num, kv_head_num)

    return (hshare_block_tables, hshare_kv_len_after_store)

def moe_all2all_gen_gather_index(token_num: torch.Tensor, pad_num: int, return_cusum_token_count: bool = False):
    """
    Generate indices that transform rank-order data to expert-order and
    transform expert-order data to rank-order.

    Math:
        Please refer to tests/ops_pytest/test_moe_all2all_gen_gather_index.py:TestMoeAll2AllGenGatherIndexOp.op_impl_base

    Args:
        token_num (torch.Tensor): The table that indicates the relationship of token for each Expert Parallel part.
            The shape is (rank_num, expert_num). rank_num is the amount of device in Expert Parallel. expert_num is
            the amount of expert that handles in one device.
        pad_num (int): The max token count for each rank.
        return_cusum_token_count (bool): Returns cusum_token_count if True. Default as false.

    Type:
        token_num: int32
        pad_num: int32
        return_cusum_token_count: bool

    Return:
        gather_by_expert_index (torch.Tensor): Int. The index that transform rank-ordered data to expert-ordered.
        gather_by_rank_index (torch.Tensor): Int. The index that transform expert-ordered data to rank-ordered.
        token_count (torch.Tensor): Int. The token amount that handled by each expert.
        token_sum (torch.Tensor): Int. Total token amount.
        cusum_token_count(torch.Tensor): The prefix-sum of token_count.
    """
    rank_num = token_num.size(0)
    expert_num = token_num.size(1)
    gather_by_expert_index = torch.empty(rank_num * pad_num, dtype=torch.int32, device=token_num.device)
    gather_by_rank_index = torch.empty(rank_num * pad_num, dtype=torch.int32, device=token_num.device)
    token_count = torch.empty(expert_num, dtype=torch.int32, device=token_num.device)
    cusum_token_count = None
    if return_cusum_token_count:
        cusum_token_count = torch.empty(expert_num + 1, dtype=torch.int32, device=token_num.device)
    token_sum = torch.empty(1, dtype=torch.int32, device=token_num.device)
    torch.ops.torch_mlu_ops.moe_all2all_gen_gather_index(gather_by_expert_index, gather_by_rank_index,
                                                         token_count, cusum_token_count, token_sum, token_num, pad_num)
    if return_cusum_token_count:
        return gather_by_expert_index, gather_by_rank_index, token_count, token_sum, cusum_token_count
    else:
        return gather_by_expert_index, gather_by_rank_index, token_count, token_sum

def compress_kv(k: torch.Tensor,
                v: torch.Tensor,
                cu_seq_lens: Optional[torch.Tensor],
                k_weight: torch.Tensor,
                v_weight: torch.Tensor,
                pe_table_k: torch.Tensor,
                pe_table_v: torch.Tensor,
                max_seq_len: int,
                compress_length: int,
                compress_stride: int,
                k_out: Optional[torch.Tensor] = None,
                v_out: Optional[torch.Tensor] = None,
                compress_lens_out: Optional[torch.Tensor] = None):
    """
    pseudocode:
        K/V compute is same, so now using k to descrtibe k/v compute.
        cm = torch.nn.Conv2d(1, 1, (32, 1),
                             bias=False,
                             stride=(16, 1),
                             padding=(0, 0),
                             dilation=(1, 1))
        out_list = []
        lens_out_list = []
        for (int i = 0; i < batch; i++):
          compress_num = (max_seq_len - compress_length) / compress_stride + 1
          input = k[i, ...].reshape((1, 1, max_seq_len, head_num * head_size_k))
          output = cm(input).reshape((compress_num, head_num * head_size_k)) + weight @ pe_k
          out_list.append(output)
          lens_out_list.append(torch.tensor([compress_num], dtype=torch.int32))
        return (torch.cat(out_list, dim=0), torch.cat(lens_out_list, dim=0))

    Args:
        k (torch.Tensor): The key tensor. Support pad and pack mode.
                          Pad mode Shape: (batch, max_seq_len, head_num, head_size_k).
                          Pack mode Shape: (total_seq_len, head_num, head_size_k),
                          total_seq_len is the sum of seq_len in each batch.
                          Seq_len in each batch can be computed by cu_seq_lens.
        v (torch.Tensor): The value tensor. Support pad and pack mode.
                          Pad mode Shape: (batch, max_seq_len, head_num, head_size_v).
                          Pack mode Shape: (total_seq_len, head_num, head_size_v),
                          total_seq_len is the sum of seq_len in each batch.
                          Seq_len in each batch can be computed by cu_seq_lens.
        cu_seq_lens (optional torch.Tensor): The cusum of seq_lens if k/v is 3-D tensor or pack mode.
                          Shape is (batch+1).
        k_weight (torch.Tensor): The weight for k. Shape is (1, compress_length).
        v_weight (torch.Tensor): The weight for v. Shape is (1, compress_length).
        pe_table_k (torch.Tensor): Positional encoding table for k.
                          Shape is (compress_length, head_num, head_size_k).
        pe_table_v (torch.Tensor): Positional encoding table for v.
                          Shape is (compress_length, head_num, head_size_v).
        max_seq_len (int): The maximum value of seq_len for each batch.
        compress_length (int): The length of compress.
        compress_stride (int): The stride of compress.
        k_out (optional torch.Tensor): The output tensor for k.
                          Pad mode Shape is (batch, compress_num, head_num, head_size_k)
                          Pack mode Shape is (total_compress_num, head_num, head_size_k).
        v_out (optional torch.Tensor): The output tensor for v.
                          Pad mode Shape is (batch, compress_num, head_num, head_size_v)
                          Pack mode Shape is (total_compress_num, head_num, head_size_v).
        compress_lens_out (optional torch.Tensor): The output compressed num tensor.
                          Shape is (batch,).
    Type:
        k/v/k_weight/v_weight/pe_table_k/pe_table_v/k_out/v_out: torch.float16, torch.bfloat16.
        cu_seq_lens/compress_lens_out: torch.int32.
        max_seq_len/compress_length/compress_stride: int32.

    Return:
        k_out, v_out, compress_lens_out.
    """
    def create_tensor(input: torch.Tensor,
                      PadMode: bool,
                      compress_num: int,
                      batch: int,
                      head_num: int,
                      head_size: int):
        if PadMode:
            return torch.empty(batch, compress_num, head_num, head_size, dtype=input.dtype, device=input.device)
        else:
            return torch.empty(compress_num, head_num, head_size, dtype=input.dtype, device=input.device)

    isPad = cu_seq_lens is None
    batch = k.shape[0] if isPad else cu_seq_lens.shape[0] - 1
    head_num = k.shape[2] if isPad else k.shape[1]
    head_size_k = k.shape[3] if isPad else k.shape[2]
    head_size_v = v.shape[3] if isPad else v.shape[2]
    if (isPad and max_seq_len < 32) or (not isPad and k.shape[0] < 32):
        return None, None, torch.empty(batch, dtype = torch.int32, device = k.device).fill_(0)
    # pad mode: compress_num is for each batch sequence length
    # pack mode: compress_num is for total batch sequence length
    compress_num = (max_seq_len - compress_length) // compress_stride + 1 if isPad \
                   else (k.shape[0] - compress_length) // compress_stride + 1
    if k_out is None:
       k_out = create_tensor(k, isPad, compress_num, batch, head_num, head_size_k)
    if v_out is None:
       v_out = create_tensor(v, isPad, compress_num, batch, head_num, head_size_v)
    if compress_lens_out is None:
       compress_lens_out = torch.empty(batch, dtype=torch.int32, device = k.device)
    torch.ops.torch_mlu_ops.compress_kv(k, v, cu_seq_lens, k_weight, v_weight,
                                        pe_table_k, pe_table_v, max_seq_len,
                                        compress_length, compress_stride,
                                        k_out, v_out, compress_lens_out)
    return k_out, v_out, compress_lens_out

def concat_block_table(
    first_block_table: torch.Tensor,
    first_context_lens: torch.Tensor,
    second_block_table: torch.Tensor,
    second_context_lens: torch.Tensor,
    new_block_table: Optional[torch.Tensor] = None,
    new_context_lens: Optional[torch.Tensor] = None):
    """
    Concatenate two different block tables, return the concatenated result.
    Math:
        new_context_lens = first_context_lens + second_context_lens
        total_seq = first_context_lens.size(0)
        for i in range(total_seq):
            new_block_table[i, :first_context_lens[i]] = first_block_table[i, :first_context_lens[i]]
            new_block_table[i, first_context_lens[i]:first_context_lens[i]+second_context_lens[i]] = second_block_table[i, :second_context_lens[i]]

    .. note::
        Users must ensure that ``context_lens[i] <= block_table.size(1)`` for all sequences.
        Violating this constraint may cause undefined behavior.

    Args:
        first_block_table (torch.Tensor):
            The first block table of shape `[total_seq, first_max_blkn]`.
        first_context_lens (torch.Tensor):
            The context lens of the first block table of shape `[total_seq,]`.
        second_block_table (torch.Tensor):
            The second block table of shape `[total_seq, second_max_blkn]`.
        second_context_lens (torch.Tensor):
            The context lens of the second block table of shape `[total_seq,]`.
        new_block_table (Optional[torch.Tensor]):
            The new block table of shape `[total_seq, max_new_block_number]`.
            if not None, the max_new_block_number must be large enough for the concatenated block_table
            Default: `None`.
        new_context_lens (Optional[torch.Tensor]):
            The new context lens of shape `[total_seq,]`. Default: `None`.

    Returns:
        new_block_table (torch.Tensor):
            The concatenated block table of shape `[total_seq, max_new_block_number]`.
        new_context_lens (torch.Tensor):
            The new context lens of shape `[total_seq,]`, equals first_context_lens + second_context_lens
    Type:
       INT32

    Note:
        1. All tensors must be contiguous in their last dimension.
        2. All tensors must be on the same MLU device.
        3. All tensors must be int32 dtype.
        4. first_block_table.size(0) must equal first_context_lens.size(0).
        5. second_block_table.size(0) must equal second_context_lens.size(0).
        6. first_block_table.size(0) must equal second_block_table.size(0).
        7. new_block_table.size(0) must equal first_block_table.size(0).
        8. new_block_table.size(1) must be >= first_block_table.size(1) and >= second_block_table.size(1).
        9. All block_table size(1) values must be > 0.
    """
    total_seq = first_context_lens.size(0)
    assert first_block_table.dim() == 2, "first_block_table must be 2d tensor"
    assert second_block_table.dim() == 2, "second_block_table must be 2d tensor"
    if new_context_lens is None:
        new_context_lens = torch.empty(total_seq, dtype=torch.int32, device=first_context_lens.device)
    if new_block_table is None:
        first_max_blkn = first_block_table.size(1)
        second_max_blkn = second_block_table.size(1)
        new_block_table = torch.empty((total_seq, first_max_blkn + second_max_blkn), dtype=torch.int32, device=first_context_lens.device)
    elif new_block_table.numel() > 0:
        assert new_block_table.data_ptr() != second_block_table.data_ptr(), "new_block_table and second_block_table must be different tensors"
    torch.ops.torch_mlu_ops.concat_block_table(first_block_table, first_context_lens, second_block_table, second_context_lens,
                                                new_block_table, new_context_lens)
    return (new_block_table, new_context_lens)

def moe_all2all_create(dispatch_token_byte: int,
                       combine_token_byte: int,
                       max_expert_num: int,
                       max_token_num: int,
                       rank: int,
                       nrank: int):
    """
    Create the handle of MOE All-to-All communication.
    API call order:
        1.Call torch_mlu_ops.moe_all2all_create(...) to obtain the CNCLEP handle and buffer tensor for All-to-All communication. Only needs to be done once.
        2.Gather all_exchange_info by performing an All-Gather operation on exchange_info across nrank processes. Only needs to be done once.
        3.Call torch.distributed.barrier() to ensure step 2 finish. Only needs to be done once.
        4.Call torch_mlu_ops.moe_all2all_init(...) to configure the all_exchange_info into the handle. Only needs to be done once.
        5.Call torch_mlu_ops.moe_all2all_dispatch(...) to route tokens to their designated experts.
        6.Call torch_mlu_ops.moe_all2all_combine(...) to restore tokens to their original locations.
        7.Call torch_mlu_ops.moe_all2all_destroy(...) to release the CNCLEP handle. Only needs to be done once.

    Args:
        dispatch_token_byte (int): Byte size of a single token for dispatch All-to-All operation.
        combine_token_byte (int): Byte size of a single token for combine All-to-All operation.
        max_expert_num (int): Maximum number of experts participating in the All-to-All operation.
        max_token_num (int): Maximum number of tokens to be processed.
        rank (int): Rank ID of the current process [0~nrank-1].
        nrank (int): Total number of processes in the distributed group.

    Return:
        A tuple of (handle, exchange_info_size, exchange_info, dispatch_send, dispatch_recv, combine_send and combine_recv).
        handle: The CNCLEP handle with type of integer.
        exchange_info_size: The size of exchange_info.
        exchange_info: CPU tensor, shape is [exchange_info_size], and data type is torch.int8.
        dispatch_send: MLU tensor, shape is [max_token_num * dispatch_token_byte], and data type is torch.int8.
        dispatch_recv: MLU tensor, shape is [nrank * max_token_num * dispatch_token_byte], and data type is torch.int8.
        combine_send: MLU tensor, shape is [max_token_num * combine_token_byte], and data type is torch.int8.
        combine_recv: MLU tensor, shape is [nrank * max_token_num * combine_token_byte], and data type is torch.int8.
    """
    place_holder = torch.tensor([], device="mlu")
    output0, exchange_info, dispatch_send, dispatch_recv, combine_send, combine_recv = torch.ops.torch_mlu_ops.moe_all2all_create(dispatch_token_byte, combine_token_byte, max_expert_num, max_token_num, rank, nrank, place_holder)
    handle = output0[0].item()
    exchange_info_size = output0[1].item()
    return (handle, exchange_info_size, exchange_info, dispatch_send, dispatch_recv, combine_send, combine_recv)

def moe_all2all_init(handle: int,
                     all_exchange_info: torch.Tensor) -> None:
    """
    Set the all_exchange_info into the CNCLEP handle.

    Args:
        handle (int): CNCLEP handle.
        all_exchange_info (torch.Tensor): CPU tensor containing aggregated exchange information from all nrank processes.

    Return:
        None.
    """
    place_holder = torch.tensor([], device="mlu")
    torch.ops.torch_mlu_ops.moe_all2all_init(handle, all_exchange_info, place_holder)

def moe_all2all_dispatch(handle: int,
                         token_byte: int,
                         token_num: int,
                         send_layout: torch.Tensor,
                         send_token_num: torch.Tensor,
                         recv_layout: torch.Tensor,
                         recv_token_num: torch.Tensor,
                         send_token: Optional[torch.Tensor] = None,
                         recv_token: Optional[torch.Tensor] = None,
        ):
    """
    Dispatch tokens to their designated experts.

    Args:
        handle (int): CNCLEP handle.
        token_byte (int): Byte size of a single token.
        token_num (int): The number of tokens to be processed in current operation.
        send_layout (torch.Tensor): The offset and token number of each rank, which is the output of torch_mlu_ops.moe_all2all_gen_send_layout(token_count, nrank).
                                    The token_count is generated by moe_gen_idx. Shape is [nrank, 2].
        send_token_num (torch.Tensor): The number of token send to each expert. Shape is [max_expert_num].
        recv_layout (torch.Tensor): The offset and token number from peer ranks. Shape is [nrank, 2].
        recv_token_num (torch.Tensor): The expected number of tokens to receive. Shape is [max_expert_num].
        send_token (torch.Tensor) optional: The tokens to dispatch. If not given, will use dispatch_send created by moe_all2all_create().
        recv_token (torch.Tensor) optional: Buffer for receiving tokens. If not given, will use dispatch_recv created by moe_all2all_create().

    Type:
        send_layout: int32.
        send_token_num: int32.
        recv_layout: int32.
        recv_token_num: int32.

    Return:
        None.
    """
    torch.ops.torch_mlu_ops.moe_all2all_dispatch(handle, token_byte, token_num, send_layout, send_token_num, recv_layout, recv_token_num, send_token, recv_token)

def moe_all2all_combine(handle: int,
                        token_byte: int,
                        token_num: int,
                        send_src_layout: torch.Tensor,
                        send_dst_layout: torch.Tensor,
                        send_token: Optional[torch.Tensor] = None,
                        recv_token: Optional[torch.Tensor] = None,
        ):
    """
    Restore tokens to their original process.

    Args:
        handle (int): CNCLEP handle.
        token_byte (int): Byte size of a single token.
        token_num (int): The number of tokens to receive.
        send_src_layout (torch.Tensor): The offset and token number of each rank, which is the output of torch_mlu_ops.moe_all2all_gen_send_layout(recv_token_num, nrank). Shape is [nrank, 2].
        send_dst_layout (torch.Tensor): The expected receive pattern from peer ranks. Shape is [nrank, 2].
        send_token (torch.Tensor) optional: The tokens to dispatch. If not given, will use combine_send created by moe_all2all_create().
        recv_token (torch.Tensor) optional: Buffer for receiving tokens. If not given, will use combine_recv created by moe_all2all_create().

    Type:
        send_src_layout: int32.
        send_dst_layout: int32.

    Return:
        None.
    """
    torch.ops.torch_mlu_ops.moe_all2all_combine(handle, token_byte, token_num, send_src_layout, send_dst_layout, send_token, recv_token)

def moe_all2all_destroy(handle: int) -> None:
    """
    Release the CNCLEP handle.

    Args:
        handle (int): CNCLEP handle.

    Return:
        None.
    """
    place_holder = torch.tensor([], device="mlu")
    torch.ops.torch_mlu_ops.moe_all2all_destroy(handle, place_holder)

def fused_indexer_k(x: torch.Tensor,
                    wk: torch.Tensor,
                    wproj: torch.Tensor,
                    sin_table: torch.Tensor,
                    cos_table: torch.Tensor,
                    position_id: torch.Tensor,
                    slot_mapping: torch.Tensor,
                    k_cache: torch.Tensor,
                    k_cache_scale: Optional[torch.Tensor] = None,
                    hadamard_matrix: Optional[torch.Tensor] = None,
                    interleaved: bool = True,
                    gamma: Optional[torch.Tensor] = None,
                    beta: Optional[torch.Tensor] = None,
                    eps: float = 1e-6):
      """
      Perform wk(x), layernorm, rope, wproj(x) and quant to paged k_cache.

      Math:
          k = matmul(x, k)
          head_weights = matmul(x, wproj)
          k.layernorm()
          k[..., 0:rope_dim].rotary_embedding(interleaved)
          if has_hadamard:
              k = matmul(k, hadamard_matrix)
          if quant:
              scale = max(k.abs, dim=-1)
              k = k / scale.to(int8)
          for i in range(m):
              block_id = slot_mapping[i] // block_size
              block_offset = slot_mapping[i] % block_size
              k_cachep[block_id, :, block_offset, :] = k[i]

      Args:
          x (torch.Tensor): Shape is (m, dim).
          wk (torch.Tensor): Shape is (head_size, dim).
          wproj (torch.Tensor): Shape is (head_num, dim).
          sin_table (torch.Tensor):  The rotary sin table tensor. Shape is (rotary_seq, rope_dim).
          cos_table (torch.Tensor):  The rotary cos table tensor. Shape is (rotary_seq, rope_dim).
          position_id (torch.Tensor): The rotary seq_len offset of each m. Shape is (m).
          slot_mapping (torch.Tensor): The slot_mapping tensor. Shape is (m).
          k_cache (torch.Tensor): The cache tensor. Shape is (block_num, 1, block_size, head_size).
          k_cache_scale (torch.Tensor, optional): The cache_scale tensor. Shape is (block_num, 1, block_size).
          hadamard_matrix (torch.Tensor, optional): The hadamard_matrix tensor for rotate_activation. Shape is (head_size, head_size).
          interleaved (bool, optional): Describing the rope mode, Cross mode if True else fold mode.
          gamma (torch.Tensor, optional): The weight of layernorm. Shape is (head_size).
          beta (torch.Tensor, optional): The bias of layernorm. Shape is (head_size).
          eps (float, optional): The eps of laynorm.

      Type:
          x: half, bfloat16
          wk: same as x
          wproj: same as x
          sin_table: same as x
          cos_table: same as x
          position_id: int32
          slot_mapping: int32
          k_cache: half, bfloat16, int8
          k_cache_scale: float
          hadamard_matrix: same as x
          gamma: float
          beta: float

      Return:
          head_weights (torch.Tensor): Shape is (m, head_num)
      """
      head_weights = torch.empty((x.size(0), wproj.size(0)), dtype=x.dtype, device=x.device)
      torch.ops.torch_mlu_ops.fused_indexer_k(x, wk, wproj, sin_table, cos_table, position_id, slot_mapping, head_weights, k_cache, k_cache_scale, hadamard_matrix, interleaved, gamma, beta, eps)
      return (head_weights, k_cache, k_cache_scale) if k_cache_scale is not None else (head_weights, k_cache)

def masked_indexer_select_paged_kv(query: torch.Tensor,
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
                                   kv_cache_block_table_offset: Optional[torch.Tensor] = None):
    """
    Calculates the matmul of q and k, apply causal mask, applies RELU, sums on the head_num dim, then
    generates the new selected block_table according to topk important logits.
    this op only support causal mask

    Math:
       for each batch:
       k = k.repeat(head_num, 1, 1) # shape is head_num, seq_k, head_size
       qk = bmm(q, k)
       qk = qk.relu() # shape is head_num, seq_q, seq_k

       if q_scale is not None
         weights = weights * q_scale
       weights = weights * softmax_scale  # shape is seq_q, head_num, 1
       logits = qk.permute(1, 0, 2) * weights  # shape is seq_q, head_num, seq_k
       logits = logits.sum(dim=1) # shape is seq_q, seq_k , sum along head_num
       if k_scale_cache is not None:
          logits = logits * k_scale_cache
       _, selected = topk(logits[:causal_mask], k=index_topk)
       sparse_block_table = kv_cache_block_table[selected]
       sparse_context_lens = causal_mask > index_topk ? index_topk : causal_mask for each seq_q
       if compress_ratio is bigger than 1, causal mask is compressed_causal_mask

    Args:
       query(torch.Tensor): Query tensor.
                            If is_prefill is True, shape is [total_seq_q,head_num,head_size], otherwise shape is either [batch, seq_q, head_num, head_size] or [total_seq_q,head_num,head_size].
       k_cache(torch.Tensor): Paged cache of k tensor.
                              If is_prefill is True, shape is [total_seq_k, head_size], otherwise shape is [total_k_block_num, 1, k_block_size, head_size].
                              Increasing k_block_size might speedup computation.
       weights(torch.Tensor): Query weights, to be multiplied into q_scale.
                              If is_prefill is True, shape is [total_seq_q, head_num, 1], otherwise shape is [batch, seq_q, head_num, 1] or [total_seq_q, head_num, 1] according to shape of query.
                              The last two dimensions must be contiguous; the first one or two dimensions may be non-contiguous.
       kv_cache_block_table(torch.Tensor): Indices for the actual KV cache blocks.
                                            Shape is [batch, kv_cache_max_blkn].
       cu_seq_q_lens(torch.Tensor): The cumsum of seq_q.
                                     Must not be None if is_prefill is True, otherwise could be None.
                                     The First element must be 0.
       cu_seq_k_lens(torch.Tensor): The cumsum of seq_k.
                                     Must not be None if is_prefill is True, otherwise could be None.
                                     The First element must be 0.
                                     If compress_ratio > 1, this is the cumsum of the seq_k BEFORE the compression(divide compress_ratio).
       k_context_lens(torch.Tensor): Context lengths of k, shape is [batch].
                                      Could be None if is_prefill is True, otherwise must not be None.
                                      If compress_ratio > 1, this is the lengths BEFORE the compression(divide compress_ratio).
       k_cache_block_table(torch.Tensor): Indices for k and k_scale cache blocks, used in decoding.
                                          Shape is [batch, k_max_block_num], could be None if is_prefill is True, otherwise must not be None.
       is_prefill(bool): Whether it is prefill.
                         If this is True, k_cache should not be page cached, and shapes are different from when is_prefill is False.
       index_topk(int): The number of topk important logits to be selected.
       kv_cache_block_size(int): Size of each block in kv_cache_block_table, should be 1, or multiple of 16.
                                  If kv_cache_block_size is larger than 1, there is an additional convertion which may have overhead.
       softmax_scale(float): (head_dim ** -0.5) * (n_heads ** -0.5), should be calculated at host.
                            Note: This parameter accepts double (float64) type at interface, but will be converted to float32 internally.
       q_scale(torch.Tensor): Quant-scale of query.
                              If is_prefill is True, shape is: [total_seq_q, head_num, quant_block_num], otherwise shape is [batch, seq_q, head_num, quant_block_num] or [total_seq_q, head_num, quant_block_num], must be None if dtype of query is FP16 or BF16.
                              Required if dtype of query is quantized. quant_block_num is the number of quantization blocks, which is head_size/32=4 for MX quantization and 1 for per-token quantization.
                              Must be positive.
       k_scale_cache(torch.Tensor): Quant-scale of k.
                                    If is_prefill is True, shape is: [total_seq_k, quant_block_num], otherwise shape is [total_k_block_num, 1, k_block_size, quant_block_num].
                                    If Type of k is FP16 or BF16, this must be None.
                                    Required if k_cache is quantized.
                                    Must be positive.
                                    When k_cache and k_scale_cache are sliced from the same tensor (interleaved storage pattern
                                    where head_size k elements are followed by corresponding k_scale elements along the last dimension),
                                    they may be non-contiguous on the last dimension, but all other dimensions must be contiguous.
                                    This interleaved pattern is auto-detected: k_cache and k_scale_cache must share the same storage,
                                    k_cache starts at a lower address, and their stride on the second-to-last dimension must match.
       sparse_block_table(torch.Tensor): Selected block indices from kv_cache_block_table(for KV cache lookup).
                                          If is_prefill is True, shape is [total_seq_q, index_topk], otherwise shape is [batch, seq_q, index_topk].
       sparse_context_lens(torch.Tensor): Valid context_lens of sparse_block_table, shape is [total_seq_q].
                                          If is_prefill is False, total_seq_q = batchnum * seq_q.
       is_score_float(bool): Whether the score is float.
                             If True, the score is float, otherwise the score is bf16.
       compress_ratio(int): compress ratio, default is 1.
                           If compress_ratio is bigger than 1, causal mask is compressed_causal_mask.
                           Must be bigger than 0.
       kv_cache_block_table_offset(torch.Tensor): Shape is (batch).
                                                   For each batch, the real kv_cache_block_table starts from this offset.
                                                   User should guarantee that DIV_UP(kv_cache_block_table_offset, kv_cache_block_size) + the real used block number does not surpass kv_cache_block_table.size(1).
                                                   kv_cache_block_table_offset must be positive.

    User should guarantee that real length of Q must not be greater than real length of K, in which case no error would be thrown but it does not make sense for the causalmask.
    When is_prefill is False, the sequence length in each query is usually smaller than 10, bigger sequence length will cause a warning, in case of miss-using

    Note:
       head_num must be 64 or 32.
       head_size must be 128.
       All input tensors except weights, k_cache, and k_scale_cache must be contiguous.
       Weights may be non-contiguous provided the last two dimensions (head_num, 1) are contiguous.
       k_cache and k_scale_cache support non-contiguous on the last dimension when they are sliced
       from the same underlying tensor (interleaved storage: head_size k elements followed by 1 or 4
       k_scale elements). In this case all dimensions except the last must be contiguous.
       If k_cache and k_scale_cache are not sliced from the same tensor, they must be contiguous.


    Type:
       query: BF16, FP16, INT8, FLOAT8_E4M3FN
       k_cache: BF16, FP16, INT8, FLOAT8_E4M3FN
       weights: BF16, FP32
       kv_cache_block_table: INT32
       cu_seq_q_lens: INT32
       cu_seq_k_lens: INT32
       k_context_lens: INT32
       k_cache_block_table: INT32
       index_topk: INT32
       kv_cache_block_size: INT32
       q_scale: FP32
       k_scale_cache: FP32
       sparse_block_table: INT32
       sparse_context_lens: INT32
       is_prefill: BOOL
       softmax_scale: FP32
       is_score_float: BOOL
       compress_ratio: INT32
       kv_cache_block_table_offset: INT32


    """
    if sparse_block_table is None:
        sparse_block_table = torch.empty(query.shape[:-2] + (index_topk,), dtype=torch.int32, device=query.device)
    if sparse_context_lens is None:
        sparse_context_lens = torch.empty(query.shape[:-2], dtype=torch.int32, device=query.device).flatten()
    torch.ops.torch_mlu_ops.masked_indexer_select_paged_kv(query, k_cache, weights, kv_cache_block_table,
           cu_seq_q_lens, cu_seq_k_lens, k_context_lens,
           k_cache_block_table, is_prefill, index_topk, kv_cache_block_size, softmax_scale, q_scale, k_scale_cache,
           sparse_block_table, sparse_context_lens, is_score_float, compress_ratio, kv_cache_block_table_offset)
    return (sparse_block_table,  sparse_context_lens)

def fused_masked_mul_topk_select_paged_kv(
    q: torch.Tensor,
    km: torch.Tensor,
    cu_seq_lens_q: torch.Tensor,
    origin_block_table: torch.Tensor,
    origin_context_lens: torch.Tensor,
    max_seq_q: int,
    max_seq_k: int,
    block_size_q: int,
    block_size_k: int,
    ratio: float,
    recent_window: int,
    window_included: bool = True):
    """
    Calculate group_gemm for label_q and k, then generates the new selected block_table according to topk important logits.

    Math:
        1. Global computation:
           - Perform group_gemm between label_q and km

        2. Sparse selection:
           - Determine top-k blocks to keep based on the ratio parameter
           - Always include recent blocks according to recent_window
           - Combine local and important blocks in the selection

        3. Output generation:
           - Construct sparse block table from selected blocks
           - Compute effective context lengths for each query block
           - Generate new cumulative query length offsets

    Args:
        q(torch.Tensor): Query tensor. shape is [total_q, head_num_q, head_size].
        km(torch.Tensor): shape is [batch, head_num_k, max_k, 2 * head_size]. km is the concatenation of k_max and k_min in the lowest dimension.
        cu_seq_lens_q(torch.Tensor): Cumulative length of label_q sequence. Shape is [batch+1, ].
        origin_block_table(torch.Tensor): Original complete KV block table. Shape is [batch, max_block_num_k].
        origin_context_lens(torch.Tensor): Original context length. Shape is [batch, ].
        max_seq_q(int): The maximum query sequence length.
        max_seq_k(int): The maximum key sequence length.
        block_size_q(int): The block size for q.
        block_size_k(int): The block size for k.
        ratio(float): The ratio of blocks to keep when performing top-k selection.
        recent_window(int): The window size for local attention (number of recent blocks to always include.
        window_included(bool): Flag indicating whether the recent window blocks are included in the sparse_ratio calculation.

    Type:
        q: BF16, FP16
        km: BF16, FP16
        cu_seq_lens_q: INT32
        origin_block_table: INT32
        origin_context_lens: INT32
        max_seq_q: INT32
        max_seq_k: INT32
        block_size_q: INT32
        block_size_k: INT32
        ratio: FP32
        recent_window: INT32
        window_included: BOOL

    Return:
        sparse_block_table (torch.Tensor): Sparsified KV block table. Shape is (batch*max_q, head_num_q, max_block_num_k)
        sparse_context_lens (torch.Tensor): Total length of Key elements attended by each Q block. Shape is (batch*max_q, )
        new_cu_seq_lens_q (torch.Tensor): Cumulative length of Q sequence after block-wise division. Shape is (batch*max_q+1, )
    """
    batch, head_num_k, max_k, _ = km.size()
    head_num_q, head_size = q.size(1), q.size(-1)
    max_q = (max_seq_q + block_size_q - 1) // block_size_q

    assert not q.numel() == 0, f"q should not be empty."
    assert not km.numel() == 0, f"k should not be empty."

    sparse_block_table = torch.zeros(batch*max_q, head_num_q, origin_block_table.size(-1), device=origin_block_table.device, dtype=torch.int32)
    sparse_context_lens = torch.zeros(batch*max_q, device=origin_block_table.device, dtype=torch.int32)
    new_cu_seq_lens_q = torch.zeros(batch*max_q+1, device=origin_block_table.device, dtype=torch.int32)

    lable_q_index = torch.empty(batch * max_q, device=origin_block_table.device, dtype=torch.int32)
    index_num = torch.empty(1, device=origin_block_table.device, dtype=torch.int32)
    gmm_m_list = torch.empty(batch * head_num_q, device=origin_block_table.device, dtype=torch.int32)
    gmm_n_list = torch.empty(batch * head_num_q, device=origin_block_table.device, dtype=torch.int32)
    gmm_a_ptrs = torch.empty(batch * head_num_q, device=origin_block_table.device, dtype=torch.int64)
    gmm_b_ptrs = torch.empty(batch * head_num_q, device=origin_block_table.device, dtype=torch.int64)
    gmm_d_ptrs = torch.empty(batch * head_num_q, device=origin_block_table.device, dtype=torch.int64)
    gmm_lda = torch.empty(batch * head_num_q, device=origin_block_table.device, dtype=torch.int32)
    label_q = torch.empty(batch * max_q, head_num_q, head_size * 2, device=origin_block_table.device, dtype=q.dtype)
    qk_logit = torch.empty(batch * head_num_q * max_q * max_k, device=origin_block_table.device, dtype=torch.float32)

    torch.ops.torch_mlu_ops.gen_label_q_idx(cu_seq_lens_q, origin_context_lens, label_q, km, qk_logit, lable_q_index, index_num,
                                            new_cu_seq_lens_q, gmm_m_list, gmm_n_list, gmm_a_ptrs, gmm_b_ptrs, gmm_d_ptrs,
                                            gmm_lda, max_seq_q, max_seq_k, block_size_q, block_size_k)

    torch.ops.torch_mlu_ops.fused_gather_clamp_concat(label_q, q, lable_q_index, index_num)

    allow_tf32 = False
    if torch.backends.mlu.matmul.fp32_precision == 'tf32' or \
            (torch.backends.mlu.matmul.fp32_precision == 'none' and torch.backends.fp32_precision == 'tf32'):
        allow_tf32 = True
    torch.ops.torch_mlu_ops.variable_n_group_gemm(gmm_a_ptrs, gmm_b_ptrs, gmm_d_ptrs, gmm_m_list, gmm_n_list, gmm_lda,
                                                  max_q, max_k, head_size * 2, _torchDtype2Str(q.dtype), _torchDtype2Str(torch.float32),
                                                  allow_tf32)

    torch.ops.torch_mlu_ops.masked_topk_select_block_table(qk_logit, origin_block_table, cu_seq_lens_q, origin_context_lens,
                                                           sparse_context_lens, sparse_block_table, max_seq_q, max_seq_k,
                                                           block_size_q, block_size_k, recent_window, ratio, window_included)

    return sparse_block_table, sparse_context_lens, new_cu_seq_lens_q

def transpose_all2all(cncl_comm: int,
                      pre_num_block: int,
                      pre_block_count: int,
                      post_num_block: int,
                      post_block_count: int,
                      send: torch.Tensor,
                      recv: torch.Tensor) -> torch.Tensor:
    """
    all_to_all with pre-transpose/post-transpose.

    Args:
        cncl_comm (int): The handle of CnclComm.
        pre_num_block (int): The value is token1 if has pre-transpose, else is 0.
        pre_block_count (int): The value is hidden_size1 if has pre-transpose, else is 0.
        post_num_block (int): The value is token2 if has post-transpose, else is 0.
        post_block_count (int): The value is hidden_size2 if has post-transpose, else is 0.
        send (torch.Tensor): Shape is [token1, nrank, hidden_size1] if has pre-transpose, else shape is [nrank, token1, hidden_size1].
        recv (torch.Tensor): Shape is [token2, nrank, hidden_size2] if has post-transpose, else shape is [nrank, token2, hidden_size2].

    Returns:
        torch.Tensor: recv

    Note:
        If has pre-transpose or post-transpose only, token1 must be equal to token2, and hidden_size1 must be equal to hidden_size2.
        If have both pre-transpose and post-transpose, token1 * hidden_size1 must be equal to token1 * hidden_size2.
        Only support transpose(0, 1), don't support transpose(0, 2) or transpose(1, 2).
    """
    torch.ops.torch_mlu_ops.transpose_all2all(cncl_comm,
                                              pre_num_block,
                                              pre_block_count,
                                              post_num_block,
                                              post_block_count,
                                              send,
                                              recv)
    return recv

def convert_vertical_slash_index(
    seqlens: torch.Tensor,
    ctxlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    max_seqlen_q: int,
    block_size_M: int,
    block_size_N: int
) -> Tuple[torch.Tensor]:
    """
    Preprocess vertical and slash indexes for efficient sparse attention computation in Vertical-Slash pattern.

    Args:
        seqlens (torch.Tensor): Sequence lengths of query, shape is (batch,)
        ctxlens (torch.Tensor): Context lengths (K/V) per batch, shape (batch,)
        vertical_indexes (torch.Tensor): Vertical line intercepts on K-axis, shape [batch, head, nnz_v]
        slash_indexes (torch.Tensor): Slash line intercepts (pre-processed as total_len-slash_idx-1), shape [batch, head, nnz_s]
        max_seqlen_q (int): Maximum query sequence length
        block_size_M (int): Block size for query dimension
        block_size_N (int): Block size for key dimension

    Returns:
        Tuple[torch.Tensor]: A tuple containing four tensors:
            - block_count: Number of slash blocks per query block row, shape [batch, head, ceil_div(max_seqlen_q, block_size_M)]
            - block_offset: Block offsets for slash patterns, shape [batch, head, ceil_div(max_seqlen_q, block_size_M), nnz_s]
            - column_count: Number of vertical columns per query block, shape [batch, head, ceil_div(max_seqlen_q, block_size_M)]
            - column_index: Vertical column indices, shape [batch, head, ceil_div(max_seqlen_q, block_size_M), nnz_v]

    Note:
        - Input slash_indexes are pre-processed as (total_len - original_slash_idx - 1)
        - Both input indexes must be strictly increasing
        - Designed for MInference's Vertical-Slash sparse attention pattern
    """
    return torch.ops.torch_mlu_ops.convert_vertical_slash_index(
            seqlens,
            ctxlens,
            vertical_indexes,
            slash_indexes,
            max_seqlen_q,
            block_size_M,
            block_size_N
    )

def hamming_score(query_code: torch.Tensor,
                  key_codes: torch.Tensor,
                  block_table_opt: Optional[torch.Tensor],
                  seq_len: torch.Tensor,
                  max_seq_len: int,
                  sink: int,
                  recent: int) -> torch.Tensor:
    """
    Calculate Hamming distance between query hash codes and key hash codes for approximate attention computation.

    Math:
        For each batch b:
        For each query position i:
        For each key position j:
            XOR_result = query_code[b, 0, head, dim] XOR key_codes[b, j, 0, dim]
            # block mode: key_codes indexed by block_table
            distance[i, j] = popcount(XOR_result) # count set bits in binary representation

        # Apply masking
        distance[i, :sink] = 0  # forced attention to sink tokens
        distance[i, seq_len[b]-recent:seq_len[b]] = 0  # forced attention to recent tokens

    Args:
        query_code(torch.Tensor): Query hash codes. shape is [batch, 1, head_num_q, hash_dim].
        key_codes(torch.Tensor): Key hash codes.
           - In block mode: shape is [num_blocks, 1, block_size, hash_dim]
           - In non-block mode: shape is [batch_size, max_seq_len, 1, hash_dim]
        block_table_opt(Optional[torch.Tensor]): Optional sparse block table for KV cache.
           Shape is [batch, max_num_block_per_seq]. None indicates non-block mode.
        seq_len(torch.Tensor): Actual sequence length for each example in batch. Shape is [batch, ].
        max_seq_len(int): Maximum sequence length (output size).
        sink(int): Number of sink tokens to force-include from the beginning (distance set to 0).
        recent(int): Number of recent tokens to force-include from the end (distance set to 0).

    Type:
        query_code: INT32
        key_codes: INT32
        block_table_opt: INT32 (optional)
        seq_len: INT32
        max_seq_len: INT32
        sink: INT32
        recent: INT32

    Return:
        output(torch.Tensor): Hamming distance scores. Shape is [batch, 1, max_seq_len], dtype is FP16.

    Note:
        - In block mode, key_codes must have shape [num_blocks, 1, block_size, hash_dim]
        - In non-block mode, key_codes must have shape [batch_size, max_seq_len, 1, hash_dim]
        - query_code must have size 1 at dimension 1: [batch, 1, head_num_q, hash_dim]
        - Sink tokens (first positions) and recent tokens (last positions) get distance of 0
    """

    output = torch.ops.torch_mlu_ops.hamming_score(query_code, key_codes, block_table_opt,
                                                   seq_len, max_seq_len, sink, recent)
    return output

def solve_tril(input: torch.Tensor,
               output: torch.Tensor = None,
               cu_seqlens: torch.Tensor = None,
               out_dtype:torch.dtype = torch.float):
    """
    Compute the inverse of the matrix I + input.
    input should be strictly lower triangular, i.e., input.triu() == 0.
    I is a identity matrix.

    Args:
        input(torch.Tensor): The tensor to be inverse. shape is [total_seqlen, head_num, dim] or [batch, seqlen, head_num, dim].
        output(torch.Tensor): The output tensor, shape is the same as input.
        cu_seqlens(torch.Tensor): The sequence length of each batch in pack mode.
        out_dtype(torch.dtype): The output dtype when output is None, support BF16, FP16, FP32. Defaults to torch.float.
    Type:
        input: FP32
        output: BF16, FP16, FP32
        cu_seqlens: INT32

    Return:
        (I + input)^-1 with the same shape as input.
    """
    if output is None:
        output = torch.empty(input.shape, device=input.device, dtype=out_dtype)
    torch.ops.torch_mlu_ops.solve_tril(input, output, cu_seqlens)
    return output

def fused_mul_reduce_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Fused multiply and reduce sum operation for weighted tensor aggregation.

    MATH:
    Given x(n, t, c) and w(n, t)
    output = (x * w.unsqueeze(-1)).sum(dim=1,  keepdim=False)

    Args:
        x(torch.Tensor): Input tensor with shape (bs, seq_len, dim), where bs is batch size,
            seq_len is sequence length, dim is channel dimension.
        w(torch.Tensor): Weight tensor with shape (bs, seq_len), must have same batch size
            and sequence length as x.

    Type:
        x: BF16, FP16, w: FP32

    Constraints:
        0 < seq_len <= 8, 0 < dim <= 4096.

    Returns:
        torch.Tensor: Output tensor with shape (bs, dim).
    """
    return torch.ops.torch_mlu_ops.fused_mul_reduce_sum(x, w)

def fused_compress_single_kv(kv: torch.Tensor,
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
    """

    Args:
        kv (torch.Tensor): Shape is [T, coff_dim].
        score (torch.Tensor): Shape is [T, coff_dim].
        position (torch.Tensor): Shape is [T].
        ape (torch.Tensor): Shape is [ratio, coff_dim].
        gamma (torch.Tensor): Shape is [head_dim].
        sin (torch.Tensor): Shape is [table_len, rope_dim].
        cos (torch.Tensor): Shape is [table_len, rope_dim].
        hadamard_matrix (torch.Tensor, optional): Shape is [head_dim, head_dim].
        slot_mapping (torch.Tensor): Shape is [T].
        kv_cache (torch.Tensor): Shape is [block_num, block_size, head_dim].
        kv_cache_scale (torch.Tensor, optional): Shape is [block_num, block_size].
        eps (float): The eps of normalization.
        overlap (bool): Whether to overlap.
        state_cache (torch.Tensor): Shape is [state_blocks, state_block_size, 2 * state_width].
        state_bt (torch.Tensor): State block table, Shape is [batch, max_state_blocks].
        state_width (int): coff * head_dim.
        state_block_size (int): Size of each block in the state block table.
        cu_query_len (torch.Tensor): The cumulative query length for each batch. Shape is [bsz+1,].
        K (int): The extra prediction token num when using multi-token prediction.

    Type:
        kv: BF16, FP32
        score: same as kv
        position: INT32
        ape: FP32
        gamma: BF16, FP32
        sin: BF16, FP32
        cos: BF16, FP32
        hadamard_matrix: same as kv_cache
        slot_mapping: INT32
        kv_cache: BF16, FP32
        kv_cache_scale: FP32
        state_cache: FP32
        state_bt: INT32
        cu_query_len: INT32

    Returns:
        Only support inplace outputs, include state_cache, kv_cache, kv_cache_scale

    Note:
        coff = overlap + 1
        state_width = coff * head_dim
        R = coff * ratio
        T: all token num
    """
    torch.ops.torch_mlu_ops.fused_compress_single_kv(kv, score, position, ape, gamma, sin, cos,
                            hadamard_matrix, slot_mapping, kv_cache, kv_cache_scale, eps, overlap,
                            state_cache, state_bt, state_width, state_block_size, cu_query_len, K)
    return state_cache, kv_cache, kv_cache_scale

def fused_rmsnorm_rope_store_paged_cache(qkv: torch.Tensor,
                                         k_cache: torch.Tensor,
                                         v_cache: torch.Tensor,
                                         sin_cache: torch.Tensor,
                                         cos_cache: torch.Tensor,
                                         position_id: torch.Tensor,
                                         k_gamma: torch.Tensor,
                                         q_gamma: torch.Tensor,
                                         slot_mapping: torch.Tensor,
                                         k_scale: Optional[torch.Tensor] = None,
                                         v_scale: Optional[torch.Tensor] = None,
                                         eps: float = 1e-5):
    """
    Fused RMSNorm + RoPE + Store to Paged KV Cache operation.

    This operator performs the following operations in a single kernel:
    1. RMSNorm: Apply RMSNorm to Q (using q_gamma) and K (using k_gamma)
    2. RoPE: Apply Rotary Position Embedding to Q and K using sin_cache and cos_cache
    3. Store: Store the transformed K and V into paged cache (k_cache, v_cache)
       according to slot_mapping

    When k_scale and v_scale are provided, K and V are quantized using per-channel
    dynamic quantization before storing to cache.

    Math:
        For Q RMSNorm:  q = qkv[..., :q_heads*head_size] / rms(qkv) * q_gamma
                        where rms = sqrt(mean(qkv^2) + eps)
        For K RMSNorm:  k = qkv[..., q_heads*head_size:] / rms(qkv) * k_gamma
        For RoPE:       q = q * cos + rotate(q) * sin
                        k = k * cos + rotate(k) * sin

    Args:
        qkv (torch.Tensor): Input tensor containing Q, K, V concatenated.
            Shape: [batch, seq_len, q_heads + 2*kv_heads, head_size]
        k_cache (torch.Tensor): K cache tensor for paged storage.
            Shape: [num_blocks, kv_heads, block_size, head_size]
        v_cache (torch.Tensor): V cache tensor for paged storage.
            Shape: [num_blocks, kv_heads, block_size, head_size]
        sin_cache (torch.Tensor): Sinusoidal cache for RoPE. Shape: [max_seq_len, rope_dim]
        cos_cache (torch.Tensor): Cosine cache for RoPE. Shape: [max_seq_len, rope_dim]
        position_id (torch.Tensor): Position IDs for each token. Shape: [batch,]
        k_gamma (torch.Tensor): K RMSNorm gamma parameter. Shape is [head_size,].
        q_gamma (torch.Tensor): Q RMSNorm gamma parameter. Shape is [head_size,].
            Could be None if not do RMSNorm for q.
        slot_mapping (torch.Tensor): Maps tokens to cache slots. Shape: [batch * seq_len,]
            Values >= 0 indicate valid slots, -1 indicates invalid slots.
        k_scale (Optional[torch.Tensor]): K quantization scale.
            Shape of [num_blocks, kv_heads, block_size] for dynamic per-token quantization.
            Shape of [kv_heads, head_size] for static per-channel quantization.
            None for no quantization.
        v_scale (Optional[torch.Tensor]): V quantization scale. Shape: [num_blocks, kv_heads, block_size].
        eps (float): Epsilon for RMSNorm numerical stability. Default: 1e-5.

    Returns:
        Tuple containing:
            - qkv (torch.Tensor): RMSNorm and RoPE processed Q tensor.
            - k_cache (torch.Tensor): K cache with processed K stored.
            - v_cache (torch.Tensor): V cache with processed V stored.
            - k_scale (torch.Tensor, optional): Updated K scale if quantization enabled.
            - v_scale (torch.Tensor, optional): Updated V scale if quantization enabled.
    """
    torch.ops.torch_mlu_ops.fused_rope(qkv, k_cache, v_cache, None, None,
        sin_cache, cos_cache, position_id, k_gamma, None, k_scale, v_scale,
        None, None, None, None, None, None, slot_mapping, None, "rmsnorm", 0, eps, q_gamma)
    out = (qkv, k_cache, v_cache)
    if k_scale is not None:
        out += (k_scale, v_scale)
    return out

def update_compressor_states(kv_state: torch.Tensor,
                             score_state: torch.Tensor,
                             accept_tokens: torch.Tensor,
                             batch_to_kv_state: torch.Tensor,
                             positions: torch.Tensor,
                             cu_query_len: torch.Tensor,
                             overlap: bool,
                             K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update compressor states for KV cache compression.

    MATH:
    For each batch i, the update process is:
      start_pos = positions[cu_query_len[i]]
      end_pos = start_pos + accept_tokens[i]
      state_idx = batch_to_kv_state[i]

    Compression triggers when both conditions are met:
      - end_pos >= (overlap ? 2*ratio : ratio)
      - (start_pos // ratio) != (end_pos // ratio) OR (start_pos % ratio == 0)

    When compressed:
      - overlap=true: length = end_pos - start_pos + (start_pos % ratio)
      - overlap=false: length = end_pos % ratio
      - kv_state[state_idx, 0:length] = kv_state[state_idx, ratio:ratio+length]
      - score_state[state_idx, 0:length] = score_state[state_idx, ratio:ratio+length]

    Args:
        kv_state(torch.Tensor): The kv state tensor. Shape is [max_batch, (overlap+1)*ratio + K, dim].
        score_state(torch.Tensor): The score state tensor. Shape is [max_batch, (overlap+1)*ratio + K, dim].
        accept_tokens(torch.Tensor): The number of accepted tokens for each batch. Shape is [bsz,].
        batch_to_kv_state(torch.Tensor): Mapping from batch index to kv_state index. Shape is [bsz,].
        positions(torch.Tensor): The position indices for each batch. Shape is [bsz,].
        cu_query_len(torch.Tensor): The cumulative query length for each batch. Shape is [bsz+1,].
        overlap(bool): Whether to use overlap mode.
        K(int): Additional K elements in the state tensors (1-4).

    Type:
        kv_state, score_state: FP32, BF16, FP16
        accept_tokens, batch_to_kv_state, positions, cu_query_len: INT32

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The updated kv_state and score_state.

    Note:
        - kv_state and score_state are modified in-place.
        - ratio is derived from kv_state shape: (second_dim - K) / (overlap + 1).
    """
    torch.ops.torch_mlu_ops.update_compressor_states(kv_state,
                                                    score_state,
                                                    accept_tokens,
                                                    batch_to_kv_state,
                                                    positions,
                                                    cu_query_len,
                                                    overlap,
                                                    K)
    return (kv_state, score_state)

def fused_mhc_post(x: torch.Tensor,
                   residual: torch.Tensor,
                   post: torch.Tensor,
                   comb: torch.Tensor,
                   compute_rms: bool,
                   eps: float,
                   output: torch.Tensor = None,
                   output_rms: Optional[torch.Tensor] = None):
    """
    Apply post processing in mHC network.

    Math:
        output = post * x + (comb * residual).sum(dim=1)
        output_rms = rsqrt(output.square().mean(dim=-1) + eps)

    Args:
        x (torch.Tensor): Shape is [n, dim].
        residual (torch.Tensor): Shape is [n, hc_mult, dim].
        post (torch.Tensor): Shape is [n, hc_mult].
        comb (torch.Tensor): Shape is [n, hc_mult, hc_mult].
        compute_rms (bool): Whether to compute output_rms.
        eps (float): The eps of normalization.
        output (torch.Tensor, optional): Shape is [n, hc_mult, dim]. Defaults to None.
        output_rms (torch.Tensor, optional): Shape is [n]. Defaults to None.

    Type:
        x: float, half, bfloat16.
        residual: same as x.
        post: float.
        comb: float.
        output: same as x.
        output_rms: float.

    Returns:
        output, output_rms

    Limitation:
        dim <= 7168
        hc_mult must be 4.
    """
    if output is None:
        output = torch.empty_like(residual)
    if compute_rms and output_rms is None:
        output_rms = torch.empty((x.size(0),), dtype=torch.float32, device=x.device)
    torch.ops.torch_mlu_ops.fused_mhc_post(x, residual, post, comb, output, output_rms, compute_rms, eps)
    return (output, output_rms) if compute_rms else output

def fused_compress_multi_kv(kv: torch.Tensor,
                            score: torch.Tensor,
                            state_cache: torch.Tensor,
                            state_block_table: torch.Tensor,
                            cu_seqlens: torch.Tensor,
                            positions: torch.Tensor,
                            ape: torch.Tensor,
                            max_seqlen:int,
                            overlap: bool,
                            compressed_kv: torch.Tensor
                            ):
    """
    Compress key/value tensors using attention-based compression with paged KV cache.

    This operation writes kv and score+ape into a paged state_cache indexed by
    state_block_table, then performs softmax-weighted compression.

    Args:
        kv (torch.Tensor): Input key/value tensor in packed format. Shape is [total_seqlen, coff * head_dim].
        score (torch.Tensor): Input score tensor in packed format. Shape is [total_seqlen, coff * head_dim].
        state_cache (torch.Tensor): Paged state cache. Shape is [block_num, block_size, 2 * coff * head_dim], FP32.
        state_block_table (torch.Tensor): Block table mapping logical to physical blocks. Shape is [batch, max_state_blocks], INT32.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths for packed inputs. Shape is [batch + 1].
        positions (torch.Tensor): Per-token absolute positions. Shape is [total_seqlen].
        ape (torch.Tensor): Absolute positional encoding table. Shape is [ratio, coff * head_dim].
        max_seqlen (int): Maximum sequence length across all samples in the batch.
        overlap (bool): Whether to use overlapping compression. Must be True when ratio=4, False when ratio=128.
        compressed_kv (torch.Tensor): Output compressed key/value tensor. Shape is [total_compressed_seqlen, head_dim].

    Type:
       kv: BF16
       score: same as kv
       state_cache: FP32
       state_block_table: INT32
       cu_seqlens: INT32
       positions: INT32
       ape: FP32
       compressed_kv: same as kv

    Returns:
        Tuple containing:
        - state_cache (torch.Tensor): Updated paged state cache (in-place modified)
        - compressed_kv (torch.Tensor): Compressed KV output tensor (in-place modified)

    Note:
        coff = overlap + 1
        Three modes supported:
        overlap = True, ratio = 4, head_dim = 128;
        overlap = True, ratio = 4, head_dim = 512;
        overlap = False, ratio = 128, head_dim = 512.
    """
    torch.ops.torch_mlu_ops.fused_compress_multi_kv(kv, score, state_cache, state_block_table, cu_seqlens, positions, ape, max_seqlen, overlap, compressed_kv)
    return state_cache, compressed_kv

def fused_mla_q_v2(input_q: torch.Tensor,
                    gamma: torch.Tensor,
                    smooth_quant_scale: Optional[torch.Tensor],
                    weight_b: torch.Tensor,
                    weight_b_scale: Optional[torch.Tensor],
                    sin: torch.Tensor,
                    cos: torch.Tensor,
                    position_id: torch.Tensor,
                    output: Optional[torch.Tensor] = None,
                    eps: float = 1e-6,
                    interleaved: bool = True,
                    store_norm: bool = False,
                    output_norm: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    This function applies MLA (Multi-head Latent Attention) v2 Query (Q) preprocessing.
    The fusion logic includes: RMSNorm -> Quant(Optional) -> MatMul -> RMSNorm -> RoPE.

    Math:
        qr = rmsnorm(input_q, gamma, eps)
        if quant:
            qr, q_scale = per_token_quant(norm_out, smooth_quant_scale)
        q = matmul(qr, q_scale, weight_b, weight_b_scale)
        q = q.reshape(batch, seq, n_local_heads, head_dim)
        q = rsqrt(q.square().mean(-1, keepdim=True) + eps)
        out = apply_rotary_embedding(q, sin, cos, position_id, interleaved)

    Args:
        input_q (torch.Tensor):
            The input latent query tensor. Shape is (batch, seq, q_lora_rank).
        gamma (torch.Tensor):
            The scaling parameter for the initial RMSNorm. Shape is (q_lora_rank).
        smooth_quant_scale (Optional[torch.Tensor]):
            Scale tensor for SmoothQuant migration. Can be None. Shape is (q_lora_rank).
        weight_b (torch.Tensor):
            The Q-projection weight tensor. Shape is (n_local_heads, head_dim, q_lora_rank).
        weight_b_scale (Optional[torch.Tensor]):
            The per-channel quantization scales for weight_b. Shape is (n_local_heads, head_dim).
        sin (torch.Tensor):
            Rotary embedding sine table. Shape is (max_rotary_seq_len, rotary_head_dim).
        cos (torch.Tensor):
            Rotary embedding cosine table. Shape is (max_rotary_seq_len, rotary_head_dim).
        position_id (torch.Tensor):
            Indices for the RoPE tables. Shape is (batch,).
        output (Optional[torch.Tensor]):
            Optional output tensor for the final processed Q. Shape is (batch, seq, n_local_heads, head_dim).
        eps (float):
            Small constant for RMSNorm numerical stability. Default: 1e-6.
        interleaved (bool):
            If True, apply interleaved rotary embedding, otherwise folded. Default: True.
        store_norm (bool):
            If True, the intermediate RMSNorm result (pre-MatMul) will be returned. Default: False.
        output_norm (Optional[torch.Tensor]):
            Optional tensor to store the intermediate RMSNorm result. Shape: (batch, seq, q_lora_rank).

    Type:
        input_q, gamma, sin, cos: bfloat16.
        weight_b: int8, same as input_q.
        weight_b_scale, smooth_quant_scale: float32.
        position_id: int32.
        output: same as input_q.

    Return:
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
            - If store_norm=False: output
            - If store_norm=True: (..., output_norm) is appended to the return.

    Note:
        1. input_q must be bfloat16 dtype only.
        2. rotary_head_dim (sin.size(-1)) must be even.
        3. If weight_b is int8, both smooth_quant_scale and weight_b_scale must be provided.
    """
    batch, seq, q_lora_rank = input_q.size(0), input_q.size(1), input_q.size(2)
    n_local_heads, head_dim = weight_b.size(0), weight_b.size(1)
    out_dtype = input_q.dtype
    if output is None:
        output = torch.empty(batch, seq, n_local_heads, head_dim, device='mlu', dtype=out_dtype)
    if store_norm and output_norm is None:
        output_norm = torch.empty(batch, seq, q_lora_rank, device='mlu', dtype=input_q.dtype)
    final_output_norm = output_norm if store_norm else None
    weight_b_scale = weight_b_scale if weight_b.dtype != input_q.dtype else None
    torch.ops.torch_mlu_ops.fused_mla_q_v2(input_q, output, final_output_norm, gamma, smooth_quant_scale,
                                        weight_b, weight_b_scale, sin, cos, position_id, eps, interleaved)
    outputs = [output]
    if store_norm:
        outputs.append(output_norm)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

def hc_split_sinkhorn(mixes: torch.Tensor,
                      hc_scale: torch.Tensor,
                      hc_base: torch.Tensor,
                      pre_scale: Optional[torch.Tensor] = None,
                      hc_mult: int = 4,
                      sinkhorn_iter: int = 20,
                      eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate pre, post, comb Hyper-Connections.

    Args:
        mixes (torch.Tensor): The input tensor stores the connections, the shape must be [B, S, (2 + hc_mult) * hc_mult].
        hc_scale (torch.Tensor): The input tensor stores the scales, the shape must be [3].
        hc_base (torch.Tensor): The input tensor stores the biases, the shape must be [(2 + hc_mult) * hc_mult].
        pre_scale (torch.Tensor): The input tensor stores the pre scales, the shape must be [B, S].
        hc_mult: Hyper-Connections multiplier (default: 4).
        sinkhorn_iter: Number of sinkhorn loop iterations (default: 20).
        eps: Small epsilon value (default: 1e-6).

    Types:
        mixes: float32, bfloat16.
        hc_mult: int32.
        sinkhorn_iter: int32.
        eps: float32.

    Return:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - pre: Shape is [B, S, hc_mult], data type is float32.
            - post: Shape is [B, S, hc_mult], data type is float32.
            - comb: Shape is [B, S, hc_mult, hc_mult], data type is float32.
    """

    outputs = torch.ops.torch_mlu_ops.hc_split_sinkhorn(mixes, hc_scale, hc_base, pre_scale,
                                                        hc_mult, sinkhorn_iter, eps)
    return tuple(outputs)

def get_compress_block_tables(compress_block_tables: torch.Tensor,
                             compress_context_lens: torch.Tensor,
                             seq_k_lens: torch.Tensor,
                             query_start_loc: torch.Tensor,
                             offset: torch.Tensor,
                             block_table: torch.Tensor,
                             block_size: int,
                             ratio: int):
    """
    Generate compressed block table.

    Math:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        start_positions = seq_k_lens - query_lens
        token_offset = 0
        for bz, start_pos in enumerate(start_positions):
            offset_bz = offset[bz].item()
            seqlen = (query_start_loc[bz+1] - query_start_loc[bz]).item()
            if start_pos > 0:
                for i in range(seqlen):
                    valid_num = (start_pos.item() + i + 1) // ratio
                    compress_block_tables[token_offset + i,:valid_num] = converted_block_table[offset_bz:valid_num+offset_bz]
                    compress_context_lens[token_offset + i] = valid_num
                token_offset += seqlen
            else:
                for i in range(seqlen):
                    pos = token_offset + i
                    valid_num = (i+1) // ratio
                    compress_block_tables[pos,:valid_num] = torch.arange(valid_num, device=device) + offset_bz
                    compress_context_lens[pos] = valid_num
                token_offset += seqlen

    Args:
        compress_block_tables (torch.Tensor):
            Output compressed block tables. Shape is (total_queries, max_blocks).
        compress_context_lens (torch.Tensor):
            Output compressed context lengths. Shape is (total_queries,).
        seq_k_lens (torch.Tensor):
            Sequence key lengths before compression. Shape is (batch_size,).
        query_start_loc (torch.Tensor):
            Cumulative query starting locations. Shape is (batch_size + 1,).
        offset (torch.Tensor):
            Offset for each sequence. Shape is (batch_size,).
        block_table (torch.Tensor):
            Original block table. Shape is (batch_size, max_blocks).
        block_size (int):
            Size of each block in the block table. Must be 1 or multiple of 16.
        ratio (int):
            Compression ratio.

    DataType:
        compress_block_tables: int32
        compress_context_lens: int32
        seq_k_lens: int32
        query_start_loc: int32
        offset: int32
        block_table: int32
        block_size: int32
        ratio: int32

    Return:
        A tuple of two tensors.
        compress_block_tables: int32-tensor with shape (total_queries, max_blocks)
        compress_context_lens: int32-tensor with shape (total_queries,)
    """
    torch.ops.torch_mlu_ops.get_compress_block_tables(compress_block_tables, compress_context_lens, seq_k_lens,
                                                     query_start_loc, offset, block_table, block_size, ratio)
    return compress_block_tables, compress_context_lens

def get_window_block_tables(window_block_tables: torch.Tensor,
                            window_context_lens: torch.Tensor,
                            seq_k_lens: torch.Tensor,
                            query_start_loc: torch.Tensor,
                            block_table: torch.Tensor,
                            block_size: int,
                            window_size: int):
    """
    Generate window-based block table for sliding window attention.
    start_position = query_len - k_len
    For each batch, if start_position > 0, it is treated as decode, otherwise as prefill.
    In other word, if query_len == k_len, it is treated as prefill.

    Math:
        For each query position, build a sliding window of blocks from the
        corresponding sequence's block_table.

    Args:
        window_block_tables (torch.Tensor):
            Output window block tables. Shape is (total_queries, max_blocks).
        window_context_lens (torch.Tensor):
            Output window context lengths. Shape is (total_queries,).
        seq_k_lens (torch.Tensor):
            Sequence key lengths. Shape is (batch_size,). batch_size should be smaller than or equal to 10000
        query_start_loc (torch.Tensor):
            Cumulative query starting locations. Shape is (batch_size + 1,).
        block_table (torch.Tensor):
            Original block table. Shape is (batch_size, max_blocks). Used in decoding.
        block_size (int):
            Number of tokens in each block of block_table, now only support 1 or multiple of 16.
        window_size (int):
            Number of blocks to keep in the sliding window. Should be smaller than 1024

    DataType:
        window_block_tables: int32
        window_context_lens: int32
        seq_k_lens: int32
        query_start_loc: int32
        block_table: int32
        block_size: int32
        window_size: int32

    Return:
        A tuple of two tensors.
        window_block_tables: int32-tensor with shape (total_queries, max_blocks)
        window_context_lens: int32-tensor with shape (total_queries,)
    """
    torch.ops.torch_mlu_ops.get_window_block_tables(window_block_tables, window_context_lens, seq_k_lens,
                                                     query_start_loc, block_table, block_size, window_size)
    return window_block_tables, window_context_lens

# add dump_gese as decorator automatically for every function mentioned in __all__
add_gen_case_decorator(inspect.currentframe())
