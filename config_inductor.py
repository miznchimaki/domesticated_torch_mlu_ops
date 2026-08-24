import torch

import torch._custom_ops
import torch_mlu

if hasattr(torch_mlu._inductor.config, "aot_inductor"):
    tmo_custom_ops = {
        "torch.ops.torch_mlu_ops.active.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_active(
                        AtenTensorHandle input,
                        AtenTensorHandle output,
                        AtenTensorHandle *bias,
                        AtenTensorHandle *cusum_token_count,
                        std::string act_mode,
                        bool is_gated,
                        int64_t start_expert_id,
                        int64_t expert_size,
                        double active_coef,
                        bool high_precision,
                        std::string gelu_approximate,
                        int swiglu_limit,
                        AtenTensorHandle *weight)
            """,
        ],
        "torch.ops.torch_mlu_ops.batch_matmul.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_batch_matmul(
                        AtenTensorHandle a,
                        AtenTensorHandle b,
                        AtenTensorHandle *c,
                        AtenTensorHandle *bias,
                        const char **dtype,
                        AtenTensorHandle *a_scale_tensor,
                        AtenTensorHandle *b_scale_tensor,
                        std::string act_mode,
                        double alpha,
                        double beta,
                        double a_scale,
                        double b_scale,
                        bool trans_a,
                        bool trans_b,
                        bool use_hp_active,
                        bool approximate,
                        AtenTensorHandle* ret0)
            """,
        ],
        "torch.ops.torch_mlu_ops.batch_matmul_inplace.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_batch_matmul_inplace(AtenTensorHandle a,
                                                    AtenTensorHandle b,
                                                    AtenTensorHandle d,
                                                    AtenTensorHandle *c,
                                                    AtenTensorHandle *bias,
                                                    AtenTensorHandle *a_scale_tensor,
                                                    AtenTensorHandle *b_scale_tensor,
                                                    std::string act_mode,
                                                    double alpha,
                                                    double beta,
                                                    double a_scale,
                                                    double b_scale,
                                                    bool trans_a,
                                                    bool trans_b,
                                                    bool use_hp_active,
                                                    bool approximate)
            """,
        ],
        "torch.ops.torch_mlu_ops.scaled_quantize.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_scaled_quantize(
                        AtenTensorHandle x,
                        AtenTensorHandle output,
                        AtenTensorHandle *output_scale,
                        AtenTensorHandle *x_scale,
                        AtenTensorHandle *x_zero,
                        AtenTensorHandle *m_list,
                        AtenTensorHandle *gather_index,
                        AtenTensorHandle *gather_index_start_position,
                        AtenTensorHandle *scale_upper_bound,
                        std::string quant_mode,
                        std::string act_mode,
                        double active_coef,
                        bool is_gated,
                        int quant_bit_size,
                        bool need_output_scale_trans,
                        AtenTensorHandle *output_reduced,
                        int64_t group_size)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_layernorm.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_fused_layernorm(
                        AtenTensorHandle input,
                        AtenTensorHandle out,
                        AtenTensorHandle *residual,
                        AtenTensorHandle *gamma,
                        AtenTensorHandle *beta,
                        AtenTensorHandle *bias,
                        AtenTensorHandle *quant_scale,
                        AtenTensorHandle *residual_out,
                        AtenTensorHandle *smooth_quant_scale,
                        AtenTensorHandle *normed_out,
                        std::string norm_mode,
                        double eps,
                        bool store_output_before_norm,
                        bool store_output_after_norm,
                        bool dynamic_quant,
                        bool mx_quant,
                        bool transpose_4d_1_2)
            """,
        ],
        "torch.ops.torch_mlu_ops.scaled_matmul.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_scaled_matmul(AtenTensorHandle output,
                    AtenTensorHandle a_tensor,
                    AtenTensorHandle b_tensor,
                    AtenTensorHandle *a_scale,
                    AtenTensorHandle *a_zero,
                    AtenTensorHandle *a_calib,
                    AtenTensorHandle *b_scale,
                    AtenTensorHandle *b_zero,
                    AtenTensorHandle *b_calib,
                    AtenTensorHandle *bias,
                    AtenTensorHandle *c_tensor,
                    AtenTensorHandle *c_scale,
                    AtenTensorHandle *c_zero,
                    AtenTensorHandle *gemm_output_scale,
                    AtenTensorHandle *gemm_output_zero,
                    std::string quant_algo,
                    std::string a_quant_layout,
                    std::string b_quant_layout,
                    int64_t a_quant_bit_size,
                    int64_t b_quant_bit_size,
                    std::string act_mode,
                    bool use_hp_active,
                    double act_coef,
                    double alpha,
                    double beta,
                    bool trans_a,
                    bool trans_b)
            """,
        ],
        "torch.ops.torch_mlu_ops.group_gemm_v2.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_group_gemm_v2(AtenTensorHandle a_tensor,
                    AtenTensorHandle b_tensor,
                    AtenTensorHandle dim_list,
                    AtenTensorHandle d_tensor,
                    AtenTensorHandle *gather_idx,
                    AtenTensorHandle *c_tensor,
                    AtenTensorHandle *alpha,
                    AtenTensorHandle *beta,
                    AtenTensorHandle *a_scale,
                    AtenTensorHandle *b_scale,
                    AtenTensorHandle *bias,
                    AtenTensorHandle *a_calibration,
                    AtenTensorHandle *b_calibration,
                    AtenTensorHandle *quant_flag,
                    AtenTensorHandle *b_offset,
                    int64_t max_dim,
                    bool trans_a,
                    bool trans_b,
                    const int64_t a_quant_bit,
                    AtenTensorHandle *a_lora,
                    AtenTensorHandle *b_lora,
                    AtenTensorHandle *idx_offset,
                    bool allow_tf32,
                    bool is_symmetric_quant)
            """,
        ],
        "torch.ops.torch_mlu_ops.variable_n_group_gemm.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_variable_n_group_gemm(AtenTensorHandle a_tensor,
                                                    AtenTensorHandle b_tensor,
                                                    AtenTensorHandle d_tensor,
                                                    AtenTensorHandle m_list,
                                                    AtenTensorHandle n_list,
                                                    AtenTensorHandle lda,
                                                    int64_t max_m,
                                                    int64_t max_n,
                                                    int64_t k,
                                                    std::string input_dtype,
                                                    std::string output_dtype,
                                                    bool allow_tf32)
            """,
        ],
        "torch.ops.torch_mlu_ops.layernorm_forward.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_layernorm_forward(AtenTensorHandle input,
                    AtenTensorHandle output,
                    AtenTensorHandle *gamma,
                    AtenTensorHandle *beta,
                    double eps,
                    double gamma_add_coef)
            """,
        ],
        "torch.ops.torch_mlu_ops.moe_gen_idx.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_moe_gen_idx(AtenTensorHandle expert_id,
                                              int64_t expert_num,
                                              bool return_token2expert_idx,
                                              AtenTensorHandle **ret0)
            """,
        ],
        "torch.ops.torch_mlu_ops.flash_attention.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_flash_attention(AtenTensorHandle q,
                                                  AtenTensorHandle k,
                                                  AtenTensorHandle v,
                                                  AtenTensorHandle out,
                                                  AtenTensorHandle *output_lse,
                                                  AtenTensorHandle *cu_seq_lens_q,
                                                  AtenTensorHandle *cu_seq_lens_kv,
                                                  AtenTensorHandle *alibi_slope,
                                                  AtenTensorHandle *attn_bias,
                                                  AtenTensorHandle *q_quant_scale,
                                                  AtenTensorHandle *k_cache_quant_scale,
                                                  AtenTensorHandle *v_cache_quant_scale,
                                                  AtenTensorHandle *out_quant_scale,
                                                  AtenTensorHandle *block_tables,
                                                  int64_t max_seq_len_q,
                                                  int64_t max_seq_len_kv,
                                                  double softmax_scale,
                                                  bool is_causal,
                                                  int64_t window_size_left,
                                                  int64_t window_size_right,
                                                  std::string compute_dtype,
                                                  bool return_lse,
                                                  AtenTensorHandle *q2k_block_idx,
                                                  AtenTensorHandle *q2k_block_num,
                                                  AtenTensorHandle *variable_block_sizes,
                                                  int64_t q_block_size,
                                                  int64_t k_block_size,
                                                  AtenTensorHandle *sink)
        """,
        ],
        "torch.ops.torch_mlu_ops.aot_flash_attention.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_aot_flash_attention(AtenTensorHandle q,
                                                  AtenTensorHandle k,
                                                  AtenTensorHandle v,
                                                  AtenTensorHandle *cu_seq_lens_q,
                                                  AtenTensorHandle *cu_seq_lens_kv,
                                                  AtenTensorHandle *alibi_slope,
                                                  AtenTensorHandle *attn_bias,
                                                  AtenTensorHandle *q_quant_scale,
                                                  AtenTensorHandle *k_cache_quant_scale,
                                                  AtenTensorHandle *v_cache_quant_scale,
                                                  AtenTensorHandle *out_quant_scale,
                                                  AtenTensorHandle *block_tables,
                                                  int64_t max_seq_len_q,
                                                  int64_t max_seq_len_kv,
                                                  double softmax_scale,
                                                  bool is_causal,
                                                  int64_t window_size_left,
                                                  int64_t window_size_right,
                                                  std::string compute_dtype,
                                                  bool return_lse,
                                                  AtenTensorHandle *q2k_block_idx,
                                                  AtenTensorHandle *q2k_block_num,
                                                  AtenTensorHandle *variable_block_sizes,
                                                  int64_t q_block_size,
                                                  int64_t k_block_size,
                                                  std::string out_dtype,
                                                  AtenTensorHandle *sink,
                                                  AtenTensorHandle **ret0)
        """,
        ],
        "torch.ops.torch_mlu_ops.moe_combine_result.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_moe_combine_result(AtenTensorHandle input,
                        AtenTensorHandle output,
                        AtenTensorHandle reduce_weight,
                        AtenTensorHandle gather_ids,
                        AtenTensorHandle *residual,
                        AtenTensorHandle *cusum_token_count,
                        int64_t start_expert_id,
                        int64_t expert_size,
                        AtenTensorHandle *bias)
            """,
        ],
        "torch.ops.torch_mlu_ops.moe_expand_input.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_moe_expand_input(AtenTensorHandle input,
                        AtenTensorHandle gather_idx,
                        AtenTensorHandle *cusum_token_count,
                            int64_t start_expert_id,
                            int64_t expert_size,
                            AtenTensorHandle *ret0)
            """,
        ],
        "torch.ops.torch_mlu_ops.moe_expand_input_inplace.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_moe_expand_input_inplace(AtenTensorHandle input,
                                                        AtenTensorHandle gather_idx,
                                                        AtenTensorHandle *cusum_token_count,
                                                        int64_t start_expert_id,
                                                        int64_t expert_size,
                                                        AtenTensorHandle output)
            """,
        ],
        "torch.ops.torch_mlu_ops.quant_per_block.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_quant_per_block(AtenTensorHandle q,
                    AtenTensorHandle *k,
                    AtenTensorHandle *v,
                    AtenTensorHandle *seq_lens_q,
                    AtenTensorHandle *seq_lens_k,
                    AtenTensorHandle *seq_lens_v,
                    const int64_t max_seq_q,
                    const int64_t max_seq_k,
                    const int64_t max_seq_v,
                    const int64_t block_size_q,
                    const int64_t block_size_k,
                    const bool smooth_k,
                    AtenTensorHandle quant_q,
                    AtenTensorHandle q_scale,
                    AtenTensorHandle *quant_k,
                    AtenTensorHandle *k_scale,
                    AtenTensorHandle *quant_v,
                    AtenTensorHandle *v_scale,
                    AtenTensorHandle *k_mean)
            """,
        ],
        "torch.ops.torch_mlu_ops.moe_active_topk.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_moe_active_topk(AtenTensorHandle input,
                    int64_t topk,
                    int64_t num_expert_group,
                    int64_t topk_group,
                    bool normalize,
                    AtenTensorHandle *mask,
                    const std::string normed_by,
                    const std::string act_type,
                    double route_scale,
                    AtenTensorHandle *score_bias,
                    AtenTensorHandle reduce_weight,
                    AtenTensorHandle expert_id)
            """,
        ],
        "torch.ops.torch_mlu_ops.moe_softplus_topk.default":[
            """
            AOTITorchError
             aoti_torch_mlu_ops_moe_softplus_topk(AtenTensorHandle input,
                    AtenTensorHandle *input_ids,
                    AtenTensorHandle *tid2eid,
                    AtenTensorHandle *bias,
                    int64_t topk,
                    double route_scale,
                    AtenTensorHandle reduce_weight,
                    AtenTensorHandle expert_id)
            """,
        ],
        "torch.ops.torch_mlu_ops.matmul_v2.default":[
            """
            AOTITorchError
             aoti_torch_mlu_ops_matmul_v2(AtenTensorHandle a,
                    AtenTensorHandle b,
                    AtenTensorHandle output,
                    AtenTensorHandle *bias,
                    AtenTensorHandle *c,
                    AtenTensorHandle *a_scale_tensor,
                    AtenTensorHandle *b_scale_tensor,
                    const std::string act_mode,
                    double alpha,
                    double beta,
                    bool fast_act,
                    bool approximate,
                    double a_scale,
                    double b_scale,
                    bool trans_a,
                    bool trans_b)
            """
        ],
        "torch.ops.torch_mlu_ops.matmul_aot_inductor.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_matmul_aot_inductor(AtenTensorHandle a,
                    AtenTensorHandle b,
                    AtenTensorHandle *bias,
                    AtenTensorHandle *c,
                    AtenTensorHandle *a_scale_tensor,
                    AtenTensorHandle *b_scale_tensor,
                    const char **dtype,
                    const std::string act_mode,
                    double alpha,
                    double beta,
                    bool fast_act,
                    bool approximate,
                    double a_scale,
                    double b_scale,
                    bool trans_a,
                    bool trans_b,
                    AtenTensorHandle *ret0)
            """
        ],
        "torch.ops.torch_mlu_ops.fused_indexer_k.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_fused_indexer_k(AtenTensorHandle x,
                    AtenTensorHandle wk,
                    AtenTensorHandle wproj,
                    AtenTensorHandle sin_table,
                    AtenTensorHandle cos_table,
                    AtenTensorHandle position_id,
                    AtenTensorHandle slot_mapping,
                    AtenTensorHandle head_weights,
                    AtenTensorHandle k_cache,
                    AtenTensorHandle *k_cache_scale,
                    AtenTensorHandle *hadamard_matrix,
                    bool interleaved,
                    AtenTensorHandle *gamma,
                    AtenTensorHandle *beta,
                    double eps)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_indexer_q.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_fused_indexer_q(AtenTensorHandle input_q,
                    AtenTensorHandle output,
                    AtenTensorHandle *output_scale,
                    AtenTensorHandle w_q,
                    AtenTensorHandle *w_q_scale,
                    AtenTensorHandle *hadamard_matrix,
                    AtenTensorHandle sin,
                    AtenTensorHandle cos,
                    AtenTensorHandle position_id,
                    const std::string output_quant_mode,
                    bool interleaved,
                    bool rope_at_front)
            """,
        ],
        "torch.ops.torch_mlu_ops.masked_indexer_select_paged_kv.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_masked_indexer_select_paged_kv(AtenTensorHandle query,
                    AtenTensorHandle k_scale,
                    AtenTensorHandle weights,
                    AtenTensorHandle kv_cache_block_table,
                    AtenTensorHandle *cu_seq_q_lens,
                    AtenTensorHandle *cu_seq_k_lens,
                    AtenTensorHandle *k_context_lens,
                    AtenTensorHandle *k_cache_block_table,
                    bool is_prefill,
                    int64_t index_topk,
                    int64_t kv_cache_block_size,
                    double softmax_scale,
                    AtenTensorHandle *q_scale,
                    AtenTensorHandle *k_scale_cache,
                    AtenTensorHandle new_block_table,
                    AtenTensorHandle new_context_lens,
                    bool is_score_float,
                    int64_t compress_ratio,
                    AtenTensorHandle kv_cache_block_table_offset)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_gather_clamp_concat.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_fused_gather_clamp_concat(AtenTensorHandle output,
                    AtenTensorHandle q,
                    AtenTensorHandle gather_index,
                    AtenTensorHandle total_index_num)
            """,
        ],
        "torch.ops.torch_mlu_ops.solve_tril.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_solve_tril(AtenTensorHandle input,
                    AtenTensorHandle output,
                    AtenTensorHandle *cu_seqlens)
        """,
        ],
        "torch.ops.torch_mlu_ops.ssparse_matmul.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_ssparse_matmul(AtenTensorHandle a,
                    AtenTensorHandle b,
                    AtenTensorHandle a_scale,
                    AtenTensorHandle b_scale,
                    AtenTensorHandle output,
                    const std::string act_mode,
                    AtenTensorHandle *m_list,
                    AtenTensorHandle *gather_idx,
                    AtenTensorHandle *bias,
                    AtenTensorHandle *c,
                    int64_t max_m,
                    double alpha,
                    double beta,
                    bool trans_a,
                    bool trans_b)
            """,
        ],
        "torch.ops.torch_mlu_ops.gen_label_q_idx.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_gen_label_q_idx(AtenTensorHandle cu_seqs_q,
                    AtenTensorHandle seqs_k,
                    AtenTensorHandle gmm_a,
                    AtenTensorHandle gmm_b,
                    AtenTensorHandle gmm_d,
                    AtenTensorHandle label_q_index,
                    AtenTensorHandle index_num,
                    AtenTensorHandle new_cu_seq_lens_q,
                    AtenTensorHandle gmm_m_list,
                    AtenTensorHandle gmm_n_list,
                    AtenTensorHandle gmm_a_ptrs,
                    AtenTensorHandle gmm_b_ptrs,
                    AtenTensorHandle gmm_d_ptrs,
                    AtenTensorHandle gmm_lda,
                    int64_t max_seq_q,
                    int64_t max_seq_k,
                    int64_t block_size_q,
                    int64_t block_size_k)
            """,
        ],
        "torch.ops.torch_mlu_ops.masked_topk_select_block_table.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_masked_topk_select_block_table(AtenTensorHandle qk_logit,
                    AtenTensorHandle origin_block_table,
                    AtenTensorHandle cu_seqs_q,
                    AtenTensorHandle seqs_k,
                    AtenTensorHandle sparse_context_lens,
                    AtenTensorHandle sparse_block_table,
                    int64_t max_seq_q,
                    int64_t max_seq_k,
                    int64_t block_size_q,
                    int64_t block_size_k,
                    int64_t recent_window,
                    double sparse_ratio,
                    bool window_included)
            """,
        ],
        "torch.ops.torch_mlu_ops.transpose_all2all.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_transpose_all2all(const int64_t cncl_comm,
                                                const int64_t pre_num_block,
                                                const int64_t pre_block_count,
                                                const int64_t post_num_block,
                                                const int64_t post_block_count,
                                                AtenTensorHandle send,
                                                AtenTensorHandle recv)
            """,
        ],
        "torch.ops.torch_mlu_ops.convert_vertical_slash_index.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_convert_vertical_slash_index(
                AtenTensorHandle seqlens,
                AtenTensorHandle ctxlens,
                AtenTensorHandle vertical_indexes,
                AtenTensorHandle slash_indexes,
                const int64_t max_seqlen_q,
                const int64_t block_size_M,
                const int64_t block_size_N,
                AtenTensorHandle* ret0,
                AtenTensorHandle* ret1,
                AtenTensorHandle* ret2,
                AtenTensorHandle* ret3)
            """,
        ],
        "torch.ops.torch_mlu_ops.hamming_score.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_hamming_score(AtenTensorHandle query_code,
                                              AtenTensorHandle key_codes,
                                              AtenTensorHandle* block_table_opt,
                                              AtenTensorHandle seq_len,
                                              int64_t max_seq_len,
                                              int64_t sink,
                                              int64_t recent,
                                              AtenTensorHandle *ret0)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_mul_reduce_sum.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_fused_mul_reduce_sum(AtenTensorHandle x,
                                                    AtenTensorHandle w,
                                                    AtenTensorHandle *ret0)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_compress_single_kv.default":[
            """
            AOTITorchError
            aoti_torch_mlu_ops_fused_compress_single_kv(AtenTensorHandle kv,
                                            AtenTensorHandle score,
                                            AtenTensorHandle position,
                                            AtenTensorHandle ape,
                                            AtenTensorHandle gamma,
                                            AtenTensorHandle sin,
                                            AtenTensorHandle cos,
                                            AtenTensorHandle *hadamard_matrix,
                                            AtenTensorHandle slot_mapping,
                                            AtenTensorHandle kv_cache,
                                            AtenTensorHandle *kv_cache_scale,
                                            double eps,
                                            bool overlap,
                                            AtenTensorHandle state_cache,
                                            AtenTensorHandle state_bt,
                                            int state_width,
                                            int state_block_size,
                                            AtenTensorHandle cu_query_len,
                                            int K)
            """,
        ],
        "torch.ops.torch_mlu_ops.update_compressor_states.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_update_compressor_states(AtenTensorHandle kv_state,
                                                         AtenTensorHandle score_state,
                                                         AtenTensorHandle accept_tokens,
                                                         AtenTensorHandle batch_to_kv_state,
                                                         AtenTensorHandle positions,
                                                         AtenTensorHandle cu_query_len,
                                                         bool overlap,
                                                         int64_t K)
            """,
        ],
        "torch.ops.torch_mlu_ops.get_window_block_tables.default":[
            """
            AOTITorchError
             aoti_torch_mlu_ops_get_window_block_tables(AtenTensorHandle window_block_tables,
                    AtenTensorHandle window_context_lens,
                    AtenTensorHandle seq_k_lens,
                    AtenTensorHandle query_start_loc,
                    AtenTensorHandle block_table,
                    const int64_t block_size,
                    const int64_t window_size)
            """,
        ],
        "torch.ops.torch_mlu_ops.get_compress_block_tables.default":[
            """
            AOTITorchError
             aoti_torch_mlu_ops_get_compress_block_tables(AtenTensorHandle compress_block_tables,
                    AtenTensorHandle compress_context_lens,
                    AtenTensorHandle seq_k_lens,
                    AtenTensorHandle query_start_loc,
                    AtenTensorHandle offset,
                    AtenTensorHandle block_table,
                    const int64_t block_size,
                    const int64_t ratio)
            """,
        ],
        "torch.ops.torch_mlu_ops.dynamic_per_channel_quant.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_dynamic_per_channel_quant(AtenTensorHandle input,
                    AtenTensorHandle *seq_lens,
                    const int64_t max_seq,
                    AtenTensorHandle quant_out,
                    AtenTensorHandle quant_scale)
            """,
        ],
        "torch.ops.torch_mlu_ops.concat_block_table.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_concat_block_table(AtenTensorHandle first_block_table,
                                                   AtenTensorHandle first_context_lens,
                                                   AtenTensorHandle second_block_table,
                                                   AtenTensorHandle second_context_lens,
                                                   AtenTensorHandle new_block_table,
                                                   AtenTensorHandle new_context_lens)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_mhc_post.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_fused_mhc_post(AtenTensorHandle x,
                                            AtenTensorHandle residual,
                                            AtenTensorHandle post,
                                            AtenTensorHandle comb,
                                            AtenTensorHandle output,
                                            AtenTensorHandle *output_rms,
                                            bool compute_rms,
                                            double eps)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_compress_multi_kv.default": [
            """
            AOTITorchError
            aoti_torch_mlu_ops_fused_compress_multi_kv(AtenTensorHandle kv,
                                        AtenTensorHandle score,
                                        AtenTensorHandle state_cache,
                                        AtenTensorHandle state_block_table,
                                        AtenTensorHandle cu_seqlens,
                                        AtenTensorHandle positions,
                                        AtenTensorHandle ape,
                                        const int64_t max_seqlen,
                                        const bool overlap,
                                        AtenTensorHandle compressed_kv)
            """,
        ],
        "torch.ops.torch_mlu_ops.fused_mla_q_v2.default":[
            """
            AOTITorchError
             aoti_torch_mlu_ops_fused_mla_q_v2(AtenTensorHandle input_q,
                                            AtenTensorHandle output,
                                            AtenTensorHandle *output_norm,
                                            AtenTensorHandle gamma,
                                            AtenTensorHandle *smooth_quant_scale,
                                            AtenTensorHandle w_qb,
                                            AtenTensorHandle *w_qb_scale,
                                            AtenTensorHandle sin,
                                            AtenTensorHandle cos,
                                            AtenTensorHandle position_id,
                                            double eps,
                                            bool interleaved)
            """,
        ],
        "torch.ops.torch_mlu_ops.hc_split_sinkhorn.default": [
            """
            AOTITorchError
             aoti_torch_mlu_ops_hc_split_sinkhorn(AtenTensorHandle mixes,
                                                  AtenTensorHandle hc_scale,
                                                  AtenTensorHandle hc_base,
                                                  AtenTensorHandle *pre_scale,
                                                  int64_t hc_mult,
                                                  int64_t sinkhorn_iter,
                                                  double eps,
                                                  AtenTensorHandle **ret0)
            """,
        ],
    }
    torch_mlu._inductor.config.aot_inductor.custom_ops_to_c_shims.update(tmo_custom_ops)
    if torch_mlu._inductor.config.aot_inductor.custom_op_libs is None:
        torch_mlu._inductor.config.aot_inductor.custom_op_libs = ["torch_mlu_ops"]
    else:
        torch_mlu._inductor.config.aot_inductor.custom_op_libs += ["torch_mlu_ops"]
