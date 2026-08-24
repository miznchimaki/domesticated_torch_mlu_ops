import torch
import torch_mlu
import triton
import triton.language as tl
import triton.backends.mlu.driver as driver
from typing import Optional
@triton.heuristics({
    "num_stages": lambda args: 3,
    "num_warps": lambda args: 1,
    "ONCE_BLK_NUM": lambda args:args["block_size"],
    "ONCE_H_NUM": lambda args: min(args["H"], 8),
    "ONCE_D_NUM": lambda args: min(args["D"], 128),
})
@triton.jit
def tmo_reduce_to_label_cache_kernel(
    k_ptr,  # (T, H, D)
    cu_seq_lens_ptr,  # (B + 1,)
    slot_mapping_ptr, # (B, max_blkn)
    min_paged_cache_ptr, # (PHY_BLKNUM, H, block_size, D)
    max_paged_cache_ptr, # (PHY_BLKNUM, H, block_size, D)
    k_max_ptr, # (B, H, max_blkn, D)
    kmax_mblkn_stride,
    k_min_ptr, # (B, H, max_blkn, D)
    kmin_mblkn_stride,
    block_nums_ptr, # (B, )
    B,
    H:tl.constexpr,
    D:tl.constexpr,
    block_size:tl.constexpr,
    max_blkn:tl.constexpr,
    store_paged_cache:tl.constexpr,
    ONCE_B_NUM:tl.constexpr,
    ONCE_BLK_NUM: tl.constexpr,
    ONCE_H_NUM: tl.constexpr,
    ONCE_D_NUM: tl.constexpr
):
    task_dim = tl.num_programs(0)
    task_id = tl.program_id(0)
    batch_blk_id = tl.program_id(1)
    cur_ld_seq_base_ptr = cu_seq_lens_ptr + batch_blk_id * ONCE_B_NUM
    blk_nums_mask = tl.arange(0, ONCE_B_NUM) < (B - batch_blk_id * ONCE_B_NUM)
    cur_cu_seq_lens = tl.load(cur_ld_seq_base_ptr + tl.arange(0, ONCE_B_NUM), mask = blk_nums_mask)
    cur_cu_seq_lens_nxt = tl.load(cur_ld_seq_base_ptr + 1 + tl.arange(0, ONCE_B_NUM), mask = blk_nums_mask)
    block_nums = (cur_cu_seq_lens_nxt - cur_cu_seq_lens) // block_size
    repeat_h = (H + ONCE_H_NUM - 1) // ONCE_H_NUM
    repeat_d = (D + ONCE_D_NUM - 1) // ONCE_D_NUM
    repeat_hd = repeat_h * repeat_d
    repeat_batch = min((batch_blk_id + 1) * ONCE_B_NUM, B)
    for batch_idx in range(batch_blk_id * ONCE_B_NUM, repeat_batch):
        blkni = block_nums[batch_idx - batch_blk_id * ONCE_B_NUM]
        k_batch_base_addr = k_ptr + cur_cu_seq_lens[batch_idx - batch_blk_id * ONCE_B_NUM] * H * D
        kmax_batch_offset = (batch_idx) * H * max_blkn * kmax_mblkn_stride
        kmin_batch_offset = (batch_idx) * H * max_blkn * kmin_mblkn_stride
        if store_paged_cache:
            slot_ids = tl.load(slot_mapping_ptr + (batch_idx) * max_blkn + tl.arange(0, max_blkn), mask = tl.arange(0, max_blkn) < blkni)
        for i in range(repeat_hd):
            h_idx = i // repeat_d
            d_idx = i % repeat_d
            off_d = d_idx * ONCE_D_NUM + tl.arange(0, ONCE_D_NUM)
            off_h = h_idx * ONCE_H_NUM + tl.arange(0, ONCE_H_NUM)
            mask_hd = (off_d[None, None, :] < D) & (off_h[None, :, None] < H)
            st_mask_hd = (off_d[None, :] < D) & (off_h[:, None] < H)
            # split block_size during repeat
            for blk_seq_idx in range(task_id, blkni, task_dim):
                k_blk_seq_addr = k_batch_base_addr + blk_seq_idx * block_size * H * D
                off_blk = tl.arange(0, ONCE_BLK_NUM)[:, None, None]
                ld_mask = (off_blk < block_size) & mask_hd
                k_ld_addr = k_blk_seq_addr + off_blk * H * D + off_h[None, :, None] * D + off_d[None, None, :]
                ld_data = tl.load(k_ld_addr, mask = ld_mask)
                cur_max_res = tl.max(ld_data, axis = 0)
                cur_min_res = tl.min(ld_data, axis = 0)
                tl.store(k_max_ptr + kmax_batch_offset + blk_seq_idx * kmax_mblkn_stride \
                        + off_h[:, None] * max_blkn * kmax_mblkn_stride + off_d[None, :], cur_max_res, mask = st_mask_hd)
                tl.store(k_min_ptr + kmin_batch_offset + blk_seq_idx * kmin_mblkn_stride \
                        + off_h[:, None] * max_blkn * kmin_mblkn_stride + off_d[None, :], cur_min_res, mask = st_mask_hd)
                if store_paged_cache:
                    slot_id = slot_ids[blk_seq_idx]
                    slot_blk_id = slot_id // block_size
                    slot_blk_rem = slot_id % block_size
                    tl.store(max_paged_cache_ptr + slot_blk_id * H * block_size * D + slot_blk_rem * D + off_h[:, None] * block_size * D + off_d[None, :], cur_max_res, mask = st_mask_hd)
                    tl.store(min_paged_cache_ptr + slot_blk_id * H * block_size * D + slot_blk_rem * D + off_h[:, None] * block_size * D + off_d[None, :], cur_min_res, mask = st_mask_hd)
    if task_id == 0:
        tl.store(block_nums_ptr + batch_blk_id * ONCE_B_NUM + tl.arange(0, ONCE_B_NUM), block_nums, mask = blk_nums_mask)

def reduce_to_label_cache(k: torch.Tensor,
                          cu_seq_lens: torch.Tensor,
                          max_seq: int,
                          block_size: int,
                          k_max: torch.Tensor,
                          k_min: torch.Tensor,
                          min_paged_cache,
                          max_paged_cache,
                          slot_mapping,
                          block_nums
                          ):
    if k is None:
        raise ValueError('k cannot be None')
    if k_max is None:
        raise ValueError('k_max cannot be None')
    if k_min is None:
        raise ValueError('k_min cannot be None')
    if cu_seq_lens is None:
        raise ValueError('cu_seq_lens cannot be None')
    if k.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"k.dtype must be float16 or bfloat16, got {k.dtype}"
        )
    if block_nums.dtype != torch.int32:
        raise ValueError(
            f"block_nums.dtype must be int32, got {block_nums.dtype}"
        )
    if cu_seq_lens.dtype != torch.int32:
        raise ValueError(
            f"cu_seq_lens.dtype must be int32, got {cu_seq_lens.dtype}"
        )
    if block_size > 64:
        raise ValueError(f"block_size must be <= 64, but got {block_size}")
    B = cu_seq_lens.size(0) - 1
    H, D = k.size(1), k.size(2)
    T = k.size(0)
    max_blkn = max_seq // block_size
    INT32_MAX = 2**31 - 1
    if T * H * D > INT32_MAX:
        raise ValueError(f"can not deal large tensor yet, T*H*D = {T}*{H}*{D} = {T*H*D} > INT32_MAX ({INT32_MAX})")
    if B * max_blkn * H * D > INT32_MAX:
        raise ValueError(f"can not deal large tensor yet, B*max_blkn*H*D = {B}*{max_blkn}*{H}*{D} = {B*max_blkn*H*D} > INT32_MAX ({INT32_MAX})")
    if B * max_seq < T:
        raise ValueError(f"token num {T} must be not less than block_num({B}) * max_seq({max_seq})")
    if k_max.dtype != k.dtype:
        raise TypeError(
            f"k_max.dtype must be same as k.dtype({k.dtype}), got {k_max.dtype}"
        )
    if k_max.shape != (B, H, max_blkn, D):
        raise ValueError(
            f"k_max.shape {tuple(k_max.shape)} != expected {(B, H, max_blkn, D)}"
        )
    if k_min.dtype != k.dtype:
        raise TypeError(
            f"k_min.dtype must be same as k.dtype({k.dtype}), got {k_min.dtype}"
        )
    if k_min.shape != (B, H, max_blkn, D):
        raise ValueError(
            f"k_max.shape {tuple(k_min.shape)} != expected {(B, H, max_blkn, D)}"
        )
    if block_nums.shape != (B,):
        raise ValueError(
            f"block_nums.shape {tuple(block_nums.shape)} != expected {(B,)}"
        )
    store_paged_cache = (min_paged_cache is not None and
                        max_paged_cache is not None and
                        slot_mapping is not None)

    if store_paged_cache:
        if slot_mapping.dtype != torch.int32:
            raise ValueError(
                f"slot_mapping.dtype must be int32, got {slot_mapping.dtype}"
            )
        if min_paged_cache.shape != max_paged_cache.shape:
            raise ValueError(f"Shape mismatch: min_paged_cache {min_paged_cache.shape} != max_paged_cache {max_paged_cache.shape}")
        if slot_mapping.shape != (B, max_blkn):
            raise ValueError(
                f"slot_mapping shape mismatch. Expected ({B}, {max_blkn}), got {slot_mapping.shape}"
            )
        phy_blknum = max_paged_cache.size(0)
        total_slots_needed = B * max_blkn
        total_slots_available = phy_blknum * block_size
        if total_slots_needed > total_slots_available:
            raise ValueError(
                f"Need {total_slots_needed} slots, but only {total_slots_available} available. "
            )
    TOTAL_CORE_NUM = torch.mlu.get_device_properties(
        torch.mlu.current_device()
    ).multi_processor_count
    TASK_DIM = TOTAL_CORE_NUM
    ONCE_B_NUM = min((1 << (B.bit_length() - 1)), 32)
    grid = (TASK_DIM, (B + ONCE_B_NUM - 1) // ONCE_B_NUM)
    max_blkn = max(max_blkn, 1)
    kmax_mblkn_stride = k_max.stride()[-2]
    kmin_mblkn_stride = k_min.stride()[-2]
    kernel = tmo_reduce_to_label_cache_kernel[grid](
        k,
        cu_seq_lens,
        slot_mapping,
        min_paged_cache,
        max_paged_cache,
        k_max,
        kmax_mblkn_stride,
        k_min,
        kmin_mblkn_stride,
        block_nums,
        B, H, D, block_size, max_blkn, store_paged_cache, ONCE_B_NUM
    )
    return k_max, k_min, block_nums, min_paged_cache, max_paged_cache
