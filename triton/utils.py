import triton.backends.mlu.driver as driver
import torch
_cached_total_core_num = None

def get_total_core_num():
    global _cached_total_core_num
    if _cached_total_core_num is None:
        device_prop = driver.BangUtils().get_device_properties(torch.mlu.current_device())
        _cached_total_core_num = device_prop["cluster_num"] * device_prop["core_num_per_cluster"]
    return _cached_total_core_num
