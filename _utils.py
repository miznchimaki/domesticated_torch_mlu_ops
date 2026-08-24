import torch
from torch import Tensor
import os
import importlib.machinery
import inspect
from datetime import datetime
import sys

def enable_gen_case():
    def check_env_flag(name, default=''):
        return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
    enable_gen_case = check_env_flag('TMO_GEN_CASE')
    enalbe_dump_data = check_env_flag('TMO_GEN_CASE_DUMP_DATA')
    tmo_gen_case_op_name = os.getenv('TMO_GEN_CASE_OP_NAME')
    tmo_gen_case_overlap = check_env_flag('TMO_GEN_CASE_OVERLAP')
    tmo_gen_case_unset_log = check_env_flag('TMO_GEN_CASE_UNSET_LOG')
    op_names = tmo_gen_case_op_name.split(';') if tmo_gen_case_op_name else []
    tmo_gen_case_path = os.getenv('TMO_GEN_CASE_PATH', os.getcwd())
    return enable_gen_case, op_names, tmo_gen_case_path, enalbe_dump_data, tmo_gen_case_overlap, tmo_gen_case_unset_log

ENABLE_GEN_CASE, GEN_CASE_OP_NAMES, GEN_CASE_PATH, ENALBE_DUMP_DATA, GEN_CASE_OVERLAP, GEN_CASE_UNSET_LOG  = enable_gen_case()

def gen_case_dump(**kwargs):
    torch.save(kwargs['map'], kwargs['path'])

def to_cpu(tensor):
    if type(tensor) is not torch.Tensor:
        return tensor
    shape = tensor.shape
    stride = tensor.stride()
    is_contiguous = tensor.is_contiguous()
    dtype = tensor.dtype
    if is_contiguous:
        return tensor.cpu()
    else:
        t = torch.empty_strided(shape, stride, dtype=dtype, device="cpu")
        t.copy_(tensor)
        return t

def dump_info(value):
    info = dict()
    if value.__class__ in (list, tuple):
        cls_set = {elem.__class__ for elem in value}
        if {torch.Tensor, list, tuple} & cls_set:
            temp_li = []
            for elem in value:
                temp_li.append(dump_info(elem))
            info = {'type': value.__class__, 'has_compound': True, 'data': temp_li}
        else:
            info = {'type': value.__class__, 'has_compound': False, 'data': value}
    elif value.__class__ is dict:
        temp_dic = dict()
        for k, v in value.items():
            temp_dic[k] = dump_info(v)
        info = {'type': value.__class__, 'data': temp_dic}
    elif value.__class__ is torch.Tensor:
        if value is None:
            info = {'type': value.__class__, 'data': value}
        else:
            info = {'type': value.__class__,
                    'shape': value.shape,
                    'dtype': value.dtype,
                    'device': value.device.type,
                    'is_contiguous': value.is_contiguous(),
                    'stride': value.stride(),
                    'data': to_cpu(value) if value.dtype in (torch.int16, torch.int32, torch.int64) else ''}
    elif value.__class__ is torch.nn.parameter.Parameter:
        param = value.data
        assert param.__class__ is torch.Tensor, "torch.nn.parameter.Parameter.data must return torch.Tensor"
        info = dump_info(param)
    else:
        info = {'type': value.__class__, 'data': value}
    return info

def dump_pt_case(func, *args, **kwargs):
    op_name = func.__name__
    case_dir = os.path.join(GEN_CASE_PATH, 'tmo_gen_case', op_name)
    os.makedirs(case_dir, exist_ok=True)
    device_id = "card" + str(torch.mlu.current_device())
    if GEN_CASE_OVERLAP:
        file_name = f"{op_name}_{device_id}.pt"
    else:
        time_stamp = int(datetime.now().timestamp()*1e6)
        file_name = f"{op_name}_{device_id}_{time_stamp}.pt"
    case_path = os.path.join(case_dir, file_name)
    if not GEN_CASE_UNSET_LOG:
        print("[torch_mlu_ops] dump case ====> ", case_path)

    signature = inspect.signature(func)
    param_objs = [obj for obj in signature.parameters.values()]  # signature.parameters是有序的mapping
    params_map = {"op": op_name, "dump_data": ENALBE_DUMP_DATA}
    if ENALBE_DUMP_DATA:
        for obj, value in zip(param_objs[:len(args)], args):
            params_map[obj.name] = to_cpu(value)
        for obj in param_objs[len(args):]:
            params_map[obj.name] = to_cpu(obj.default)
        params_map = {**params_map, **kwargs}
    else:
        for obj, value in zip(param_objs[:len(args)], args):
            params_map[obj.name] = dump_info(value)
        for obj in param_objs[len(args):]:
            params_map[obj.name] = dump_info(obj.default)
        for k, v in kwargs.items():
            if k in signature.parameters.keys():
                params_map[k] = dump_info(v)
    gen_case_dump(map = params_map, path = case_path)

def dump_case(func):
    def wrapper_dump(*args, **kwargs):
        dump_pt_case(func, *args, **kwargs)
        return func(*args, **kwargs)
    def wrapper_nothing(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper_dump if (ENABLE_GEN_CASE and \
                            (GEN_CASE_OP_NAMES == [] or func.__name__ in GEN_CASE_OP_NAMES)) \
                        else wrapper_nothing

def add_gen_case_decorator(frame):
    if ENABLE_GEN_CASE:
        _current_module = inspect.getmodule(frame)
        members = dict()
        for k, v in inspect.getmembers(_current_module):
            members[k] = v

        func_names = members['__FUNCTIONS__'] if GEN_CASE_OP_NAMES == [] else GEN_CASE_OP_NAMES
        for name in func_names:
            if hasattr(_current_module, name):
                setattr(_current_module, name, dump_case(members[name]))

dtype2str_map = { torch.half: "half",
                  torch.float16: "half",
                  torch.float: "float",
                  torch.bfloat16: "bfloat16",
                  torch.int32: "int32",
                  torch.int8: "int8",
                  torch.float8_e4m3fn: "float8_e4m3fn",
                  torch.float8_e5m2: "float8_e5m2"
                }

def torchDtype2Str(torch_dtype: torch.dtype) -> str:
    assert torch_dtype in dtype2str_map, "unrecognized torch type: {}".format(torch_dtype)
    return dtype2str_map[torch_dtype]

def get_tmo_link_flags() -> str:
    link_dir = os.path.join(os.path.dirname(__file__), "lib")
    return f"-L{link_dir} -ltorch_mlu_ops"

def get_tmo_compile_flags() -> str:
    include_dir = os.path.join(os.path.dirname(__file__), "include")
    return f"-I{include_dir}"

def get_tmo_library_path() -> str:
    link_dir = os.path.join(os.path.dirname(__file__), "lib")
    return link_dir

def get_tmo_header_path() -> str:
    include_dir = os.path.join(os.path.dirname(__file__), "include")
    return include_dir

def get_tmo_head_files() -> list:
    '''
    return head files
    '''
    include_dir = os.path.join(os.path.dirname(__file__), "include")
    hs = []
    for l in os.walk(include_dir):
        for k in l[2]:
            hs.append(os.path.join(l[0], k))
    return hs
