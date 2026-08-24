import torch
from torch_mlu_ops._ops import *
if torch.__version__ >= '2.3.0':
    from .abstract import *

from torch_mlu_ops.config_inductor import *
from torch_mlu_ops._version import __version__
