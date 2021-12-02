# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
PyTorch utils
"""

import datetime
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings
warnings.filterwarnings('ignore')


from utils.general import LOGGER
from torch_utils_me import *

def profile_test():
    # YOLOv5 speed/memory/FLOPs profiler
    input = torch.randn(16, 3, 640, 640)
    m0 = nn.Sigmoid()
    m1 = lambda x: x * torch.sigmoid(x)
    m2 = nn.SiLU()
    m3 = nn.Conv2d(3,3,kernel_size=3,padding='same')
    m4 = torchvision.models.mobilenet_v3_large(pretrained=False)
    m5 = torchvision.models.mobilenet_v3_small(pretrained=False)
    from utils.torch_utils import profile
    profile(input, [m0, m1, m2 , m3,m4,m5], n=10)  # profile over 100 iterations

def subprocess_check_output_test():
    import subprocess
    cmd = r"ping www.baidu.com"
    result = subprocess.check_output(cmd)
    print(result.decode("gbk"))
    cmd = r"dir"
    # result = subprocess.check_output([cmd,r'D:\GitSourceTreeCodebase\yolov5\utils'],shell=True, stderr=subprocess.STDOUT,bufsize=0).decode("gbk")[:-1]
    result = subprocess.check_output([cmd, r'D:\GitSourceTreeCodebase\yolov5\utils'], shell=True,
                                     stderr=subprocess.STDOUT).decode("gbk")[:-1]
    print(result)

if __name__ == '__main__':
    subprocess_check_output_test()
    print(__file__,end=': ')
    print(date_modified(path=__file__))
    print(git_describe())
    print(select_device(device='0'))
    profile_test()