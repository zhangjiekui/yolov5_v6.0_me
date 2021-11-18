# -*- coding: utf-8 -*-
# @Time : 2021/11/16 22:09
# @Author : zjk
# @File : general_me.py
# @Project : YOLOX
# @IDE :PyCharm

# https://github.com/ultralytics/yolov5/blob/master/utils/general.py

import contextlib
import glob
import logging
import math
import os, platform
import random
import re
import shutil
import signal
import time
import urllib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
from torch.functional import Tensor
import torchvision
import yaml

# from utils.downloads import gsutil_getsize
# from utils.metrics import box_iou,fitness
from downloads import gsutil_getsize
from metrics import box_iou,fitness

# 环境设置
torch.set_printoptions(precision=5,linewidth=320,profile='long', sci_mode=False)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def set_logging(name=None, verbose=True):
    rank = int(os.getenv('RANK',-1)) # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    # 指定name，返回一个名称为name的Logger实例。如果再次使用相同的名字，返回同一个对象。未指定name，返回Logger实例，名称是root，即根Logger。
    # Return a logger with the specified name, creating it if necessary.
    return logging.getLogger(name)

LOGGER = set_logging(__name__)  # define globally (used in train.py, val.py, detect.py, etc.)
print(__name__) # 本身执行时，__name__ =‘__main__’, 被别的文件导入时，__name__ =‘本文件的名称’,

class Profile(contextlib.ContextDecorator):
    """
    Usage: @Profile() decorator or 'with Profile():' context manager
    """
    def __init__(self,name=None):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.name is None:
            print(f'Profile results: {time.time() - self.start:.5f}s')
        else:
            print(f'Profile method "{self.name}" results: {time.time() - self.start:.5f}s')

class Timeout(contextlib.ContextDecorator):
    """
    Usage:    @Timeout(seconds) decorator
          or 'with Timeout(seconds):' context manager
    """
    def __init__(self, seconds, *, timeout_msg = '', suppress_timeout_errors = False):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._timeout_handler) # Set handler for SIGALRM
        signal.alarm(self.seconds) # start countdown for SIGALRM to be raised
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
       signal.alarm(0)  # Cancel SIGALRM if it's scheduled
       if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
           return True

class WorkingDirectory(contextlib.ContextDecorator):
    '''
    Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    '''
    def __init__(self,new_dir) -> None:
        super().__init__()
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve() # current dir
    def __enter__(self):
        os.chdir(self.dir)

    def __exit__(self,exc_type,exc_value,exc_traceback):
        os.chdir(self.cwd)

def try_except(func):
    '''
    try-except function. Usage: @try_except decorator
    '''
    def handler(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            LOGGER.error(colorstr('red',"ErrorFound  .............."))
            LOGGER.error(f"    {e}")
            LOGGER.error(colorstr('red',"ErrorIgnored.............."))

    return handler

def methods(instance,logging_out=True):
    # Get class/instance methods
    _methods = [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith('__')]
    methods = [f for f in _methods if not f.startswith('_')]
    _m = [f for f in _methods if f.startswith('_')]
    methods.extend(_m)
    if logging_out:
        print(colorstr(f'Methods in {instance}'))
        for m in methods:
            print(f"  ..  {m}")
    return methods

def print_args(name, opt):
    # Print argparser arguments
    LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

def init_seeds(seed=0,logging_out=True):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)
    if logging_out:
       LOGGER.info(colorstr("init_seeds:  ")+colorstr('bright_red',f'{seed=},{cudnn.benchmark=},{cudnn.deterministic=}. cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible '))

@try_except
def intersect_dicts(da:dict, db:dict, exclude:tuple=()):
    # Dictionary(torch.tensor or numpy.array) intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

@try_except
def print_tensor(t:torch.tensor,name:str=None):
    t=[t] if isinstance(t,torch.Tensor) else t
    for elem in t:
        l = len(elem.shape)
        prefix = f"Tensor({name})" if name else "Tensor"
        print(colorstr(f"{prefix} Shape")+f'({",".join([str(elem.size(i)) for i in range(l)])}),'+ colorstr(f'Device:')+f'{elem.device},'+ colorstr(f'Dtype:')+f'{elem.dtype},'+ colorstr(f'Numel:')+f'{elem.numel()},'+ colorstr('Values:'))
        print("  "+colorstr('t[ 0]:\n')+ f'        ({elem[0].detach().numpy()})'[0:120]+'...')
        print("            "+ colorstr('......\n'))
        print("            "+ colorstr('......\n'))
        print("  "+colorstr('t[-1]:\n')+ f'        ({elem[0].detach().numpy()})'[0:120]+'...')










