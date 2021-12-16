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

from utils.general import LOGGER
from utils.general import colorstr
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

'''
1. 定义一个上下文管理器类【参见general.py 中的 class Timeout(contextlib.ContextDecorator)】
    class MyResource:
        # __enter__ 返回的对象会被with语句中as后的变量接受
        def __enter__(self):
            print('connect to resource')
            return self
    
        def __exit__(self, exc_type, exc_value, tb):
            print('close resource conection')
    
        def query(self):
            print('query data')
        # 类中有两个特殊的魔术方法:
        # __enter__: with语句中的代码块执行前, 会执行__enter__, 返回的值将赋值给with句中as后的变量.
        # __exit__: with语句中的代码块执行结束或出错, 会执行_exit__
        
2. 使用contextmanager
    from contextlib import contextmanager
    class MyResource:
        def query(self):
            print('query data')
    
    @contextmanager
    def make_myresource():
        print('start to connect')
        yield MyResource()
        print('end connect')
        pass         
        
3. 方法调用及结果
        # with MyResource() as r:
        #   r.query()
        
        # 调用结果
        # connect to resource
        # query data
        # close resource conection

4.@contextmanager还有另一个用法，与上下文管理器无关，先暂时称之为获取器吧。
     def book_mark():#自动补全书名号
        print("《", end='')
        yield
        print("》", end='')
    
    with book_mark():
        print("生命中不能承受之轻", end='')
5. yield 关键字，可以让函数执行到yield处【或返回MyResource()】之后会处于一个中断的状态，
   在外面执行了print("生命中不能承受之轻", end='')【或MyResource()的r.query()】 之后，
   再次回到 yield 这儿继续执行后面的代码。带有 yield 的函数又称作为生成器   
'''
'''完整代码实例
def context_management_test():
    import contextlib
    from contextlib import contextmanager
    class MyResource1: # 不一定要继承
    # class MyResource1(contextlib.ContextDecorator):
        # __enter__ 返回的对象会被with语句中as后的变量接受
        def __enter__(self):
            print('1 connect to resource')
            return self

        def __exit__(self, exc_type, exc_value, tb):
            print('1 close resource conection')

        def query(self):
            print('1 query data')


    class MyResource2(contextlib.ContextDecorator):
        def query(self):
            print('2 query data')

    @contextmanager
    def make_myresource():
        print('2 start to connect')
        yield MyResource2()
        print('2 end connect')
        pass

    with MyResource1() as r:
        r.query()
    print("===========================")
    with make_myresource() as r:
        r.query()
'''
# done
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'

# done
def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository
# done
def select_device(device='', batch_size=None, newline=True,logger_out=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    if not newline:
        s = s.rstrip()
    if logger_out:
        LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')

# done
def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# done
def profile(input, ops, n=10, device=None):
    # YOLOv5 speed/memory/FLOPs profiler
    #
    # Usage:
    #     input = torch.randn(16, 3, 640, 640)
    #     m1 = lambda x: x * torch.sigmoid(x)
    #     m2 = nn.SiLU()
    #     profile(input, [m1, m2], n=100)  # profile over 100 iterations
    ops = ops if isinstance(ops, list) else [ops]
    print("")
    print(colorstr(f'Proile Modles .....'))
    results = []
    device = device or select_device(logger_out=False)
    print(f"{'ModuleName':>60s}{'Device':>10s}{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
          f"{'input':>24s}{'output':>60s}")

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad_ = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, 'to') else m  # device
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                if hasattr(m, '__name__'):
                    name = m.__name__
                    _type = "Function"
                    name = "".join(list(filter(str.isalnum, name)))
                else:
                    name = m.__class__
                    name = str(name).split("\'")[1]
                    _type = "Class"


                # name = m.__name__ if hasattr(m, '__name__') else m.__class__
                # _type = "Function" if hasattr(m, '__name__') else "Class"
                #
                # name = str(name).split("\'")[1]
                # name = "".join(list(filter(str.isalnum, name)))
                name = f'{_type}({name})'
            except:
                name = "Exception(Unknown)"

            try:
                # FLOPs：注意s小写，是floating point operations的缩写（s表复数），
                # 意指浮点运算数，理解为计算量。可以用来衡量算法 / 模型的复杂度
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs
                # flops = thop.profile(m, inputs=(x,), verbose=True)[0]   # GFLOPs
            except:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception as e:  # no backward method
                        # print(e)  # for debug
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                # s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
                # s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
                s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list['+"|".join([str(tuple(e.shape)) for e in y])+"]"
                s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list['+"|".join([str(tuple(e.shape)) for e in y])+"]"
                p = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters
                print(f'{name:>60s}{str(device):>10s}{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>70s}')
                results.append([p, flops, mem, tf, tb, s_in, s_out, device,name])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results

# done
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

# done
def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

# done
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
# done
# list(model.named_modules(remove_duplicate=False))
def find_modules(model, mclass=nn.Conv2d,remove_duplicate=True):
    '''
    测试用例见train.py
    from utils.torch_utils import find_modules
    ms = find_modules(model, mclass=models.yolo.Detect)
    ms2 = find_modules(model, mclass=models.yolo.Detect,remove_duplicate=False)
    :param model:
    :param mclass:
    :param remove_duplicate:
    :return:
    '''
    # Finds layer indices matching module class 'mclass'
    # return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)] #todo，该方法未曾调用过，而且没有model.module_list属性
    all_layers = len(list(model.named_modules(remove_duplicate=False)))
    all_layers_remove_duplicate=len(list(model.named_modules(remove_duplicate=True)))
    module_idx_list = []
    if remove_duplicate:
        module_idx_list = [i for i, m in enumerate(model.modules()) if isinstance(m, mclass)]
    else:
        i = 0
        all_named_modules_with_duplicated = model.named_modules(remove_duplicate=False) #generator
        for _, module in all_named_modules_with_duplicated:
            if isinstance(module, mclass):
                module_idx_list.append(i)
            i+=1
    return module_idx_list,all_layers,all_layers_remove_duplicate

# done
def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b.item() / a

# done
def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    print('current %.2f global sparsity' % sparsity(model))
    for name, m in model.named_modules(remove_duplicate=False):
        if isinstance(m, nn.Conv2d):
            # print("    正常：",m.state_dict().keys())
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            # print("    l1_unstructured:",m.state_dict().keys())
            prune.remove(m, 'weight')  # make permanent
            # print("    remove:",m.state_dict().keys())
    print('             ...after pruned %.3g global sparsity' % sparsity(model))

# to be done
def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs

        # stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        # device = next(model.parameters()).device
        # img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=device)  # input
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        img1 = torch.zeros((1, model.yaml.get('ch', 3), img_size, img_size), device=device)
        img16 = torch.zeros((16, model.yaml.get('ch', 3), img_size, img_size), device=device)
        # # [[name, p, flops, mem, tf, tb, s_in, s_out]]
        # out = profile([img1,img16], deepcopy(model),  device=device)
        # flops1_cpu  = out[0][2]  # stride GFLOPs
        # flops16_cpu = out[1][2]   # stride GFLOPs
        flops_gpu = profile([img1,img16], deepcopy(model),  device='cuda:0')  # stride GFLOPs
        flops1_gpu  = flops_gpu[0][2]  # stride GFLOPs
        flops16_gpu = flops_gpu[1][2]
        # print(f"{flops1_gpu=},{flops16_gpu=}")
        # print(f"{flops1_cpu=},{flops16_cpu=},{flops1_gpu=},{flops16_gpu=}")


        # profile()
        # flops2 = thop.profile(deepcopy(model), inputs=(img1,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        # fs = ', %.1f GFLOPs' % (flops1_gpu * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
        fs = f"GFLOPs [batch_size(1) = {flops1_gpu:.1f}, batch_size(16) = {flops16_gpu:.1f}]"
    except (ImportError, Exception):
        fs = 'GFLOPs计算出错'

    LOGGER.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients, {fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
def model_named_params_statedict(model):
    for name,param in model.named_parameters():
        splits=name.split('.')
        _name=''
        for s in splits:
            try:
                ints=int(s)
                _s=f'[{ints}]'
                # _name+=_s
            except:
                _s=s if len(_name)==0 else '.'+s
                # _name+=_s
            _name+=_s
        # compare = id(eval("model."+_name))==id(param)
        # assert (param.data != model.model.state_dict()[name.replace('model.', '')]).sum()==0
        # # 值相同，但id不一样
        # id_compare=id(param.data) == id(model.model.state_dict()[name.replace('model.', '')])
        # print(f'{id_compare=}')
        # # print(name,_name,compare)
        # assert id(eval("model."+_name))==id(param)
    # print('over')

def test_fucs(model):
    import models
    # from utils.torch_utils import find_modules, is_parallel
    # print(is_parallel(model))
    # ms1 = find_modules(model, mclass=models.yolo.Detect)
    # ms2 = find_modules(model, mclass=models.yolo.Detect, remove_duplicate=False)
    # ms3 = find_modules(model, mclass=nn.Conv2d)
    # ms4 = find_modules(model, mclass=nn.Conv2d, remove_duplicate=False)
    # return ms1, ms2, ms3, ms4
    # sparsity_ration = sparsity(model)
    prune(model)

    model_named_params_statedict(model)




if __name__ == '__main__':
    model_info()
