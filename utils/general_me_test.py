from matplotlib import pyplot as plt

import general_me
import time
import os
from pathlib import Path
from general_me import LOGGER,Profile, Timeout,WorkingDirectory,try_except #,methods
from general_me import *
# from models import tf
import atexit

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
# GitSourceTreeCodebase/yolov5
def profile_test():
    import math
    @Profile('create_list')
    def create_list(x):
        return [i for i in range(x)]

    with Profile('create_list'):
        print("with Profile('create_list')")
        x = 100
        l =  [i for i in range(x)]

    @Profile('cal_sin')
    @Timeout(3,timeout_msg='须在3秒内执行完毕！')
    def cal_sin(i):
        # time.sleep(3)
        r= [math.sin(math.sin(math.sin(x))) for x in range(i*i)]        
        # for _r in r:
        #     print(_r)
    
    with Timeout(3,timeout_msg='须在3秒内执行完毕！'):
        print("with Timeout(3,timeout_msg='须在3秒内执行完毕！')")
        i=1000
        r= [math.sin(math.sin(math.sin(x))) for x in range(i*i)] 


    # 开始测试
    create_list(10000)
    # cal_sin(2)
    cal_sin(100)

@try_except
@WorkingDirectory('/home/raise_except_for_test')
def working_dir_test():
    print("current in method",Path.cwd().resolve())


def intersect_dicts_test(dict_a=None,dict_b=None):
    if dict_a is None:
        dict_a={0:torch.zeros((2,3)),1:torch.ones((2,3))}
    if dict_b is None:
        dict_b={1:torch.ones((2,3))*11,2:torch.ones((2,3))*22}
    return intersect_dicts(dict_a,dict_b)


def type_holder(a,b,c,d) -> object:
    """
    就是做一个测试示例
    :param a: 第一个参数a
    :type a: int
    :param b: 第二个参数
    :type b: float
    :param c: 第三个参数
    :type c: torch.Tensor
    :param d: 第四个参数
    :type d: numpy.ndarray
    """
    ...


if __name__ == '__main__':
    import this
    a = [1, 2]
    b = [3, 4]
    c = [5, 6]
    d=sum((a, b, c), [])
    print(d)
    e=sum([a, b, c], [])
    print(e)

    # print(general_me.__name__,)
    # profile_test()

    # print("before",Path.cwd().resolve())
    # with WorkingDirectory('/home/zjk') as w:
    #     print("current",Path.cwd().resolve())    
    # print("after",Path.cwd().resolve())

    # print("**************************")
    # print("before",Path.cwd().resolve())
    # working_dir_test()
    # print("after",Path.cwd().resolve())
    # print("++++++++++++++++++++++++++++")
    # methods(Path,logging_out=False)
    # methods(Path,logging_out=True)
    # # print_args("Path",Path)
    # print_args("WorkingDirectory",WorkingDirectory)
    # # WorkingDirectory
    # init_seeds()
    # r=intersect_dicts_test()
    # print(r)
    a = torch.rand((10,10,100))

    # print_tensor(a, 'my_tensor')
    # get_latest_run(search_dir=r'D:\GitSourceTreeCodebase\yolov5')
    # files=search_files_by_extension(search_dir=r'D:\GitSourceTreeCodebase\yolov5', extension='py')
    # print(files)
    # p=use_config_dir()
    # print(p)
    # print(file_size())
    # print(file_size(p))
    # print(file_size(ROOT))

    # check_git_status()
    check_python()
    check_pytorch()
    from torch.torch_version import TorchVersion

    os.getcwdb()


    # path = Path('.')
    # paths= path.resolve()
    # print(paths)
    # patha=path.absolute()
    # print(patha)
    #
    # print('====activations.py')
    # path = Path('activation.py')
    # paths= path.resolve()
    # print("paths：",paths)
    # patha=path.absolute()
    # print("patha：",patha)
    #
    #
    # print('====activation.py,file not exists')
    # path = Path('activation.py')
    # paths= path.resolve()
    # print("paths：",paths)
    # patha=path.absolute()
    # print("patha：",patha)
    #
    # print('====activation.py,file not exists,strict=True')
    # path = Path('activation.py')
    # patha=path.absolute()
    # print("patha：",patha)
    #
    # paths= path.resolve(strict=True)
    # print("paths：",paths)


    # type_holder()

    # path = user_config_dir()
    # print(path)
    files=search_files_by_extension(search_dir=r'D:\GitSourceTreeCodebase\yolov5', extension='py')
    # print(files)
    #
    # s="YOLOv5 🚀 by Ultralytics, GPL - 3.0 license"
    # emojis(s)

    # check_requirements()
    # print("cv2.imshow():",check_imshow())
    # data = r"https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128_with_yaml.zip"
    # check_dataset(data, autodownload=True)
    # url2file()
    # make_divisible()
    x=range(300)
    y1=list(map(one_cycle(steps=50),x))
    y2 = list(map(one_cycle(steps=100), x))
    plt.plot(x, y1, color='r',label= 'steps=50')
    plt.plot(x, y2, color='b',label= 'steps=100')
    plt.legend()
    plt.show()
    print(one_cycle()(5))


