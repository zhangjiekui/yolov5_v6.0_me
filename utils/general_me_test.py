import general_me
import time
import os
from pathlib import Path
from general_me import LOGGER,Profile, Timeout,WorkingDirectory,try_except #,methods
from general_me import *
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

if __name__ == '__main__':
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

    print_tensor(a, 'my_tensor')
