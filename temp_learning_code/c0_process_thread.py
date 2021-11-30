# -*- coding: utf-8 -*-
# @Time : 2021/11/26 0:24
# @Author : zjk
# @File : c0_process_thread.py
# @Project : YOLOX
# @IDE :PyCharm


import threading
import time
import multiprocessing
# multiprocessing.set_start_method('fork') #Windows下无法使用，需放到main函数下执行

def task(name,i):
    time.sleep(i)
    print(name,i)

if __name__ == '__main__':

    # Windows（spawn方式）下进程必须放到main函数下才能正确执行
    # 但Linux（fork方式）下无所谓
    # multiprocessing.set_start_method('fork')

    '''
    首先fork和spawn都是构建子进程的不同方式，区别在于：

    fork：除了必要的启动资源外，其他变量，包，数据等都继承自父进程，并且是copy-on-write的，也就是共享了父进程的一些内存页，因此启动较快，但是由于大部分都用的父进程数据，所以是不安全的进程
    spawn：从头构建一个子进程，父进程的数据等拷贝到子进程空间内，拥有自己的Python解释器，所以需要重新加载一遍父进程的包，因此启动较慢，由于数据都是自己的，安全性较高
    
    实际使用中可以根据子进程具体做什么来选取用fork还是spawn~

    '''
    d={'threading':threading.Thread,'multiprocessing':multiprocessing.Process}

    for k,v in d.items():
        print("========启动"+k)
        for i in range(10):
            # t = v(target=task, args=(k,i))
            t = v(target=task, args=[k, i])
            t.start()
    print("主程序已执行完最后一行！")