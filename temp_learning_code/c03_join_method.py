# -*- coding: utf-8 -*-
# @Time : 2021/11/26 1:44
# @Author : zjk
# @File : c03_join_method.py
# @Project : YOLOX
# @IDE :PyCharm

# import numpy as np
# np.set_printoptions(suppress=True) # turn off scientific notation
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# import torch
import threading

number=0
def add_task(i=100000000):
    global number
    for j in range(i):
        number+=1

t=threading.Thread(target=add_task)
t.start()
# t.join() # 等待子线程t执行完毕，注释掉查看效果
print(number)