import threading
import time
import multiprocessing
multiprocessing.set_start_method('fork')

T=multiprocessing.Process
T=threading.Thread

def task(name,i):
    time.sleep(i)
    print(name,i)

# if __name__ == '__main__':

# 进程必须放到main函数下才能正确执行
d={'threading':threading.Thread,'multiprocessing':multiprocessing.Process}

for k,v in d.items():
    print("========启动"+k)
    for i in range(10):
        # t = v(target=task, args=(k,i))
        t = v(target=task, args=[k, i])
        t.start()
print("主程序已执行完最后一行！")