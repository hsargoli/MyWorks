# FIBONACI
import time
AA0011=time.time()
def fibo(num=1):
    if num==1:
        f=[1]
    else:
        NUM=num-2
        f=[1,1]
        for i in range(NUM):
            a=f[-1]
            b=f[-2]
            A=a+b
            f.append(A)
    return f
print(fibo(20))
AA00=time.time()-AA0011
print(AA00)