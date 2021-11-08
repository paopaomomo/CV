import numpy as np
import matplotlib.pyplot as plt

vector =[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
f_v=np.fft.fft(vector)
import pdb
pdb.set_trace()
print(vector)
print(f_v)

def get_A_b(z):
    A =np.abs(z)
    b = np.arcsin(z.real/(A+0.0000001))
    return A,b

def generate_sin(A,w,b):
    # A*sin(wx+b))
    # x=0, b
    # x=11, w*11+b
    # w = pi/6,b=-3
    #T = 12
    
    #x1=np.linspace(-np.pi*0.5,np.pi*1.5,12)
    x1=np.linspace(b,2*np.pi*w*16+b,16)
    y=np.sin(x1)*A
    return y
generate_sin(1,np.pi/8,-3)
pdb.set_trace()
print(vector)
def get_vector(w_times):
    vector_sum=np.zeros((16))
    for i in range(0,8):
        A,b=get_A_b(f_v[i])
        #w = np.pi*2/(i*w_times+0.00000001)
        w = i*w_times
        vector_tmp=generate_sin(A*2/16,w,b)
        if i==0:
             vector_tmp = vector_tmp*2
        print(A,w,b)
        print(vector_tmp)
        vector_sum= vector_sum+vector_tmp
    return vector_sum
w_times=1
vector_sum = get_vector(w_times)
print(vector)
print(vector_sum)
pdb.set_trace()
print("end")


