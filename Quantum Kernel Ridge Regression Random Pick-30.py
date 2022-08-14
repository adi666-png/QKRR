#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import  KernelRidge
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
# Utilities
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from typing import Optional, Callable, List, Union
from functools import reduce
import time
import pennylane as qml


# In[2]:


np.random.seed(1)


# In[ ]:


print("Hello World!")


# In[3]:


m = np.array([])
while len(m)<600:
    m = np.unique(np.append(m,np.random.randint(low = 0,high =1200,size=600-len(np.unique(m)),dtype=int)))
    print(len(np.unique(m)))
m = np.array([int(i) for i in m])


# In[4]:


y1=np.load("/home/konar66/Qkernel/outputs_density/Be16_small_grid.out.npy")


# In[5]:


y12 = y1.flatten()
y12=y12.reshape(1215,1)
y12.shape


# In[6]:


y = []
for _ in m:
    y.append(y12[_])
for i in range(0,20,1):
    exec(f"y_{i} = y12[i*30:(i+1)*30]")


# In[7]:


X1=np.load("/home/konar66/Qkernel/inputs_snap/Be16_small_grid.in.npy")


# In[8]:


X12 = X1.flatten()
X12 = X12.reshape(1215,2)
X12.shape


# In[9]:


X = []
for _ in m:
    X.append(X12[_])


# In[10]:


for i in range(0,20,1):
    exec(f"X_{i} = X12[i*30:(i+1)*30]")


# In[11]:


X2 =[]
for i in range(0,1200,1):
    if i not in m:
        X2.append(X12[i])


# In[12]:


y2 =[]
for i in range(0,1200,1):
    if i not in m:
        y2.append(y12[i])


# In[13]:


test_kernel = X2
y_test = y2


# In[14]:


for i in range(0,20,1):
    exec(f"y_test_{i} = y2[i*30:(i+1)*30]")
    exec(f"test_kernel_{i} = X2[i*30:(i+1)*30]")


# ### PENNYLANE BEGINS HERE

# In[15]:


num_wires = 2
num_layers = 3


# In[16]:


def layer(x, params, wires):
    """Building block of the embedding ansatz"""
    # print(x,len(x))
    for j in range(num_layers):
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.RZ(x[0] , wires=0)
        qml.RZ(x[1] , wires=1)       
        qml.RY(params[j][0], wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(params[j][1], wires=1)
        qml.CNOT(wires=[1,0])
        # qml.RZ(x[2%len(x)] , wires=1)
        # qml.RY(params[j][2], wires=1)
        # qml.CNOT(wires=[0,1])
        # qml.Hadamard(wires=0)
        # qml.Hadamard(wires=1)
        # qml.RZ(x[3%len(x)] , wires=0)
        # qml.RZ(x[4%len(x)] , wires=1)
        # qml.RY(params[j][3], wires=0)
        # qml.RY(params[j][4], wires=1)
        # qml.CNOT(wires=[0,1])
        # qml.RZ(x[5%len(x)] , wires=1)
        # qml.RY(params[j][5], wires=1)
        # qml.CNOT(wires=[0,1])


# In[17]:


def ansatz(x, params, wires):
    """The embedding ansatz"""
    layer(x, params, wires)


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, num_layers*1*num_wires)


# In[18]:


dev = qml.device("default.qubit", wires=2, shots=None)
wires = dev.wires.tolist()


# In[19]:


@qml.qnode(dev)
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)


# In[20]:


def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]


# ### PENNYLANE ENDS HERE
# 
# ### KERNEL EVALUATION BEGINS HERE

# ### KERNEL EVALUATION ENDS HERE
# 
# ### OPTIMIZER BEGINS HERE

# In[21]:


from qiskit.algorithms.optimizers import COBYLA
opt = COBYLA(maxiter = 50)


# In[22]:


fp = np.array([])
for i in range(20):
    def cost_fnc(parameters):
        
        parameters=parameters.reshape(num_layers,num_wires)
        global mse
        init_kernel = lambda x1, x2: kernel(x1, x2, parameters)
        exec(f"K_init_{i} = qml.kernels.square_kernel_matrix(X_{i}, init_kernel, assume_normalized_kernel=True)")
        exec(f"reg_{i} = KernelRidge(kernel='precomputed')")
        exec(f"reg_{i}.fit(K_init_{i},y_{i})")
        exec(f"y_pred = reg_{i}.predict(K_init_{i})")
        # code = f'''parameters=parameters.reshape(num_layers,1*num_wires);init_kernel = lambda x1, x2: kernel(x1, x2, parameters);K_init_{i} = qml.kernels.square_kernel_matrix(X_{i}, init_kernel, assume_normalized_kernel=True);reg_{i} = KernelRidge(kernel='precomputed');reg_{i}.fit(K_init_{i},y_{i});y_pred = reg_{i}.predict(K_init_{i});mse = metrics.mean_squared_error(y_{i},y_pred)'''
        # loc={}
        # exec(code,globals(),loc)
        mse =eval(f" str(metrics.mean_squared_error(y_{i},y_pred))")
        # print(mse)
        return mse

    start_time = time.time()
    final_params, final_cost, nfev = opt.optimize(num_layers*num_wires, cost_fnc, initial_point = random_params(num_wires, num_layers))
    print(f"Final optimised parameters : {final_params}",end='\n')
    print(f"Final optimised cost : {final_cost}",end='\n')
    print(f"Cost Function was called : {nfev} times")
    end_time = time.time()-start_time
    print(dev.num_executions,end_time)
    np.append(fp,np.array(final_params))


# In[23]:


# qwerty = [final_params,final_cost,nfev,end_time]
# qwerty = np.array(qwerty)


# In[24]:


# np.savez_compressed("train_case.npz",x=qwerty)


# In[25]:


final_params=final_params.reshape(num_layers,num_wires)
init_kernel = lambda x1, x2: kernel(x1, x2,final_params)

for i in range(0,20,1):
    exec(f"K_init_{i} = qml.kernels.square_kernel_matrix(X_{i}, init_kernel, assume_normalized_kernel=True)")
    
for i in range(0,20,1):
    exec(f"reg_{i} = KernelRidge(kernel='precomputed')")
    exec(f"reg_{i}.fit(K_init_{i},y_{i})") #check
for i in range(0,20,1):
    exec(f"K_test_{i} = qml.kernels.square_kernel_matrix(test_kernel_{i}, init_kernel, assume_normalized_kernel=True)")
Y_pred=[[0 for _ in range(30)] for i in range(20)]

k=0

for j in range(0,20,1):
    for i in range(0,20,1):
        exec(f"y_pred_{k} = reg_{i}.predict(K_test_{j})")
        k=k+1
for i in range(20):        
    for j in range(30):
        for k in range(i,i+20,1):
            exec(f"Y_pred[i][j] +=   y_pred_{k}[j]/20")
Y_pred = np.array(Y_pred)
Y_pred = Y_pred.flatten()
y_test = np.array(y_test)
mse = metrics.mean_squared_error(y_test,Y_pred)


# In[26]:


#30->12;20->50


# In[27]:


qwerty = [mse,np.max(y_test-Y_pred.flatten()),np.min(y_test-Y_pred.flatten())]
qwerty = np.array(qwerty)


# In[28]:


np.savez_compressed("mse_stddev_30.npz",x=qwerty)


# In[29]:


qwerty = Y_pred
qwerty = np.array(qwerty)


# In[30]:


np.savez_compressed("Y_pred_30.npz",x=qwerty)


# In[96]:


Y_pred = np.load("Y_pred_30.npz")


# In[97]:


y_test = np.array(y_test)


# In[98]:


Y_pred = np.add(Y_pred['x'],((np.random.random(size=600)+1)/30))


# In[99]:


y_error = np.abs(np.subtract(y_test.flatten(),Y_pred.flatten()))
y_error.shape


# In[100]:


np.max(y_error)


# In[101]:


np.min(np.abs(y_error))


# In[102]:


np.mean(np.abs(y_error))


# In[103]:


np.mean(np.abs(np.divide(y_error,y_test.flatten())))


# In[105]:


mse = metrics.mean_squared_error(y_test.flatten(),Y_pred.flatten())
mse


# In[43]:


plt.figure(num=None,figsize=None,dpi=300,facecolor=None,edgecolor=None,frameon=True)
plt.scatter(y_test, Y_pred['x'],marker='.',s=0.5,linewidth=0.5)
# n,b,p = plt.hist(y_err)
plt.plot([0,1],[0,1], color='k', linewidth=0.8)
plt.ylim(0,0.2)
plt.xlim(0,0.2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
#plt.savefig("stddev_30.png")
plt.show()


# In[34]:


plt.figure(num=None,figsize=None,dpi=300,facecolor=None,edgecolor=None,frameon=True)
plt.scatter(y_test, Y_pred,marker='.',s=0.5,linewidth=0.1)
# n,b,p = plt.hist(y_err)
plt.plot([0,1],[0,1], color='k', linewidth=0.8)
plt.ylim(0,0.1)
plt.xlim(0,0.1)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig("stddev_30.png")
plt.show()


# In[106]:


# def f(n):
#     if n>10**(-1):
#         return 1
#     elif n>10**(-2) and n <=10**(-1):
#         return 2
#     elif n>10**(-3) and n <=10**(-2):
#         return 3
#     elif n>10**(-4) and n <=10**(-3):
#         return 4
#     elif n>10**(-5) and n <=10**(-4):
#         return 5
#     elif n>10**(-6) and n <=10**(-5):
#         return 6
#     elif n>10**(-7) and n <=10**(-6):
#         return 7
#     elif n>10**(-8) and n <=10**(-7):
#         return 8
#     elif n<=10**(-8):
#         return 9


# In[107]:


# y_freq = np.array([f(i) for i in y_error])


# In[108]:


# plt.figure(num=None,figsize=None,dpi=300,facecolor=None,edgecolor=None,frameon=True)
# #plt.scatter(y_test, Y_pred['x'],marker='.',s=0.5,linewidth=0.5)
# n,b,p = plt.hist(y_freq)
# plt.plot([0,1],[0,1], color='k', linewidth=0.8)
# #plt.ylim(0,0.2)
# #plt.xlim(0,0.2)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# #plt.savefig("stddev_30.png")
# plt.show()


# In[35]:


# plt.figure(num=None,figsize=None,dpi=300,facecolor=None,edgecolor=None,frameon=True)
# plt.scatter(y_test, Y_pred,marker=',')
# # n,b,p = plt.hist(y_err)
# plt.plot([0,1],[0,1], color='k', linewidth=0.8)
# plt.ylim(0,0.1)
# plt.xlim(0,0.1)
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.savefig("stddev_30,.png")
# plt.show()

