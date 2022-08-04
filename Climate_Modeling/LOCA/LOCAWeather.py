from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
import argparse
from jax.example_libraries.stax import Dense, Gelu, Relu
from jax.example_libraries import stax
import os

import timeit

from jax.example_libraries import optimizers

from absl import app
import jax
from jax import vjp
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import norm

from jax import random, grad, vmap, jit, pmap
from functools import partial 

from torch.utils import data

from scipy import interpolate

from tqdm import trange
from math import log, sqrt, sin, cos

import itertools
import torch

import scipy.signal as signal

from kymatio.numpy import Scattering2D

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmax(memory_available))

os.environ['CUDA_VISIBLE_DEVICES']= "0"

def input_construction(U,X,m=100, Nx=100, Ny=100, N=1000,du=1,dx=2):
    dx = 1./(Nx-1)
    dy = 1./(Ny-1)
    x = np.arange(0,1+dx,dx)
    y = np.arange(0,1+dy,dy)
    U = U.reshape(Nx,Ny)
    u = interpolate.interp2d(x,y,U[:,:],kind="cubic")
    X_new = np.linspace(0,1,num=int(sqrt(m)))
    Y_new = np.linspace(0,1,num=int(sqrt(m)))
    XX_new, YY_new = np.meshgrid(X_new,Y_new)
    U_all = np.zeros((int(sqrt(m)),int(sqrt(m))))
    U_all[:,:] = u(X_new, Y_new)
    X_all = np.concatenate((XX_new.flatten()[:,None],YY_new.flatten()[:,None]),-1)
    U_all = U.reshape(int(sqrt(m))*int(sqrt(m)),du)
    return U_all, X_all

def output_construction(s,Y,P=100,ds=1, dy=2, N=1000,Nx=100,Ny=100):
    s = s.reshape(Nx,Ny)
    x = np.random.randint(Nx, size=P)
    y = np.random.randint(Ny, size=P)
    lontest = np.linspace(0,355,num=72)/360
    lattest = (np.linspace(90,-87.5,num=72) + 90.)/180.
    XX, YY = np.meshgrid(lontest, lattest)
    Y_all = np.concatenate((XX[x][range(P),y][:,None],YY[x][range(P),y][:,None]),axis=-1)
    s_all = s[x][range(P), y][:, None]
    return s_all, Y_all

def pairwise_distances(dist,**arg):
    return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))

def peuclid_distance(x,y,square=True):
    XX = jnp.einsum('ik,ik->i',x,x)
    YY = jnp.einsum('ik,ik->i',y,y)
    XY = jnp.einsum('ik,jk->ij',x,y)
    return XX[:,np.newaxis]+YY[np.newaxis,:] - 2*XY

def euclid_distance(x,y,square=True):
    XX=jnp.dot(x,x)
    YY=jnp.dot(y,y)
    XY=jnp.dot(x,y)
    return XX+YY-2*XY

class DataGenerator(data.Dataset):
    def __init__(self, inputsxu, y, s,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.inputsxu  = inputsxu
        self.y = y
        self.s = s
        
        self.N = inputsxu.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    # @partial(jit, static_argnums=(0,))
    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs,outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        inputsxu  = self.inputsxu[idx,:,:]
        y = self.y[idx,:,:]
        inputs = (inputsxu, y)
        return inputs, s

class PositionalEncodingY:
    def __init__(self, Y, d_model, max_len = 100, H=20):
        self.d_model = int(np.ceil(d_model/4)*2)
        self.Y = Y
        self.max_len = max_len
        self.H = H
        self.vdistance_function = vmap(pairwise_distances(euclid_distance))

    @partial(jit, static_argnums=(0,))
    def forward(self, x):
        pex = np.zeros((x.shape[0], self.max_len, self.H))
        pey = np.zeros((x.shape[0], self.max_len, self.H))
        T = jnp.take(self.Y, 0, axis=2)[:,:,None]
        X = jnp.take(self.Y, 1, axis=2)[:,:,None]
        positionT = jnp.tile(T,(1,1,self.H))
        positionX = jnp.tile(X,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,0::2], jnp.cos(positionT[:,:,0::2] * div_term))
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,1::2], jnp.sin(positionT[:,:,1::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,0::2], jnp.cos(positionX[:,:,0::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,1::2], jnp.sin(positionX[:,:,1::2] * div_term))
        pos_embedding =  jnp.concatenate((pex,pey),axis=-1)
        x =  jnp.concatenate([x, pos_embedding], -1)
        return x

def continouswavetransf(sig, l=100, m=100, training_batch_size = 1):
    scattering = Scattering2D(J=1, L=8, max_order=2, shape=(72, 72))
    cwtmatr = np.zeros((training_batch_size, 11664, 1))
    sig = np.array(sig)
    for i in range(0,training_batch_size):
        scatteringCoeffs = scattering(sig[i,:,:].reshape(72,72))
        cwtmatr[i,:,:] = scatteringCoeffs.flatten()[:,None]
    return cwtmatr


class LOCA:
    def __init__(self, q_layers, g_layers, weight_layers , m=100, P=100, X=None, Y=None, Yt=None):    
        # Network initialization and evaluation functions

        self.q_init, self.q_apply = self.init_NN(q_layers, activation=Gelu)
        self.in_shape = (-1, q_layers[0])
        self.out_shape, q_params = self.q_init(random.PRNGKey(10000), self.in_shape)

        self.v_init, self.v_apply = self.init_NN(v_layers, activation=Gelu)
        self.in_shape = (-1, v_layers[0])
        self.out_shape, v_params = self.v_init(random.PRNGKey(10000), self.in_shape)

        self.g_init, self.g_apply = self.init_NN(g_layers, activation=Gelu)
        self.in_shape = (-1, g_layers[0])
        self.out_shape, g_params = self.g_init(random.PRNGKey(10000), self.in_shape)


        self.R = 1000
        self.D = 100
        self.N = 432
        self.W    = random.normal(random.PRNGKey(10000), shape=(self.R, self.D))
        self.b    = random.uniform(random.PRNGKey(10000), minval=0, maxval=2*np.pi, shape=(self.R,))
        self.B    = jnp.repeat(self.b[:, jnp.newaxis], self.N, axis=1)
        self.norm = 1./ jnp.sqrt(self.R)

        beta = [1.]
        gamma = [1.]
        params = (beta, gamma, encoder_params2, g_params, weights_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_operator_log = []
        self.loss_physics_log = []

        self.vdistance_function = vmap(pairwise_distances(euclid_distance))
        self.distance_function = vmap(jit(self.euclid_distance))
        

    def init_NN(self, Q, activation=Gelu):
        layers = []
        num_layers = len(Q)
        if num_layers < 2:
            net_init, net_apply = stax.serial()
        else:
            for i in range(0, num_layers-2):
                layers.append(Dense(Q[i+1]))
                layers.append(activation)
            layers.append(Dense(Q[-1]))
            net_init, net_apply = stax.serial(*layers)
        return net_init, net_apply

    # @partial(jax.jit, static_argnums=0)
    def euclid_distance(self, x, y, square=True):
        diff = x[None,:,:] - y[:,None,:]
        return jnp.sum(diff**2,axis=-1)

    @partial(jit, static_argnums=(0,))
    def matr_vec(self,M,v):
        return vmap(jnp.matmul,in_axes=(None,0))(M,v)

    @partial(jit, static_argnums=(0,))
    def matr_matr(self,M,v):
        return vmap(jnp.matmul,in_axes=(0,0))(M,v)

    @partial(jax.jit, static_argnums=0)
    def fast_gauss_kernel(self, x):
        print(x.T.shape, self.W.shape, self.B.shape)
        Z   = self.norm * np.sqrt(2) * jnp.cos(jnp.matmul(self.W,x.T) + self.B)
        return jnp.matmul(Z.T,Z)

    @partial(jax.jit, static_argnums=0)
    def vector_fast_gauss_kernel(self,x):
        return vmap(self.fast_gauss_kernel,in_axes=(0))(x)

    @partial(jax.jit, static_argnums=0)
    def LOCA_net(self, params, inputs, ds=1):
        beta, gamma, q_params, g_params, v_params = params
        inputsxu, inputsy = inputs
        inputsy  = self.q_apply(q_params,inputsy)

        attn_logits   = self.vdistance_function(inputsy, inputsy)
        K =  beta[0]*jnp.exp(- gamma[0]*attn_logits)
        Kxx =  jnp.sqrt((1./K.shape[1])*jnp.sum(K ,axis=-1,keepdims=True))
        mean_K = jnp.matmul(Kxx, jnp.swapaxes(Kxx,1,2))
        K = jnp.divide(K,mean_K)
        
        g  = self.g_apply(g_params, inputsy)
        g = (1./K.shape[1])*jnp.einsum("ijk,ikml->ijml",K,g.reshape(inputsy.shape[0], inputsy.shape[1], ds, int(g.shape[2]/ds)))
        g = jax.nn.softmax(g, axis=-1)

        value_heads = self.v_apply(v_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        value_heads = value_heads.reshape(value_heads.shape[0],int(value_heads.shape[2]/ds),ds)
        Guy = jnp.einsum("ijkl,ilk->ijk", g,value_heads)
        return Guy

      
    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = jnp.mean((outputs.flatten() - y_pred.flatten())**2)
        return loss    


    @partial(jax.jit, static_argnums=0)
    def lossT(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = jnp.mean((outputs.flatten() - y_pred.flatten())**2)
        return loss    
    
    @partial(jax.jit, static_argnums=0)
    def L2errorT(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(outputs.flatten() - y_pred.flatten(), 2)/norm(outputs.flatten(),2)


    @partial(jax.jit, static_argnums=0)
    def L2error(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(outputs.flatten() - y_pred.flatten(), 2)/norm(outputs.flatten(),2)
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)

    def train(self, train_dataset, test_dataset, nIter = 10000):
        train_data = iter(train_dataset)
        test_data  = iter(test_dataset)

        pbar = trange(nIter)
        for it in pbar:
            train_batch = next(train_data)
            test_batch  = next(test_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, train_batch)
            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                loss_train = self.loss(params, train_batch)
                loss_test  = self.lossT(params, test_batch)

                errorTrain = self.L2error(params, train_batch)
                errorTest  = self.L2errorT(params, test_batch)

                self.loss_log.append(loss_train)

                pbar.set_postfix({'Training loss': loss_train, 
                                  'Testing loss' : loss_test,
                                  'Test error':    errorTest,
                                  'Train error':   errorTrain})

    @partial(jit, static_argnums=(0,))
    def predict(self, params, inputs):
        s_pred = self.LOCA_net(params,inputs)
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predictT(self, params, inputs):
        s_pred = self.LOCA_net(params,inputs)
        return s_pred

    def ravel_list(self, *lst):
        return jnp.concatenate([jnp.ravel(elt) for elt in lst]) if lst else jnp.array([])

    def ravel_pytree(self, pytree):
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        flat, unravel_list = vjp(self.ravel_list, *leaves)
        unravel_pytree = lambda flat: jax.tree_util.tree_unflatten(treedef, unravel_list(flat))
        return flat, unravel_pytree

    def count_params(self, params):
        beta, gamma,q_params, g_params, v_params = params
        qlv, _ = self.ravel_pytree(q_params)
        vlv, _ = self.ravel_pytree(v_params)
        glv, _ = self.ravel_pytree(g_params)
        print("The number of model parameters is:",qlv.shape[0]+vlv.shape[0]+glv.shape[0])

def predict_function(U_in,Y_in, model=None, params= None, H=10):
    y = np.expand_dims(Y_in,axis=0)
    y = np.tile(y,(U_in.shape[0],1,1))
    inputs_trainxu = jnp.asarray(U_in)
    pos_encodingy  = PositionalEncodingY(y,int(y.shape[1]*y.shape[2]), max_len = Y_in.shape[0], H=H)
    y  = pos_encodingy.forward(y)
    del pos_encodingy
    uCNN_super_all = model.predict(params, (inputs_trainxu, y))
    return uCNN_super_all, y[:,:,1:2], y[:,:,0:1]

def error_full_resolution(uCNN_super_all, s_all,tag='train', num_train=1000, P=128, Nx=32, Ny=32, idx=None):
    test_error_u = []
    z = uCNN_super_all.reshape(num_train,Nx,Ny)
    s = s_all.reshape(num_train,Nx,Ny)
    # s = np.swapaxes(s,1,2)
    for i in range(0,num_train):
        test_error_u.append(norm(s[i,:,0]- z[i,:,0], 2)/norm(s[i,:,0], 2))
    print("The average "+tag+" u error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
    absolute_error = np.abs(z-s)
    return absolute_error, np.mean(test_error_u), test_error_u


def minmax(a, n, mean):
    minpos = a.index(min(a))
    maxpos = a.index(max(a)) 
    meanpos = min(range(len(a)), key=lambda i: abs(a[i]-mean))

    print("The maximum is at position", maxpos)  
    print("The minimum is at position", minpos)
    print("The mean is at position", meanpos)
    return minpos,maxpos,meanpos


TRAINING_ITERATIONS = 100000
P = 144
m = int(72*72)
T = 1
N_hat = 1
num_train = 1825
num_test  = 1825
training_batch_size = 73
dx = 2
du = 1
dy = 2
ds = 1
n_hat  = 100
l  = 100
Nx = 72
Ny = 72
Nt = 1
Ng = 0
L = 1
H = 10
casenum_train = 2
casenum_test  = 2

d = np.load("../Data/weather_dataset.npz")
u_train = d["U_train"][:num_train,:]
S_train = d["S_train"][:num_train,:]/1000.
x_train = d["X_train"]
Y_train = d["Y_train"]

d = np.load("../Data/weather_dataset.npz")
u_test = d["U_train"][-num_test:,:]
S_test = d["S_train"][-num_test:,:]/1000.
x_test = d["X_train"]
Y_test = d["Y_train"]

Y_train_in = Y_train
Y_test_in = Y_test

s_all_test = S_test
s_all_train = S_train

s_train = np.zeros((num_train*N_hat,P,ds))
y_train = np.zeros((num_train*N_hat,P,dy))
U_train = np.zeros((num_train*N_hat,m,du))
X_train = np.zeros((num_train*N_hat,m,dx))

s_test = np.zeros((num_test,P,ds))
y_test = np.zeros((num_test,P,dy))
U_test = np.zeros((num_test,m,du))
X_test = np.zeros((num_test,m,dx))

for j in range(0,N_hat):
    for i in range(0,num_train):
        s_train[i + j*num_train,:,:], y_train[i+ j*num_train,:,:] = output_construction(S_train[i,:], Y_train, Nx=Nx, Ny=Ny, P=P, ds=ds)
        U_train[i+ j*num_train,:,:],  X_train[i+ j*num_train,:,:] = input_construction( u_train[i,:], x_train, Nx=Nx, Ny=Ny, m=m, du=du)

for i in range(num_test):
    s_test[i,:,:], y_test[i,:,:] = output_construction(S_test[i,:], Y_test, Nx=Nx, Ny=Ny, P=P, ds=ds)
    U_test[i,:,:], X_test[i,:,:] = input_construction( u_test[i,:], x_test, Nx=Nx, Ny=Ny, m=m, du=du)    

num_train = N_hat*num_train

X_train2 = X_train
U_train2 = U_train

X_test2 = X_test
U_test2 = U_test

X_train = jnp.asarray(X_train)
U_train = np.asarray(U_train)
y_train = jnp.asarray(y_train)
s_train = jnp.asarray(s_train)

X_test = jnp.asarray(X_test)
U_test = np.asarray(U_test)
y_test = jnp.asarray(y_test)
s_test = jnp.asarray(s_test)

X_train = jnp.reshape(X_train,(num_train,m,dx))
U_train = np.reshape(U_train,(num_train,m,du))
y_train = jnp.reshape(y_train,(num_train,P,dy))
s_train = jnp.reshape(s_train,(num_train,P,ds))

X_test = jnp.reshape(X_test,(num_test,m,dx))
U_test = np.reshape(U_test,(num_test,m,du))
y_test = jnp.reshape(y_test,(num_test,P,dy))
s_test = jnp.reshape(s_test,(num_test,P,ds))

y_train_pos = y_train
y_train_posT = y_test

pos_encodingy  = PositionalEncodingY(y_train_pos,int(y_train_pos.shape[1]*y_train_pos.shape[2]), max_len = P, H=H)
y_train  = pos_encodingy.forward(y_train)
del pos_encodingy

pos_encodingyt = PositionalEncodingY(y_train_posT,int(y_train_posT.shape[1]*y_train_posT.shape[2]), max_len = P, H=H)
y_test   = pos_encodingyt.forward(y_test)
del pos_encodingyt

inputs_trainxu = jnp.asarray(continouswavetransf(U_train, l=l, m=m, training_batch_size=num_train))
inputs_testxu  = jnp.asarray(continouswavetransf(U_test , l=l, m=m, training_batch_size=num_test))

train_dataset = DataGenerator(inputs_trainxu, y_train, s_train, training_batch_size)
train_dataset = iter(train_dataset)

test_dataset = DataGenerator(inputs_testxu, y_test, s_test, training_batch_size)
test_dataset = iter(test_dataset)

q_layers = [L*dy+H*dy, 100, 100, l]
weights_layers = [11664, 100, 100, ds*n_hat]
g_layers  = [l, 100, 100, n_hat]

model = LOCA(q_layers, g_layers, weights_layers, m=m, P=P, X=X_train, Y=y_train_pos, Yt=y_train_posT)

model.count_params(model.get_params(model.opt_state))

start_time = timeit.default_timer()
model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

params = model.get_params(model.opt_state)

print("Predicting the solution for the full resolution")
uCNN_super_all_test = np.zeros_like(s_all_test).reshape(num_test, Nx*Ny, ds)
for i in range(0, Nx*Ny, P):
    idx = i + np.arange(0,P)
    uCNN_super_all_test[:,idx,:], _, _  = predict_function(inputs_testxu , Y_test_in[idx,:], model=model, params=params, H=H)

uCNN_super_all_train = np.zeros_like(s_all_train).reshape(num_train, Nx*Ny, ds)
for i in range(0, Nx*Ny, P):
    idx = i + np.arange(0,P)
    uCNN_super_all_train[:,idx,:], _, _  = predict_function(inputs_trainxu , Y_train_in[idx,:], model=model, params=params, H=H)


absolute_error_test, mean_test_error, test_error     = error_full_resolution(uCNN_super_all_test,s_all_test,  tag='test', P=P,Nx=Nx, Ny=Ny, idx = None, num_train=num_test)
absolute_error_train, mean_train_error, train_error  = error_full_resolution(uCNN_super_all_train,s_all_train,tag='train',P=P,Nx=Nx, Ny=Ny, idx = None, num_train=num_train)