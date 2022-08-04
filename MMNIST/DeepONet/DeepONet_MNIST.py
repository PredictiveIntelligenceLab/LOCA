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
import jax.numpy as jnp
from jax import vjp
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
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmin(memory_available))

os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"


def output_construction(s,T, X, Y,P=100,ds=1, dy=2, N=1000,Nx=100,Ny=100, Nt=2):
    U_all = np.zeros((P,ds))
    Y_all = np.zeros((P,ds))
    t = np.random.randint(Nt, size=P)
    x = np.random.randint(Nx, size=P)
    y = np.random.randint(Ny, size=P)
    Y_all = np.concatenate((T[t,x][range(P),y][:,None], X[t,x][range(P),y][:,None], Y[t,x][range(P),y][:,None]),axis=-1)
    U_all[:,:] = s[t,x][range(P), y]
    return U_all, Y_all


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

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
        u  = self.u[idx,:,:]
        y = self.y[idx,:,:]
        inputs = (u, y)
        return inputs, s

class PositionalEncodingY: 
    def __init__(self, Y, d_model, max_len = 100, H=4): 
        self.d_model = int(np.ceil(d_model/6)*2)
        self.Y = Y 
        self.max_len = max_len 
        self.H = H
 
    def forward(self, x):
        pet = np.zeros((x.shape[0], self.max_len, self.H))
        pex = np.zeros((x.shape[0], self.max_len, self.H))
        pey = np.zeros((x.shape[0], self.max_len, self.H))
        T = jnp.take(self.Y, 0, axis=2)[:,:,None]
        X = jnp.take(self.Y, 1, axis=2)[:,:,None]
        Y = jnp.take(self.Y, 2, axis=2)[:,:,None]
        positionT = jnp.tile(T,(1,1,self.H))
        positionX = jnp.tile(X,(1,1,self.H))
        positionY = jnp.tile(Y,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        pet = jax.ops.index_update(pet, jax.ops.index[:,:,0::2], jnp.cos(positionT[:,:,0::2] * div_term))
        pet = jax.ops.index_update(pet, jax.ops.index[:,:,1::2], jnp.sin(positionT[:,:,1::2] * div_term))
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,0::2], jnp.cos(positionX[:,:,0::2] * div_term))
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,1::2], jnp.sin(positionX[:,:,1::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,0::2], jnp.cos(positionY[:,:,0::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,1::2], jnp.sin(positionY[:,:,1::2] * div_term))
        pos_embedding =  jnp.concatenate((pet,pex,pey),axis=-1)
        x =  jnp.concatenate([x, pos_embedding], -1)
        return x


class PositionalEncodingU:
    def __init__(self, Y, d_model, max_len = 100, H=20):
        self.d_model = int(np.ceil(d_model/4)*2)
        self.Y = Y
        self.max_len = max_len
        self.H = H

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

class DON:
    def __init__(self,branch_layers, trunk_layers , m=100, P=100, mn=None, std=None):    
        # Network initialization and evaluation functions
        seed = np.random.randint(10000)
        self.branch_init, self.branch_apply = self.init_NN(branch_layers, activation=Gelu)
        self.in_shape = (-1, branch_layers[0])
        self.out_shape, branch_params = self.branch_init(random.PRNGKey(seed), self.in_shape)

        seed = np.random.randint(10000)
        self.trunk_init, self.trunk_apply = self.init_NN(trunk_layers, activation=Gelu)
        self.in_shape = (-1, trunk_layers[0])
        self.out_shape, trunk_params = self.trunk_init(random.PRNGKey(seed), self.in_shape)

        params = (trunk_params, branch_params)

        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.loss_log = []
        self.mean = mn
        self.std = std

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

    @partial(jax.jit, static_argnums=0)
    def DON(self, params, inputs, ds=2):
        trunk_params, branch_params = params
        inputsxu, inputsy = inputs
        t = self.trunk_apply(trunk_params, inputsy).reshape(inputsy.shape[0], inputsy.shape[1], ds, int(1000/ds))
        b = self.branch_apply(branch_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        b = b.reshape(b.shape[0],int(b.shape[2]/ds),ds)
        Guy = jnp.einsum("ijkl,ilk->ijk", t,b)
        return Guy
      
    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, y = batch
        y_pred = self.DON(params,inputs)
        y = y*self.std + self.mean
        y_pred = y_pred*self.std + self.mean
        loss = np.mean((y.flatten() - y_pred.flatten())**2)
        return loss    

    @partial(jax.jit, static_argnums=0)
    def lossT(self, params, batch):
        inputs, outputs = batch
        y_pred = self.DON(params,inputs)
        y_pred = y_pred*self.std + self.mean
        loss = np.mean((outputs.flatten() - y_pred.flatten())**2)
        return loss    
    
    @partial(jax.jit, static_argnums=0)
    def L2errorT(self, params, batch):
        inputs, y = batch
        y_pred = self.DON(params,inputs)
        y_pred = y_pred*self.std + self.mean
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)

    @partial(jax.jit, static_argnums=0)
    def L2error(self, params, batch):
        inputs, y = batch
        y_pred = self.DON(params,inputs)
        y = y*self.std + self.mean
        y_pred = y_pred*self.std + self.mean
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)

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
        s_pred = self.DON(params,inputs)
        return s_pred*self.std + self.mean

    def ravel_list(self, *lst):
        return jnp.concatenate([jnp.ravel(elt) for elt in lst]) if lst else jnp.array([])

    def ravel_pytree(self, pytree):
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        flat, unravel_list = vjp(self.ravel_list, *leaves)
        unravel_pytree = lambda flat: jax.tree_util.tree_unflatten(treedef, unravel_list(flat))
        return flat, unravel_pytree

    def count_params(self, params):
        trunk_params, branch_params = params
        blv, _ = self.ravel_pytree(branch_params)
        tlv, _ = self.ravel_pytree(trunk_params)
        print("The number of model parameters is:",blv.shape[0]+tlv.shape[0])

def predict_function(U_in,Y_in,num_test=1000, model=None,params= None, H=4):
    y = np.expand_dims(Y_in,axis=0)
    y = np.tile(y,(num_test,1,1))
    inputs_trainxu = jnp.asarray(U_in)
    pos_encodingy  = PositionalEncodingY(y,int(y.shape[1]*y.shape[2]), max_len = Y_in.shape[0], H=H)
    y  = pos_encodingy.forward(y)
    del pos_encodingy
    uCNN_super_all = model.predict(params, (inputs_trainxu, y))
    return uCNN_super_all, y[:,:,1:2], y[:,:,0:1]

def error_full_resolution(uCNN_super_all, s_all,tag='train', num_train=1000,P=128, Nx=30, Ny=30, Nt=10, idx=None, ds=2):
    z = uCNN_super_all.reshape(num_train,Nt*Nx*Ny,ds)
    s = s_all.reshape(num_train,Nt*Nx*Ny,ds)
    test_error_u = []
    for i in range(0,num_train):
        test_error_u.append(norm(s[i,:,0]- z[i,:,0], 2)/norm(s[i,:,0], 2))
    print("The average "+tag+" u error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
    test_error_v = []
    for i in range(0,num_train):
        test_error_v.append(norm(s[i,:,1]- z[i,:,1], 2)/norm(s[i,:,1], 2))
    print("The average "+tag+" v error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_v),np.std(test_error_v),np.min(test_error_v),np.max(test_error_v)))
    absolute_error = np.abs(z-s)
    return absolute_error, np.mean(test_error_u),np.mean(test_error_v), test_error_u, test_error_v

def minmax(a, n, mean):
    minpos = a.index(min(a))
    maxpos = a.index(max(a)) 
    meanpos = min(range(len(a)), key=lambda i: abs(a[i]-mean))

    print("The maximum is at position", maxpos)  
    print("The minimum is at position", minpos)
    print("The mean is at position", meanpos)
    return minpos,maxpos,meanpos

TRAINING_ITERATIONS = 100000
P = 56
m = int(28*28)
N_hat = 1
num_train = 60000
num_test  = 10000
training_batch_size = 500
dx = 2
du = 2
dy = 3
ds = 2
n_hat  = 500
Nx = 28
Ny = 28
H_y = 10
H_u = 10
ind = 0

idxT = [11]
Nt = len(idxT)
d = np.load("../Data/MMNIST_dataset_train.npz")
dispx_allsteps_train = d["dispx_allsteps_train"][:num_train,idxT,:,:,None]
dispy_allsteps_train = d["dispy_allsteps_train"][:num_train,idxT,:,:,None]
u_trainx = d["dispx_allsteps_train"][:num_train,7,:,:,None]
u_trainy = d["dispy_allsteps_train"][:num_train,7,:,:,None]
bitmap_train   = d["MNIST_inputs_train"][:,:,:,None]/255.

d = np.load("../Data/MMNIST_dataset_test.npz")
dispx_allsteps_test = d["dispx_allsteps_test"][:num_test,idxT,:,:,None]
dispy_allsteps_test = d["dispy_allsteps_test"][:num_test,idxT,:,:,None]
u_testx = d["dispx_allsteps_test"][:num_test,7,:,:,None]
u_testy = d["dispy_allsteps_test"][:num_test,7,:,:,None]
bitmap_test   = d["MNIST_inputs_test"][:,:,:,None]/255.

S_train = np.concatenate((dispx_allsteps_train,dispy_allsteps_train),axis=-1)
S_test  = np.concatenate((dispx_allsteps_test,dispy_allsteps_test),axis=-1)

u_train  = np.concatenate((u_trainx,u_trainy),axis=-1)
u_test   = np.concatenate((u_testx,u_testy),axis=-1)

X = np.zeros((Nt,Nx,Ny))
Y = np.zeros((Nt,Nx,Ny))
T = np.zeros((Nt,Nx,Ny))

dx = 0.037037037037037035

for ii in range(0,Nt):
    T[ii,:,:] = ii
    for kk in range(0,Nx):
            for jj in range(0,Ny):
                    X[ii, kk,jj] = jj*dx# 0.5 # x is columns
                    Y[ii, kk,jj] = kk*dx# 0.5 # y is rows 

Y_train = np.concatenate((T.flatten()[:,None], X.flatten()[:,None], Y.flatten()[:,None]),axis=-1)
Y_test  = np.concatenate((T.flatten()[:,None], X.flatten()[:,None], Y.flatten()[:,None]),axis=-1)
Y_train_in = Y_train
Y_test_in = Y_test


CX = np.linspace(0,1,num=Nx)
CY = np.linspace(0,1,num=Ny)

s_all_test = S_test
s_all_train = S_train

# num_train = num_train*N_hat
s_train = np.zeros((num_train*N_hat,P,ds))
y_train = np.zeros((num_train*N_hat,P,dy))
U_train = np.zeros((num_train*N_hat,m,du))

s_test = np.zeros((num_test,P,ds))
y_test = np.zeros((num_test,P,dy))
U_test = np.zeros((num_test,m,du))

for j in range(0,N_hat):
    for i in range(0,num_train):
        s_train[i + j*num_train,:,:], y_train[i+ j*num_train,:,:] = output_construction(S_train[i,:,:,:,:], T, X, Y, P=P,Nt=Nt, Nx=Nx, Ny=Ny, ds=ds, dy=dy)
        U_train[i+ j*num_train,:,:] = u_train[i,:,:,:].reshape(Nx*Ny,du)

for i in range(num_test):
    s_test[i,:,:], y_test[i,:,:] = output_construction(S_test[i,:,:,:,:], T, X,Y, P=P,Nt=Nt, Nx=Nx, Ny=Ny, ds=ds, dy=dy)
    U_test[i,:,:] = u_test[i,:,:,:].reshape(Nx*Ny,du)

del S_train, S_test, dispx_allsteps_train, dispy_allsteps_train, dispx_allsteps_test, dispy_allsteps_test, u_train

U_train = jnp.asarray(U_train)
y_train = jnp.asarray(y_train)
s_train = jnp.asarray(s_train)

U_test = jnp.asarray(U_test)
y_test = jnp.asarray(y_test)
s_test = jnp.asarray(s_test)

U_train = jnp.reshape(U_train,(num_train,m,du))
y_train = jnp.reshape(y_train,(num_train,P,dy))
s_train = jnp.reshape(s_train,(num_train,P,ds))

U_test = jnp.reshape(U_test,(num_test,m,du))
y_test = jnp.reshape(y_test,(num_test,P,dy))
s_test = jnp.reshape(s_test,(num_test,P,ds))

pos_encodingy  = PositionalEncodingY(y_train,int(y_train.shape[1]*y_train.shape[2]), max_len = P, H=H_y)
y_train  = pos_encodingy.forward(y_train)
del pos_encodingy

pos_encodingyt = PositionalEncodingY(y_test,int(y_test.shape[1]*y_test.shape[2]), max_len = P, H=H_y)
y_test   = pos_encodingyt.forward(y_test)
del pos_encodingyt

pos_encodingy  = PositionalEncodingU(U_train,int(U_train.shape[1]*U_train.shape[2]), max_len = m, H=H_u)
U_train  = pos_encodingy.forward(U_train)
del pos_encodingy

pos_encodingyt = PositionalEncodingU(U_test,int(U_test.shape[1]*U_test.shape[2]), max_len = m, H=H_u)
U_test   = pos_encodingyt.forward(U_test)
del pos_encodingyt

s_train_mean = jnp.mean(s_train,axis=0)
s_train_std  = jnp.std(s_train,axis=0)

s_train = (s_train - s_train_mean)/s_train_std

train_dataset = DataGenerator(U_train, y_train, s_train, training_batch_size)
train_dataset = iter(train_dataset)

test_dataset = DataGenerator(U_test, y_test, s_test, training_batch_size)
test_dataset = iter(test_dataset)

print(U_train.shape, U_test.shape, y_train.shape, y_test.shape, s_train.shape, s_test.shape)
branch_layers = [m*(du*H_u+du),100, 100, 100, 100, ds*n_hat]
trunk_layers  = [H_y*dy + dy,  100, 100, 100, 100, ds*n_hat]

model = DON(branch_layers, trunk_layers, m=m, P=P, mn=s_train_mean,  std=s_train_std)

model.count_params(model.get_params(model.opt_state))
del U_train, y_train, s_train

start_time = timeit.default_timer()
model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)


params = model.get_params(model.opt_state)
tag = "NN"
# in_noise_test  = 0.15*np.random.normal(loc=0.0, scale=1.0, size=(u_test.shape))
# u_test = u_test + in_noise_test

U_test = np.zeros((num_test,m,du))

for i in range(num_test):
    U_test[i,:,:] = u_test[i,:,:,:].reshape(Nx*Ny,du)

pos_encodingyt = PositionalEncodingU(U_test,int(U_test.shape[1]*U_test.shape[2]), max_len = m, H=H_u)
U_test   = pos_encodingyt.forward(U_test)
del pos_encodingyt

rint("Predicting the solution for the full resolution")
uCNN_super_all_test = np.zeros_like(s_all_test).reshape(num_test, Nx*Ny*Nt, ds)

if P>300:
    PP = int(P/4)
else:
    PP = P

for i in range(0, Nx*Ny, PP):
    idx = i + np.arange(0,PP)
    uCNN_super_all_test[:,idx,:], _, _  = predict_function(U_test , Y_test_in[idx,:], model=model, params=params, num_test=num_test, H=H_y)

absolute_error_test, mean_test_error_u, mean_test_error_v, test_error_u, test_error_v  = error_full_resolution(uCNN_super_all_test,s_all_test,tag='test',P=P,Nx=Nx, Ny=Ny, Nt=Nt, idx = None, num_train=num_test)
