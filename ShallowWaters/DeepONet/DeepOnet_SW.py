from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathos.pools import ProcessPool
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
import argparse
from jax.experimental.stax import Dense, Gelu, Relu
from jax.experimental import stax
import os

import timeit

from jax.experimental import optimizers

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

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmin(memory_available))

os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()

def output_construction(Ux,t_his,cx, cy, ng,P=1000, num_train=1000, ds=3, Nx=30, Ny=30, Nt=100):
    U_all = np.zeros((P,ds))
    Y_all = np.zeros((P,ds))
    it = np.random.randint(Nt, size=P)
    x  = np.random.randint(Nx, size=P)
    y  = np.random.randint(Ny, size=P)
    T, X, Y = np.meshgrid(t_his,cx,cy,indexing="ij")
    Y_all[:,:] = np.concatenate((T[it,x][range(P),y][:,None], X[it,x][range(P),y][:,None], Y[it,x][range(P),y][:,None]),axis=-1)
    U_all[:,:] = Ux[it,x][range(P),y]
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

class DON:
    def __init__(self,branch_layers, trunk_layers , m=100, P=100, mn=None, std=None):    
        # Network initialization and evaluation functions

        self.branch_init, self.branch_apply = self.init_NN(branch_layers, activation=Gelu)
        self.in_shape = (-1, branch_layers[0])
        self.out_shape, branch_params = self.branch_init(random.PRNGKey(10000), self.in_shape)

        self.trunk_init, self.trunk_apply = self.init_NN(trunk_layers, activation=Gelu)
        self.in_shape = (-1, trunk_layers[0])
        self.out_shape, trunk_params = self.trunk_init(random.PRNGKey(10000), self.in_shape)

        params = (trunk_params, branch_params)
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
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
    def DON(self, params, inputs, ds=3):
        trunk_params, branch_params = params
        inputsxu, inputsy = inputs
        t = self.trunk_apply(trunk_params, inputsy).reshape(inputsy.shape[0], inputsy.shape[1], ds, int(12/ds))
        b = self.branch_apply(branch_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        b = b.reshape(b.shape[0],int(b.shape[2]/ds),ds)
        Guy = jnp.einsum("ijkl,ilk->ijk", t,b)
        return Guy
      
    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, outputs = batch
        y_pred = self.DON(params,inputs)
        outputs = outputs*self.std + self.mean
        y_pred = y_pred*self.std + self.mean
        loss = np.mean((outputs[:,:,0].flatten() - y_pred[:,:,0].flatten())**2) + 100.*np.mean((outputs[:,:,1].flatten() - y_pred[:,:,1].flatten())**2) + 100.*np.mean((outputs[:,:,2].flatten() - y_pred[:,:,2].flatten())**2)
        return loss    

    @partial(jax.jit, static_argnums=0)
    def lossT(self, params, batch):
        inputs, outputs = batch
        y_pred = self.DON(params,inputs)
        y_pred = y_pred*self.std + self.mean
        loss = np.mean((outputs[:,:,0].flatten() - y_pred[:,:,0].flatten())**2) + 100.*np.mean((outputs[:,:,1].flatten() - y_pred[:,:,1].flatten())**2) + 100.*np.mean((outputs[:,:,2].flatten() - y_pred[:,:,2].flatten())**2)
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

    @partial(jit, static_argnums=(0,))
    def predictT(self, params, inputs):
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


def predict_function(U_in,y, model=None,params= None, H=None):
    inputs_trainxu = jnp.asarray(U_in)
    uCNN_super_all = model.predict(params, (inputs_trainxu, y))
    return uCNN_super_all, y[:,:,1:2], y[:,:,0:1]

def error_full_resolution(uCNN_super_all, s_all,tag='train', num_train=1000,P=128, Nx=30, Ny=30, Nt=10, idx=None, ds=3):
    print(s_all.shape)
    z = uCNN_super_all.reshape(num_train,Nx*Ny*Nt,ds)
    s = s_all.reshape(num_train,Nx*Ny*Nt,ds)
    test_error_rho = []
    for i in range(0,num_train):
        test_error_rho.append(norm(s[i,:,0]- z[i,:,0], 2)/norm(s[i,:,0], 2))
    print("The average "+tag+" rho error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_rho),np.std(test_error_rho),np.min(test_error_rho),np.max(test_error_rho)))

    test_error_u = []
    for i in range(0,num_train):
        test_error_u.append(norm(s[i,:,1]- z[i,:,1], 2)/norm(s[i,:,1], 2))
    print("The average "+tag+" u error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))

    test_error_v = []
    for i in range(0,num_train):
        test_error_v.append(norm(s[i,:,2]- z[i,:,2], 2)/norm(s[i,:,2], 2))
    print("The average "+tag+" v error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_v),np.std(test_error_v),np.min(test_error_v),np.max(test_error_v)))

    absolute_error = np.abs(z-s)
    return absolute_error, np.mean(test_error_rho), np.mean(test_error_u),np.mean(test_error_v), (test_error_rho, test_error_u, test_error_v) 


# if __name__ == "__main__":
TRAINING_ITERATIONS = 100000
P = 128
m = 1024
num_train = 1000
num_test  = 1000
training_batch_size = 100
dx = 3
du = 3
dy = 3
ds = 3
n_hat  = 4
l  = 100
Nx = 32
Ny = 32
Nt = 5
Ng = 0
H_y = 2
H_u = 2

idxT = [10,15,20,25,30]
d = np.load("/scratch/gkissas/all_train_SW_Nx%d_Ny%d_numtrain%d.npz"%(Nx,Ny,1000))
u_train = d["U_train"][:,:,:,:]
S_train = d["s_train"][:,idxT,:,:,:]
T  = d["T_train"][idxT]
CX = d["X_train"]
CY = d["Y_train"]

d = np.load("/scratch/gkissas/all_test_SW_Nx%d_Ny%d_numtest%d.npz"%(Nx,Ny,1000))
u_test = d["U_test"][:,:,:,:]
S_test = d["s_test"][:,idxT,:,:,:]
T  = d["T_test"][idxT]
CX = d["X_test"]
CY = d["Y_test"]


s_all_test = S_test
s_all_train = S_train

s_train = np.zeros((num_train,P,ds))
y_train = np.zeros((num_train,P,dy))
U_train = np.zeros((num_train,m,du))
X_train = np.zeros((num_train,m,dx))

s_test = np.zeros((num_test,P,ds))
y_test = np.zeros((num_test,P,dy))
U_test = np.zeros((num_test,m,du))
X_test = np.zeros((num_test,m,dx))

for i in range(0,num_train):
    s_train[i ,:,:], y_train[i,:,:] = output_construction(S_train[i,:,:,:,:], T, CX, CY, Ng,P=P,Nt=Nt)
    U_train[i,:,:] = u_train[i,:,:,:].reshape(Nx*Ny,du)

for i in range(num_test):
    s_test[i,:,:],  y_test[i,:,:]  = output_construction(S_test[i,:,:,:,:], T, CX, CY, Ng,P=P,Nt=Nt)
    U_test[i,:,:] = u_test[i,:,:,:].reshape(Nx*Ny,du)

U_train = jnp.asarray(U_train)
y_train = jnp.asarray(y_train)
s_train = jnp.asarray(s_train)

U_test = jnp.asarray(U_test)
y_test = jnp.asarray(y_test)
s_test = jnp.asarray(s_test)

X_train = jnp.reshape(X_train,(num_train,m,dx))
U_train = jnp.reshape(U_train,(num_train,m,du))
y_train = jnp.reshape(y_train,(num_train,P,dy))
s_train = jnp.reshape(s_train,(num_train,P,ds))

X_test = jnp.reshape(X_test,(num_test,m,dx))
U_test = jnp.reshape(U_test,(num_test,m,du))
y_test = jnp.reshape(y_test,(num_test,P,dy))
s_test = jnp.reshape(s_test,(num_test,P,ds))

s_train_mean = jnp.mean(s_train,axis=0)
s_train_std  = jnp.std(s_train,axis=0)

s_train = (s_train - s_train_mean)/s_train_std

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

train_dataset = DataGenerator(U_train, y_train, s_train, training_batch_size)
train_dataset = iter(train_dataset)

test_dataset = DataGenerator(U_test, y_test, s_test, training_batch_size)
test_dataset = iter(test_dataset)

branch_layers = [m*(du*H_u+du), 100, 100, 100, ds*n_hat]
trunk_layers  = [H_y*dy + dy  , 100, 100, 100, ds*n_hat]

model = DON(branch_layers, trunk_layers, m=m, P=P, mn=s_train_mean,  std=s_train_std)

model.count_params(model.get_params(model.opt_state))

start_time = timeit.default_timer()
model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

params = model.get_params(model.opt_state)

uCNN_test = model.predict(params, (U_test, y_test))
test_error_u = []
for i in range(0,num_train):
    test_error_u.append(norm(s_test[i,:,0]- uCNN_test[i,:,0],2)/norm(s_test[i,:,0],2))
print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))

uCNN_train = model.predict(params, (U_train, y_train))
train_error_u = []
for i in range(0,num_test):
    train_error_u.append(norm(s_train[i,:,0]- uCNN_train[i,:,0],2)/norm(s_train[i,:,0],2))
print("The average train u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_u),np.std(train_error_u),np.min(train_error_u),np.max(train_error_u)))

TT, XX, YY = np.meshgrid(T, CX, CY, indexing="ij")

TT = np.expand_dims(TT,axis=0)
XX = np.expand_dims(XX,axis=0)
YY = np.expand_dims(YY,axis=0)

TT = np.tile(TT,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)
XX = np.tile(XX,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)
YY = np.tile(YY,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)

Y_test_in = np.concatenate((TT, XX, YY),axis=-1)
Y_train_in = np.concatenate((TT, XX, YY),axis=-1)

pos_encodingy  = PositionalEncodingY(Y_train_in,int(Y_train_in.shape[1]*Y_train_in.shape[2]), max_len = Y_train_in.shape[1], H=H_y)
Y_train_in  = pos_encodingy.forward(Y_train_in)
del pos_encodingy

pos_encodingy  = PositionalEncodingY(Y_test_in,int(Y_test_in.shape[1]*Y_test_in.shape[2]), max_len = Y_test_in.shape[1], H=H_y)
Y_test_in  = pos_encodingy.forward(Y_test_in)
del pos_encodingy

print("Predicting the solution for the full resolution")
uCNN_super_all_test = np.zeros_like(s_all_test).reshape(num_test, Nx*Ny*Nt, ds)
for i in range(0, Nx*Ny*Nt, P):
    idx = i + np.arange(0,P)
    uCNN_super_all_test[:,idx,:], _, _  = predict_function(U_test , Y_test_in[:,idx,:], model=model, params=params, H=H_y)

uCNN_super_all_train = np.zeros_like(s_all_train).reshape(num_train, Nx*Ny*Nt, ds)
for i in range(0, Nx*Ny*Nt, P):
    idx = i + np.arange(0,P)
    uCNN_super_all_train[:,idx,:], _, _  = predict_function(U_train , Y_train_in[:,idx,:], model=model, params=params, H=H_y)

absolute_error_train, mean_train_error_rho, mean_train_error_u, mean_train_error_v, train_error  = error_full_resolution(uCNN_super_all_train,s_all_train,tag='train',P=P,Nx=Nx, Ny=Ny, Nt=Nt, idx = None, num_train=num_train)
absolute_error_test, mean_test_error_rho, mean_test_error_u, mean_test_error_v, test_error  = error_full_resolution(uCNN_super_all_test,s_all_test,tag='test',P=P,Nx=Nx, Ny=Ny, Nt=Nt, idx = None, num_train=num_test)
