from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax.flatten_util import ravel_pytree
from jax.experimental.stax import Dense, Gelu
from jax.experimental import stax
import os

import timeit

from jax.experimental import optimizers

import jax
import jax.numpy as jnp
from jax import vjp
import numpy as np
from jax.numpy.linalg import norm

from jax import random, grad, jit
from functools import partial 

from torch.utils import data

from scipy import interpolate

from tqdm import trange
from math import sqrt

import itertools

from kymatio.numpy import Scattering2D

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmin(memory_available))

os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"


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
    U_all = U_all.reshape(int(sqrt(m))*int(sqrt(m)),du)
    return U_all, X_all

def output_construction(s,Y,P=100,ds=1, dy=2, N=1000,Nx=100,Ny=100):
    s = s.reshape(Nx,Ny)
    x = np.random.randint(Nx, size=P)
    y = np.random.randint(Ny, size=P)
    Y_all = np.hstack([x[:, None], y[:,None]]) * [1./(Nx - 1), 1./(Ny - 1)]
    s_all = s[x][range(P), y][:, None]
    return s_all, Y_all

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

class PositionalEncodingU:
    def __init__(self, U, d_model, max_len = 100, H=20):
        self.d_model = int(np.ceil(d_model/2)*2)
        self.U = U
        self.max_len = max_len
        self.H = H

    @partial(jit, static_argnums=(0,))
    def forward(self, x):
        peu = np.zeros((x.shape[0], self.max_len, self.H))
        U = jnp.take(self.U, 0, axis=2)[:,:,None]
        positionU = jnp.tile(U,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        peu = jax.ops.index_update(peu, jax.ops.index[:,:,0::2], jnp.cos(positionU[:,:,0::2] * div_term))
        peu = jax.ops.index_update(peu, jax.ops.index[:,:,1::2], jnp.sin(positionU[:,:,1::2] * div_term))
        x =  jnp.concatenate([x, peu], -1)
        return x

def scatteringTransform(sig, m=100, training_batch_size = 100):
    scattering = Scattering2D(J=1, L=3, max_order=2, shape=(32, 32))
    cwtmatr = np.zeros((training_batch_size, 768, 1))
    sig = np.array(sig)
    for i in range(0,training_batch_size):
        scatteringCoeffs = scattering(sig[i,:,:].reshape(32,32))
        cwtmatr[i,:,:] = scatteringCoeffs[:3,:,:].flatten()[:,None]
    return cwtmatr

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
    def DON(self, params, inputs, ds=1):
        trunk_params, branch_params = params
        inputsxu, inputsy = inputs
        t = self.trunk_apply(trunk_params, inputsy).reshape(inputsy.shape[0], inputsy.shape[1], ds, int(100/ds))
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

def predict_function(U_in,Y_in, model=None, params= None, H=10):
    y = np.expand_dims(Y_in,axis=0)
    y = np.tile(y,(U_in.shape[0],1,1))
    inputs_trainxu = jnp.asarray(U_in)

    pos_encodingy  = PositionalEncodingY(y,int(y.shape[1]*y.shape[2]), max_len = Y_in.shape[0], H=H)
    y  = pos_encodingy.forward(y)
    del pos_encodingy

    uCNN_super_all = model.predict(params, (inputs_trainxu, y))
    return uCNN_super_all, y[:,:,1:2], y[:,:,0:1]

def error_full_resolution(uCNN_super_all, s_all,tag='train', num_train=1000, Nx=32, Ny=32):
    test_error_u = []
    z = uCNN_super_all.reshape(num_train,Nx,Ny)
    s = s_all.reshape(num_train,Nx,Ny)
    s = np.swapaxes(s,1,2)
    for i in range(0,num_train):
        test_error_u.append(norm(s[i,:,:]- z[i,:,:], 2)/norm(s[i,:,:], 2))
    print("The average "+tag+" u error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
    absolute_error = np.abs(z-s)
    return absolute_error, np.mean(test_error_u), test_error_u


TRAINING_ITERATIONS = 20000
P = 128
m = 1024
num_train = 1000
num_test  = 1000
training_batch_size = 100
dx = 2
du = 1
dy = 2
ds = 1
n_hat  = 100
Nx = 32
Ny = 32
H_y = 6
H_u = 6

d = np.load("/scratch/gkissas/Darcy/train_darcy_dataset.npz")
u_train   = d["U_train"]
x_train   = d["X_train"]
Y_train   = d["Y_train"]
S_train   = d["s_train"]

d = np.load("/scratch/gkissas/Darcy/test_darcy_dataset.npz")
u_test   = d["U_test"]
x_test   = d["X_test"]
Y_test   = d["Y_test"]
S_test   = d["s_test"]

Y_train_in = Y_train
Y_test_in = Y_test

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
    s_train[i,:,:], y_train[i,:,:] = output_construction(S_train[i,:], Y_train, Nx=Nx, Ny=Ny, P=P, ds=ds)
    U_train[i,:,:],  X_train[i,:,:] = input_construction( u_train[i,:], x_train, Nx=Nx, Ny=Ny, m=m, du=du)

for i in range(num_test):
    s_test[i,:,:], y_test[i,:,:] = output_construction(S_test[i,:], Y_test, Nx=Nx, Ny=Ny, P=P, ds=ds)
    U_test[i,:,:], X_test[i,:,:] = input_construction( u_test[i,:], x_test, Nx=Nx, Ny=Ny, m=m, du=du)    

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


branch_layers = [m*(du*H_u+du), 1024, 1024, 1024, ds*n_hat]
trunk_layers  = [H_y*dy + dy , 1024, 1024, 1024, ds*n_hat]

model = DON(branch_layers, trunk_layers, m=m, P=P, mn=s_train_mean,  std=s_train_std)

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
    uCNN_super_all_test[:,idx,:], _, _  = predict_function(U_test , Y_test_in[idx,:], model=model, params=params, H=H_y)

uCNN_super_all_train = np.zeros_like(s_all_train).reshape(num_train, Nx*Ny, ds)
for i in range(0, Nx*Ny, P):
    idx = i + np.arange(0,P)
    uCNN_super_all_train[:,idx,:], _, _  = predict_function(U_train , Y_train_in[idx,:], model=model, params=params, H=H_y)

absolute_error_test, mean_test_error, test_error     = error_full_resolution(uCNN_super_all_test,  s_all_test,  tag='test', num_train=num_train,Nx=Nx, Ny=Ny)
absolute_error_train, mean_train_error, train_error  = error_full_resolution(uCNN_super_all_train, s_all_train, tag='train',num_train=num_test ,Nx=Nx, Ny=Ny)