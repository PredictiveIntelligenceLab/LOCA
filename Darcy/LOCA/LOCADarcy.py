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

# from jax.api import vjp
from jax import vjp
from absl import app
import jax
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

from kymatio.numpy import Scattering2D

from numpy.polynomial.legendre import leggauss

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

def pairwise_distances(dist,**arg):
    return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))

def euclid_distance(x,y,square=True):
    XX=jnp.dot(x,x)
    YY=jnp.dot(y,y)
    XY=jnp.dot(x,y)
    return XX+YY-2*XY

class DataGenerator(data.Dataset):
    def __init__(self, inputsxuy, inputsxu, y, s, z, w,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.inputsxuy = inputsxuy
        self.inputsxu  = inputsxu
        self.y = y
        self.s = s
        self.z = z
        self.w = w
        
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
        z = self.z[idx,:,:]
        w = self.w[idx,:,:]
        inputs = (inputsxu, y, z, w)
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
        pos_embedding =  jnp.concatenate((pex,pey),axis=-1) # [[x,pex],
                                                            # [y,pey]]
        x =  jnp.concatenate([x, pos_embedding], -1)
        return x

def scatteringTransform(sig, l=100, m=100, training_batch_size = 100):
    scattering = Scattering2D(J=1, L=3, max_order=2, shape=(32, 32))
    cwtmatr = np.zeros((training_batch_size, 768, 1))
    sig = np.array(sig)
    for i in range(0,training_batch_size):
        scatteringCoeffs = scattering(sig[i,:,:].reshape(32,32))
        cwtmatr[i,:,:] = scatteringCoeffs[:3,:,:].flatten()[:,None]
    return cwtmatr

class LOCA:
    def __init__(self, q_layers, g_layers, v_layers , m=100, P=100, jac_det=None):    
        # Network initialization and evaluation functions
        self.q_init, self.q_apply = self.init_NN(q_layers, activation=Gelu)
        self.in_shape = (-1, q_layers[0])
        self.out_shape, q_params = self.q_init(random.PRNGKey(10000), self.in_shape)

        self.v_init, self.v_apply = self.init_NN(v_layers, activation=Gelu)
        self.in_shape = (-1, v_layers[0])
        self.out_shape, v_params = self.v_init(random.PRNGKey(10000), self.in_shape)
        self.v_apply = jit(self.v_apply)

        self.g_init, self.g_apply = self.init_NN(g_layers, activation=Gelu)
        self.in_shape = (-1, g_layers[0])
        self.out_shape, g_params = self.g_init(random.PRNGKey(10000), self.in_shape)
        self.g_apply = jit(self.g_apply)

        # RBF kernel parameters
        beta = [10.]
        gamma = [0.2]

        # Model parameters
        params = (beta, gamma,q_params, g_params, v_params)

        self.jac_det = jac_det

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

        self.grads = []

        self.vdistance_function = vmap(pairwise_distances(euclid_distance))

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
    def RBF(self, X, Y, gamma, beta):
        d = self.vdistance_function(X, Y)
        return beta[0]*jnp.exp(-gamma[0]*d) + 1e-5

    @partial(jax.jit, static_argnums=0)
    def Mattern_32(self, X, Y, gamma, beta):
        d = self.vdistance_function(X, Y)
        return (1 + (jnp.sqrt(3)*gamma[0])*d)*beta[0]*jnp.exp(-(jnp.sqrt(3)*gamma[0])*d)

    @partial(jax.jit, static_argnums=0)
    def Mattern_52(self, X, Y, gamma, beta):
        d = self.vdistance_function(X, Y)
        return (1 + (jnp.sqrt(5)*gamma[0])*d + (5/3*gamma[0])*d**2)*beta[0]*jnp.exp(-(jnp.sqrt(5)*gamma[0])*d)

    @partial(jax.jit, static_argnums=0)
    def periodic(self, X, Y, gamma, beta):
        d = self.vdistance_function(X, Y)
        return jnp.exp(-2.0*jnp.sin(jnp.pi*d*beta[0])*gamma[0]**2)

    @partial(jax.jit, static_argnums=0)
    def RQK(self, X, Y, gamma, beta):
        d = self.vdistance_function(X, Y)
        return beta[0]*(1 + (1./3.)*gamma[0]*d)**(gamma[0]) + 1e-4

    @partial(jax.jit, static_argnums=0)
    def LOCA_net(self, params, inputs, ds=1):
        beta, gamma, q_params, g_params, v_params = params
        u, y, z, w = inputs
        y  = self.q_apply(q_params,y)
        z  = self.q_apply(q_params,z)

        K =  self.RBF(z, z, gamma, beta)
        Kzz =  jnp.sqrt(self.jac_det*jnp.einsum("ijk,ikl->ijl",K,w))

        K =  self.RBF(y, z, gamma, beta)
        Kyz =  jnp.sqrt(self.jac_det*jnp.einsum("ijk,ikl->ijl",K,w))

        mean_K = jnp.matmul(Kyz, jnp.swapaxes(Kzz,1,2))
        K = jnp.divide(K,mean_K)
        # K = jnp.linalg.matrix_power(jnp.divide(K,mean_K),2)
        # Ka = jnp.sum(jnp.divide(K,mean_K),axis=-1, keepdims=True)

        # mean_K = jnp.matmul(Ka, jnp.swapaxes(Ka,1,2))
        # K = jnp.divide(Ka,mean_K)

        g  = self.g_apply(g_params,z)
        g = self.jac_det*jnp.einsum("ijk,iklm,ik->ijlm",K,g.reshape(g.shape[0],g.shape[1], ds, int(g.shape[-1]/ds)),w[:,:,-1])
        g = jax.nn.softmax(g, axis=-1)


        v = self.v_apply(v_params, u.reshape(u.shape[0],1,u.shape[1]*u.shape[2]))
        v = v.reshape(v.shape[0],int(v.shape[2]/ds),ds)
        Guy = jnp.einsum("ijkl,ilk->ijk", g,v)
        return Guy

    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = np.mean((outputs.flatten() - y_pred.flatten())**2)
        return loss    

    @partial(jax.jit, static_argnums=0)
    def lossT(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = np.mean((outputs.flatten() - y_pred.flatten())**2)
        return loss    
    
    @partial(jax.jit, static_argnums=0)
    def L2errorT(self, params, batch):
        inputs, y = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(y_pred.flatten() - y.flatten(), 2)/norm(y_pred.flatten(),2)

    @partial(jax.jit, static_argnums=0)
    def L2error(self, params, batch):
        inputs, y = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(y_pred.flatten() - y.flatten(), 2)/norm(y_pred.flatten(),2)
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state), g

    def train(self, train_dataset, test_dataset, nIter = 10000):
        train_data = iter(train_dataset)
        test_data  = iter(test_dataset)

        pbar = trange(nIter)
        for it in pbar:
            train_batch = next(train_data)
            test_batch  = next(test_data)

            self.opt_state, g = self.step(next(self.itercount), self.opt_state, train_batch)

            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
                self.grads.append(g)

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


def predict_function(U_in, X_in, Y_in, P=128, m=100, P_test=1024,num_test=200, Nx=30, Ny=32,model=None,dy=2, training_batch_size=100,params= None, L=128, mode="train", X_sim=None, Y_sim=None, H=20, z= None, w=None):
    print("Predicting the solution for the full resolution")
    ds = 1
    y = np.expand_dims(Y_in,axis=0)
    y = np.tile(y,(num_test,1,1))
    uCNN_super_all = np.zeros((num_test, Nx*Ny,ds))
    inputs_trainxu = jnp.asarray(scatteringTransform(U_in, l=L, m=m, training_batch_size=num_test))
    # inputs_trainxu = jnp.asarray(U_in)
    pos_encodingy = PositionalEncodingY(y,int(y.shape[1]*y.shape[2]), max_len = Nx*Ny, H=H)
    y_train  = pos_encodingy.forward(y)
    uCNN_super_all = model.predict(params, (inputs_trainxu,y_train, z, w))
    return uCNN_super_all, y[:,:,1:2], y[:,:,0:1]


def error_full_resolution(uCNN_super_all, s_all,tag='train', num_train=1000, P=128, Nx=32, Ny=32, idx=None):
    test_error_u = []
    z = uCNN_super_all.reshape(num_train,Nx,Ny)
    s = s_all.reshape(num_train,Nx,Ny)
    s = np.swapaxes(s,1,2)
    for i in range(0,num_train):
        test_error_u.append(norm(s[i,:,:]- z[i,:,:], 2)/norm(s[i,:,:], 2))
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


TRAINING_ITERATIONS = 20000
P = 128
m = 1024
L = 1
N_hat = 1
num_train = 1000
num_test  = 1000
casenum_train = 2
casenum_test  = 2
training_batch_size = 100
dx = 2
du = 1
dy = 2
ds = 1
n_hat  = 100
l  = 100
Nx = 32
Ny = 32
H = 6

d = np.load("../Data/train_darcy_dataset.npz")
u_train   = d["U_train"]
x_train   = d["X_train"]
Y_train   = d["Y_train"]
S_train   = d["s_train"]

d = np.load("../Data/test_darcy_dataset.npz")
u_test   = d["U_test"]
x_test   = d["X_test"]
Y_test   = d["Y_test"]
S_test   = d["s_test"]

polypoints = 14
lb = np.array([0.0, 0.0])
ub = np.array([1.0, 1.0])

# GLL nodes and weights in [-1,1]        
z1, w1 = leggauss(polypoints)
z2, w2 = leggauss(polypoints)

# Rescale nodes to [lb,ub]
x1 = 0.5*(ub[0] - lb[0])*(z1 + 1.0) + lb[0]
x2 = 0.5*(ub[1] - lb[1])*(z2 + 1.0) + lb[1]

# Determinant of Jacobian of mapping [lb,ub]-->[-1,1]^2
jac_det = 0.5**2 * (ub[0]-lb[0]) * (ub[1]-lb[1])

Z_1, Z_2 = np.meshgrid(z1,z2,indexing="ij")

Z = np.concatenate((Z_1.flatten()[:,None], Z_2.flatten()[:,None]), axis=-1)
Z = np.tile(Z,(num_train,1,1))

W = np.outer(w1, w2).flatten()[:,None]
W = np.tile(W,(num_train,1,1))

polypoints = polypoints**dy

Y_train_in = Y_train
Y_test_in = Y_test

s_all_test = S_test[:num_test,:]
s_all_train = S_train[:num_train,:]

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

z = jnp.asarray(Z)
w = jnp.asarray(W)

X_train = jnp.reshape(X_train,(num_train,m,dx))
U_train = np.reshape(U_train,(num_train,m,du))
y_train = jnp.reshape(y_train,(num_train,P,dy))
s_train = jnp.reshape(s_train,(num_train,P,ds))

X_test = jnp.reshape(X_test,(num_test,m,dx))
U_test = np.reshape(U_test,(num_test,m,du))
y_test = jnp.reshape(y_test,(num_test,P,dy))
s_test = jnp.reshape(s_test,(num_test,P,ds))

z = jnp.reshape(z,(num_test,polypoints,dy))
w = jnp.reshape(w,(num_test,polypoints,1))

y_train_pos = y_train
y_train_posT = y_test

pos_encodingy  = PositionalEncodingY(y_train_pos,int(y_train_pos.shape[1]*y_train_pos.shape[2]), max_len = P, H=H)
y_train  = pos_encodingy.forward(y_train)
del pos_encodingy

pos_encodingy  = PositionalEncodingY(z,int(z.shape[1]*z.shape[2]), max_len = polypoints, H=H)
z  = pos_encodingy.forward(z)
del pos_encodingy

pos_encodingyt = PositionalEncodingY(y_train_posT,int(y_train_posT.shape[1]*y_train_posT.shape[2]), max_len = P, H=H)
y_test   = pos_encodingyt.forward(y_test)
del pos_encodingyt


inputs_trainxu = jnp.asarray(scatteringTransform(U_train, l=l, m=m, training_batch_size=num_train))
inputs_testxu  = jnp.asarray(scatteringTransform(U_test , l=l, m=m, training_batch_size=num_test))

train_dataset = DataGenerator(inputs_trainxu, inputs_trainxu, y_train, s_train, z, w, training_batch_size)
train_dataset = iter(train_dataset)

test_dataset = DataGenerator(inputs_testxu, inputs_testxu, y_test, s_test, z, w, training_batch_size)
test_dataset = iter(test_dataset)

q_layers = [L*dy+H*dy, 100, 100, l]
v_layers = [768*du, 100, 100, ds*n_hat]
g_layers  = [l, 100, 100, ds*n_hat]

# Define model
model = LOCA(q_layers, g_layers, v_layers, m=m, P=P, jac_det=jac_det)

model.count_params(model.get_params(model.opt_state))

start_time = timeit.default_timer()
model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

params = model.get_params(model.opt_state)

uCNN_super_all_train, X, T  = predict_function(U_train, X_train, Y_train_in, model=model, P=P, L= L,Nx=Nx,Ny=Ny, params=params, H=H, z=z, w=w, num_test=num_train)
uCNN_super_all_test , X, T  = predict_function(U_test, X_test, Y_test_in, model=model, P=P, L=L,Nx=Nx,Ny=Ny, params=params,H=H, z=z, w=w, num_test=num_test)

absolute_error_test, mean_test_error, test_error     = error_full_resolution(uCNN_super_all_test,  s_all_test,  tag='test', P=P, Nx=Nx, Ny=Ny, idx = None, num_train=num_train)
absolute_error_train, mean_train_error, train_error  = error_full_resolution(uCNN_super_all_train, s_all_train, tag='train',P=P, Nx=Nx, Ny=Ny, idx = None, num_train=num_test)
