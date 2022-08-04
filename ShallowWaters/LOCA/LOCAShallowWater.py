from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax.core import as_named_shape

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

from kymatio.numpy import Scattering2D

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

def pairwise_distances(dist,**arg):
    return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))

def euclid_distance(x,y,square=True):
    XX=jnp.dot(x,x)
    YY=jnp.dot(y,y)
    XY=jnp.dot(x,y)
    return XX+YY-2*XY

class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u  = u
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
        inputsxu  = self.u[idx,:,:]
        y = self.y[idx,:,:]
        inputs = (inputsxu, y)
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

def scatteringTransform(sig, l=100, m=100, training_batch_size = 100):
    scattering = Scattering2D(J=1, L=3, max_order=2, shape=(32, 32))
    cwtmatr = np.zeros((training_batch_size, 768, 1))
    sig = np.array(sig)
    for i in range(0,training_batch_size):
        scatteringCoeffs = scattering(sig[i,:,:].reshape(32,32))
        cwtmatr[i,:,:] = scatteringCoeffs[:,:,:].flatten()[:,None]
    return cwtmatr

class LOCA:
    def __init__(self, q_layers, g_layers, v_layers , m=100, P=100, H=100):    
        # Network initialization and evaluation functions

        self.encoder_init2, self.encoder_apply2 = self.init_NN(q_layers, activation=Gelu)
        self.in_shape = (-1, q_layers[0])
        self.out_shape, encoder_params2 = self.encoder_init2(random.PRNGKey(10000), self.in_shape)
        self.encoder_apply2 = self.encoder_apply2

        self.v_init, self.v_apply = self.init_NN(v_layers, activation=Gelu)
        self.in_shape = (-1, v_layers[0])
        self.out_shape, v_params = self.v_init(random.PRNGKey(10000), self.in_shape)
        self.v_apply = self.v_apply

        self.g_init, self.g_apply = self.init_NN(g_layers, activation=Gelu)
        self.in_shape = (-1, g_layers[0])
        self.out_shape, g_params = self.g_init(random.PRNGKey(10000), self.in_shape)
        self.g_apply = self.g_apply

        beta = [1.]
        gamma = [1.]

        params = (beta,gamma,encoder_params2, g_params, v_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

        self.vdistance_function = vmap(pairwise_distances(euclid_distance))

        print("Model initialized")
        
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
    # def LOCA_net(self, params, inputs, ds=3):
    #     beta, gamma, encoder_params2, g_params, v_params = params
    #     inputsxu, inputsy = inputs
    #     inputsy  = self.encoder_apply2(encoder_params2,inputsy)

    #     attn_logits   = self.vdistance_function(inputsy, inputsy)
    #     K =  beta[0]*jnp.exp(- gamma[0]*attn_logits)
    #     Kxx =  jnp.sqrt((1./K.shape[1])*jnp.sum(K ,axis=-1,keepdims=True))
    #     mean_K = jnp.matmul(Kxx, jnp.swapaxes(Kxx,1,2))
    #     K = jnp.divide(K,mean_K)
        
    #     g  = self.g_apply(g_params, inputsy)
    #     g = (1./K.shape[1])*jnp.einsum("ijk,iklm->ijlm",K,g.reshape(inputsy.shape[0], inputsy.shape[1], ds, int(g.shape[2]/ds)))
    #     g = jax.nn.softmax(g, axis=-1)

    #     value_heads = self.v_apply(v_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
    #     value_heads = value_heads.reshape(value_heads.shape[0],int(value_heads.shape[2]/ds),ds)
    #     Guy = jnp.einsum("ijkl,ilk->ijk", g,value_heads)
    #     return Guy

    def LOCA_net(self, params, inputs, ds=3):
        beta, gamma, q_params, g_params, v_params = params
        inputsxu, inputsy = inputs
        inputsy  = self.encoder_apply2(q_params,inputsy)

        d = self.vdistance_function(inputsy, inputsy)
        K =  beta[0]*jnp.exp(-gamma[0]*d)
        Kzz =  jnp.sqrt((1./K.shape[1])*jnp.sum(K ,axis=1,keepdims=True))
        Kyz =  jnp.sqrt((1./K.shape[1])*jnp.sum(K ,axis=-1,keepdims=True))
        mean_K = jnp.matmul(Kyz, Kzz)
        K = jnp.divide(K,mean_K)

        g  = self.g_apply(g_params, inputsy)
        g = (1./K.shape[1])*jnp.einsum("ijk,ikml->ijml",K,g.reshape(inputsy.shape[0], inputsy.shape[1], ds, int(g.shape[2]/ds)))
        g = jax.nn.softmax(g, axis=-1)

        value_heads = self.v_apply(v_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        value_heads = value_heads.reshape(value_heads.shape[0],int(value_heads.shape[2]/ds),ds)
        attn_vec = jnp.einsum("ijkl,ilk->ijk", g,value_heads)

        return attn_vec


    @partial(jax.jit, static_argnums=0)
    def loss(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = np.mean((outputs- y_pred)**2)
        return loss    

    @partial(jax.jit, static_argnums=0)
    def lossT(self, params, batch):
        inputs, outputs = batch
        y_pred = self.LOCA_net(params,inputs)
        loss = np.mean((outputs - y_pred)**2)
        return loss    
    
    @partial(jax.jit, static_argnums=0)
    def L2errorT(self, params, batch):
        inputs, y = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)


    @partial(jax.jit, static_argnums=0)
    def L2error(self, params, batch):
        inputs, y = batch
        y_pred = self.LOCA_net(params,inputs)
        return norm(y.flatten() - y_pred.flatten(), 2)/norm(y.flatten(),2)
    
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        return self.opt_update(i, g, opt_state)

    def train(self, train_dataset, test_dataset, nIter = 10000):
        train_data = iter(train_dataset)
        print("Inputs made iterable")
        if test_dataset is not None:
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
        else:
           pbar = trange(nIter)
           for it in pbar:
               train_batch = next(train_data)

               self.opt_state = self.step(next(self.itercount), self.opt_state, train_batch)
               
               if it % 100 == 0:
                   params = self.get_params(self.opt_state)

                   loss_train = self.loss(params, train_batch)

                   errorTrain = self.L2error(params, train_batch)

                   self.loss_log.append(loss_train)

                   pbar.set_postfix({'Training loss': loss_train, 
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


def predict_function(inputs_trainxu,y, model=None,params= None, H=None):
    uCNN_super_all = model.predict(params, (inputs_trainxu, y))
    return uCNN_super_all, y[:,:,0:1], y[:,:,1:2], y[:,:,2:3]


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


def minmax(a, n, mean):
    minpos = a.index(min(a))
    maxpos = a.index(max(a)) 
    meanpos = min(range(len(a)), key=lambda i: abs(a[i]-mean))

    print("The maximum is at position", maxpos)  
    print("The minimum is at position", minpos)
    print("The mean is at position", meanpos)
    return minpos,maxpos,meanpos

TRAINING_ITERATIONS = 80000
P = 128
m = 1024
L = 1
T = 1
num_train = 1000
num_test  = 1000
casenum_train = 2
casenum_test  = 2
training_batch_size = 100
dx = 3
du = 3
dy = 3
ds = 3
n_hat  = 480
l  = 100
Nx = 32
Ny = 32
Nt = 5
Ng = 0
H = 2

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
s_train = np.zeros((num_train,P,ds))
y_train = np.zeros((num_train,P,dy))
U_train = np.zeros((num_train,m,du))
X_train = np.zeros((num_train,m,dx))

s_test = np.zeros((num_test,P,ds))
y_test = np.zeros((num_test,P,dy))
U_test = np.zeros((num_test,m,du))
X_test = np.zeros((num_test,m,dx))

for i in range(0,num_train):
    s_train[i,:,:], y_train[i,:,:] = output_construction(S_train[i,:,:,:,:], T, CX, CY, Ng,P=P,Nt=Nt)
    U_train[i,:,:] = u_train[i,:,:,:].reshape(Nx*Ny,du)

for i in range(num_test):
    s_test[i,:,:], y_test[i,:,:] = output_construction(S_test[i,:,:,:,:], T, CX, CY, Ng,P=P,Nt=Nt)
    U_test[i,:,:] = u_test[i,:,:,:].reshape(Nx*Ny,du)

print("Dataset created")

X_train = jnp.asarray(X_train)
U_train =  np.asarray(U_train)
y_train = jnp.asarray(y_train)
s_train = jnp.asarray(s_train)

X_test = jnp.asarray(X_test)
U_test =  np.asarray(U_test)
y_test = jnp.asarray(y_test)
s_test = jnp.asarray(s_test)

X_train = jnp.reshape(X_train,(num_train,m,dx))
U_train =  np.reshape(U_train,(num_train,m,du))
y_train = jnp.reshape(y_train,(num_train,P,dy))
s_train = jnp.reshape(s_train,(num_train,P,ds))

X_test = jnp.reshape(X_test,(num_test,m,dx))
U_test =  np.reshape(U_test,(num_test,m,du))
y_test = jnp.reshape(y_test,(num_test,P,dy))
s_test = jnp.reshape(s_test,(num_test,P,ds))

inputs_trainxu = np.zeros((num_train,768,3))
inputs_trainxu[:,:,0:1] = jnp.asarray(scatteringTransform(U_train[:,:,0:1], l=L, m=m, training_batch_size=num_train))
inputs_trainxu[:,:,1:2] = jnp.asarray(scatteringTransform(U_train[:,:,1:2], l=L, m=m, training_batch_size=num_train))
inputs_trainxu[:,:,2:3] = jnp.asarray(scatteringTransform(U_train[:,:,2:3], l=L, m=m, training_batch_size=num_train))
inputs_trainxu = jnp.array(inputs_trainxu)

inputs_testxu = np.zeros((num_test,768,3))
inputs_testxu[:,:,0:1] = jnp.asarray(scatteringTransform(U_test[:,:,0:1], l=L, m=m, training_batch_size=num_test))
inputs_testxu[:,:,1:2] = jnp.asarray(scatteringTransform(U_test[:,:,1:2], l=L, m=m, training_batch_size=num_test))
inputs_testxu[:,:,2:3] = jnp.asarray(scatteringTransform(U_test[:,:,2:3], l=L, m=m, training_batch_size=num_test))
inputs_testxu = jnp.array(inputs_testxu)
print("Model inputs created")

pos_encodingy  = PositionalEncodingY(y_train,int(y_train.shape[1]*y_train.shape[2]), max_len = P, H=H) 
y_train  = pos_encodingy.forward(y_train) 
del pos_encodingy 

pos_encodingyt = PositionalEncodingY(y_test,int(y_test.shape[1]*y_test.shape[2]), max_len = P,H=H) 
y_test   = pos_encodingyt.forward(y_test) 
del pos_encodingyt 

train_dataset = DataGenerator(inputs_trainxu, y_train, s_train, training_batch_size)
train_dataset = iter(train_dataset)

test_dataset = DataGenerator(inputs_testxu, y_test, s_test, training_batch_size)
test_dataset = iter(test_dataset)

q_layers = [L*dy+H*dy, 1024, l]
v_layers = [768*du, 1024, ds*n_hat]
g_layers  = [l, 1024, ds*n_hat]

print("DataGenerator defined")

model = LOCA(q_layers, g_layers, v_layers, m=m, P=P, H=H) 

model.count_params(model.get_params(model.opt_state))

start_time = timeit.default_timer()
model.train(train_dataset, None, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

params = model.get_params(model.opt_state)

TT, XX, YY = np.meshgrid(T, CX, CY, indexing="ij")

TT = np.expand_dims(TT,axis=0)
XX = np.expand_dims(XX,axis=0)
YY = np.expand_dims(YY,axis=0)

TT = np.tile(TT,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)
XX = np.tile(XX,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)
YY = np.tile(YY,(num_test,1,1)).reshape(num_test,Nx*Ny*Nt,1)

Y_test_in = np.concatenate((TT, XX, YY),axis=-1)
Y_train_in = np.concatenate((TT, XX, YY),axis=-1)

pos_encodingy  = PositionalEncodingY(Y_train_in,int(Y_train_in.shape[1]*Y_train_in.shape[2]), max_len = Y_train_in.shape[1], H=H)
Y_train_in  = pos_encodingy.forward(Y_train_in)
del pos_encodingy

pos_encodingy  = PositionalEncodingY(Y_test_in,int(Y_test_in.shape[1]*Y_test_in.shape[2]), max_len = Y_test_in.shape[1], H=H)
Y_test_in  = pos_encodingy.forward(Y_test_in)
del pos_encodingy

print("Predicting the solution for the full resolution")
uCNN_super_all_test = np.zeros_like(S_test).reshape(num_test, Nx*Ny*Nt, ds)
for i in range(0, Nx*Ny*Nt, P):
    idx = i + np.arange(0,P)
    uCNN_super_all_test[:,idx,:], T_out, X, Y  = predict_function(inputs_testxu , Y_test_in[:,idx,:], model=model, params=params, H=H)

uCNN_super_all_train = np.zeros_like(S_train).reshape(num_train, Nx*Ny*Nt, ds)
for i in range(0, Nx*Ny*Nt, P):
    idx = i + np.arange(0,P)
    uCNN_super_all_train[:,idx,:], T, X, Y  = predict_function(inputs_trainxu , Y_train_in[:,idx,:], model=model, params=params, H=H)

absolute_error_train, mean_train_error_rho, mean_train_error_u, mean_train_error_v, train_error  = error_full_resolution(uCNN_super_all_train,S_train,tag='train',P=P,Nx=Nx, Ny=Ny, Nt=Nt, idx = None, num_train=num_train)
absolute_error_test, mean_test_error_rho, mean_test_error_u, mean_test_error_v, test_error  = error_full_resolution(uCNN_super_all_test,S_test,tag='test',P=P,Nx=Nx, Ny=Ny, Nt=Nt, idx = None, num_train=num_test)

