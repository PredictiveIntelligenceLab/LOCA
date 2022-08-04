
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax.interpreters.xla import Backend

from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
import argparse
from jax.example_libraries.stax import Dense, Gelu
from jax.example_libraries import stax
import os

from scipy.integrate import solve_ivp

import timeit

from jax.example_libraries import optimizers

from absl import app
import jax
from jax import vjp
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import norm

from jax import random, grad, vmap, jit
from functools import partial 

from torch.utils import data

from scipy import interpolate

from tqdm import trange
from numpy.polynomial.legendre import leggauss

import itertools
import torch

from kymatio.numpy import Scattering2D

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmin(memory_available))
os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"

def output_construction(s, X, Y,P=100,ds=1, dy=2, N=1000,Nx=100,Ny=100, Nt=2):
    S_all = np.zeros((P,ds))
    Y_all = np.zeros((P,ds))
    x = np.random.randint(Nx, size=P)
    y = np.random.randint(Ny, size=P)
    Y_all = np.concatenate((X[x][range(P),y][:,None], Y[x][range(P),y][:,None]),axis=-1)
    S_all[:,:] = s[x][range(P), y]
    return S_all, Y_all

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
    def __init__(self, inputsxuy, y, s, z, w,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.inputsxuy = inputsxuy
        self.y = y
        self.s = s
        self.z = z
        self.w = w
        
        self.N = inputsxuy.shape[0]
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
        inputsxu  = self.inputsxuy[idx,:,:]
        y = self.y[idx,:,:]
        z = self.z[idx,:,:]
        w = self.w[idx,:,:]
        inputs = (inputsxu, y, z, w)
        return inputs, s

class PositionalEncodingY: 
    def __init__(self, Y, d_model, max_len = 100, H=4): 
        self.d_model = int(np.ceil(d_model/4)*2)
        self.Y = Y 
        self.max_len = max_len 
        self.H = H
 
    def forward(self, x):
        pet = np.zeros((x.shape[0], self.max_len, self.H))
        pex = np.zeros((x.shape[0], self.max_len, self.H))
        pey = np.zeros((x.shape[0], self.max_len, self.H))
        X = jnp.take(self.Y, 0, axis=2)[:,:,None]
        Y = jnp.take(self.Y, 1, axis=2)[:,:,None]
        positionX = jnp.tile(X,(1,1,self.H))
        positionY = jnp.tile(Y,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,0::2], jnp.cos(positionX[:,:,0::2] * div_term))
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,1::2], jnp.sin(positionX[:,:,1::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,0::2], jnp.cos(positionY[:,:,0::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,1::2], jnp.sin(positionY[:,:,1::2] * div_term))
        pos_embedding =  jnp.concatenate((pex,pey),axis=-1)
        x =  jnp.concatenate([x, pos_embedding], -1)
        return x

def scattering(sig, l=100, m=100, training_batch_size = 100):
    scattering = Scattering2D(J=1, L=16, max_order=2, shape=(28, 28))
    cwtmatr = np.zeros((training_batch_size, 3332, 1))
    sig = np.array(sig)
    for i in range(0,training_batch_size):
        scatteringCoeffs = scattering(sig[i,:,:].reshape(28,28))
        cwtmatr[i,:,:] = scatteringCoeffs.flatten()[:,None]
    return cwtmatr

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.shape[0]
        diff_norms = jnp.linalg.norm(y.reshape(num_examples,-1) - x.reshape(num_examples,-1), self.p, 1)
        y_norms = jnp.linalg.norm(y.reshape(num_examples,-1), self.p, 1)
        return jnp.sum(diff_norms/y_norms)/100.

    def __call__(self, x, y):
        return self.rel(x, y)

class LOCA:
    def __init__(self, q_layers, g_layers, v_layers , m=100, P=100, X=None, Y=None, Yt=None, H=30, batch_size=100, jac_det=None):    
        # Network initialization and evaluation functions

        seed = np.random.randint(10000)
        self.q_init, self.q_apply = self.init_NN(q_layers, activation=Gelu)
        self.in_shape = (-1, q_layers[0])
        self.out_shape, q_params = self.q_init(random.PRNGKey(seed), self.in_shape)

        seed = np.random.randint(10000)
        self.v_init, self.v_apply = self.init_NN(v_layers, activation=Gelu)
        self.in_shape = (-1, v_layers[0])
        self.out_shape, v_params = self.v_init(random.PRNGKey(seed), self.in_shape)

        seed = np.random.randint(10000)
        self.g_init, self.g_apply = self.init_NN(g_layers, activation=Gelu)
        self.in_shape = (-1, g_layers[0])
        self.out_shape, g_params = self.g_init(random.PRNGKey(seed), self.in_shape)

        beta = [1.]
        gamma = [1.]

        params = (beta,gamma,q_params, g_params, v_params)

        self.jac_det = jac_det
 

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

        self.P = P
        self.L = 1

        self.batchsize = batch_size
        
        self.l2loss = LpLoss(size_average=False)

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

    def LOCA_net(self, params, inputs, ds=2):
        beta, gamma, q_params, g_params, v_params = params
        inputsxu, inputsy, inputsz, w = inputs
        inputsy  = self.q_apply(q_params,inputsy)
        inputsz  = self.q_apply(q_params,inputsz)

        d = self.vdistance_function(inputsz, inputsz)
        K =  beta[0]*jnp.exp(-gamma[0]*d)
        Kzz =  jnp.sqrt(self.jac_det*jnp.matmul(K,w))

        d = self.vdistance_function(inputsy, inputsz)
        K =  beta[0]*jnp.exp(-gamma[0]*d)
        Kyz =  jnp.sqrt(self.jac_det*jnp.matmul(K,w))
        mean_K = jnp.matmul(Kyz, jnp.swapaxes(Kzz,1,2))
        K = jnp.divide(K,mean_K)

        g  = self.g_apply(g_params, inputsz)
        g = self.jac_det*jnp.einsum("ijk,iklm,ik->ijlm",K,g.reshape(g.shape[0],g.shape[1], ds, int(g.shape[-1]/ds)),w[:,:,-1])
        g = jax.nn.softmax(g, axis=-1)

        v = self.v_apply(v_params, inputsxu.reshape(inputsxu.shape[0],1,inputsxu.shape[1]*inputsxu.shape[2]))
        v = v.reshape(v.shape[0],int(v.shape[2]/ds),ds)
        attn_vec = jnp.einsum("ijkl,ilk->ijk", g,v)
        return attn_vec
     
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

    # def train(self, train_dataset, test_dataset, nIter = 10000):
    def train(self, train_dataset, nIter = 10000):
        train_data = iter(train_dataset)
        # test_data  = iter(test_dataset)

        pbar = trange(nIter)
        for it in pbar:
            train_batch = next(train_data)
            # test_batch  = next(test_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, train_batch)
            
            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                loss_train = self.loss(params, train_batch)
                # loss_test  = self.lossT(params, test_batch)

                errorTrain = self.L2error(params, train_batch)
                # errorTest  = self.L2errorT(params, test_batch)

                self.loss_log.append(loss_train)

                pbar.set_postfix({'Training loss': loss_train, 
                                  'Train error':   errorTrain})
                                #   'Testing loss' : loss_test,
                                #   'Test error':    errorTest,

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


def predict_function(U_in, Y_in, P=128, m=100, P_test=1024,num_test=1000, Nx=30, Nt=100, Ny=32,model=None,dy=2, training_batch_size=100,params= None, L=100, mode="train", z=None, w=None,H=100):
    print("Predicting the solution for the full resolution")
    dx = 2
    du = 2
    dy = 2
    ds = 2
    
    if mode=="train":
       predict = model.predict
    if mode=="test":
       predict= model.predictT

    y = np.expand_dims(Y_in,axis=0)
    y = np.tile(y,(num_test,1,1))
    s_super_all = np.zeros((num_test,Nt*Nx*Ny,ds))

    inputs_trainxu = np.zeros((num_test, 3332,du))
    inputs_trainxu[:,:,0:1] = jnp.asarray(scattering(U_in[:,:,0:1], l=L, m=m, training_batch_size=num_test))
    inputs_trainxu[:,:,1:2] = jnp.asarray(scattering(U_in[:,:,1:2], l=L, m=m, training_batch_size=num_test))
    inputs_trainxu = jnp.array(inputs_trainxu)
    for i in range(0,Nx*Ny*Nt,P):
        s_super_loc = np.zeros((num_test, P,ds))
        idx1 = i + np.arange(0,P)
        Y_super = y[:,idx1,:]
        pos_encodingy = PositionalEncodingY(Y_super,int(Y_super.shape[1]*Y_super.shape[2]), max_len = P, H=H)

        y_trainT = jnp.tile(jnp.reshape(Y_super,(num_test,P,dy))[:,:,0:1],(1,1,L))
        y_trainX = jnp.tile(jnp.reshape(Y_super,(num_test,P,dy))[:,:,1:2],(1,1,L))
        y_trainY = jnp.tile(jnp.reshape(Y_super,(num_test,P,dy))[:,:,2:3],(1,1,L))
        y_train  = jnp.concatenate((y_trainT, y_trainX, y_trainY),axis=-1)

        y_train  = pos_encodingy.forward(y_train)
        del pos_encodingy

        for j in range(0, U_in.shape[0],training_batch_size):
            idx = j + np.arange(0,training_batch_size,1)
            s_super_loc[idx,:,:] = predict(params, (inputs_trainxu[idx,:,:], y_train[idx,:,:], z, w))
        s_super_all[:,idx1,:] = s_super_loc
    return s_super_all, y[:,:,0:1], y[:,:,1:2]


def error_full_resolution(s_super_all, s_all,tag='train', num_train=1000,P=128, Nx=30, Ny=30, Nt=10, idx=None, ds=2):
    z = s_super_all.reshape(num_train,Nt*Nx*Ny,ds)
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
training_batch_size = 100
dx = 2
du = 2
dy = 2
ds = 2
n_hat  = 500
l  = 100
Nx = 28
Ny = 28
L = 1
H = 10

idxT = [11]
Nt = len(idxT)
d = np.load("../Data/MMNIST_dataset_train.npz")
dispx_allsteps_train = d["dispx_allsteps_train"][:num_train,11,:,:,None]
dispy_allsteps_train = d["dispy_allsteps_train"][:num_train,11,:,:,None]
u_trainx = d["dispx_allsteps_train"][:num_train,7,:,:,None]
u_trainy = d["dispy_allsteps_train"][:num_train,7,:,:,None]

S_train = np.concatenate((dispx_allsteps_train,dispy_allsteps_train),axis=-1)
u_train  = np.concatenate((u_trainx,u_trainy),axis=-1)

print("Dataset loaded")

polypoints = 20
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
Z = np.tile(Z,(training_batch_size,1,1))

W = np.outer(w1, w2).flatten()[:,None]
W = np.tile(W,(training_batch_size,1,1))

polypoints = polypoints**dy

# in_noise_train = 0.15*np.random.normal(loc=0.0, scale=1.0, size=(u_train.shape))
# u_train = u_train + in_noise_train

X = np.zeros((Nx,Ny))
Y = np.zeros((Nx,Ny))

dx = 0.037037037037037035
for kk in range(0,Nx):
        for jj in range(0,Ny):
                X[kk,jj] = jj*dx #+ 0.5 # x is columns
                Y[kk,jj] = kk*dx #+ 0.5 # y is rows 
Y_train = np.concatenate((X.flatten()[:,None], Y.flatten()[:,None]),axis=-1)
Y_train_in = Y_train

Y_test  = np.concatenate((X.flatten()[:,None], Y.flatten()[:,None]),axis=-1)
Y_test_in = Y_test

s_train = np.zeros((num_train*N_hat,P,ds))
y_train = np.zeros((num_train*N_hat,P,dy))
U_train = np.zeros((num_train*N_hat,m,du))

for j in range(0,N_hat):
    for i in range(0,num_train):
        s_train[i + j*num_train,:,:], y_train[i+ j*num_train,:,:] = output_construction(S_train[i,:,:,:], X, Y, P=P,Nt=Nt, Nx=Nx, Ny=Ny, ds=ds, dy=dy)
        U_train[i+ j*num_train,:,:] = u_train[i,:,:,:].reshape(Nx*Ny,du)

num_train = num_train*N_hat

z = jnp.asarray(Z)
w = jnp.asarray(W)

del S_train, dispx_allsteps_train, dispy_allsteps_train, u_train, Z, W, u_trainx, u_trainy

U_train = jnp.reshape(U_train,(num_train,m,du))
y_train = jnp.reshape(y_train,(num_train,P,dy))
s_train = jnp.reshape(s_train,(num_train,P,ds))

z = jnp.reshape(z,(training_batch_size,polypoints,dy))
w = jnp.reshape(w,(training_batch_size,polypoints,1))

pos_encodingy  = PositionalEncodingY(y_train,int(y_train.shape[1]*y_train.shape[2]), max_len = P, H=H)
y_train  = pos_encodingy.forward(y_train)
del pos_encodingy

pos_encodingy  = PositionalEncodingY(z,int(z.shape[1]*z.shape[2]), max_len = polypoints, H=H)
z  = pos_encodingy.forward(z)
del pos_encodingy

inputs_trainxu = np.zeros((num_train,3332,du))
inputs_trainxu[:,:,0:1] = jnp.asarray(scattering(U_train[:,:,0:1], l=L, m=m, training_batch_size=num_train))
inputs_trainxu[:,:,1:2] = jnp.asarray(scattering(U_train[:,:,1:2], l=L, m=m, training_batch_size=num_train))
inputs_trainxu = jnp.array(inputs_trainxu)

print("Dataset preprocessed")


train_dataset = DataGenerator(inputs_trainxu, jnp.asarray(y_train), jnp.asarray(s_train), z, w, training_batch_size)
train_dataset = iter(train_dataset)

encoder_layers2 = [L*dy+H*dy, 256, 256, l]
weights_layers = [3332*du, 256, 256, ds*n_hat]
g_layers  = [l, 256, 256, ds*n_hat]

model = LOCA(encoder_layers2, g_layers, weights_layers, m=m, P=P, H=H, batch_size=training_batch_size, jac_det=jac_det)

del inputs_trainxu, y_train, U_train, Y_train_in, s_train

print("P is equal to %d"%(P))
model.count_params(model.get_params(model.opt_state))

start_time = timeit.default_timer()
model.train(train_dataset, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

params = model.get_params(model.opt_state)

del train_dataset

d = np.load("../Data/MMNIST_dataset_test.npz")
dispx_allsteps_test = d["dispx_allsteps_test"][:num_test,11,:,:,None]
dispy_allsteps_test = d["dispy_allsteps_test"][:num_test,11,:,:,None]
u_testx = d["dispx_allsteps_test"][:num_test,7,:,:,None]
u_testy = d["dispy_allsteps_test"][:num_test,7,:,:,None]
S_test  = np.concatenate((dispx_allsteps_test,dispy_allsteps_test),axis=-1)

u_test   = np.concatenate((u_testx,u_testy),axis=-1)
s_test = np.zeros((num_test,P,ds))
y_test = np.zeros((num_test,P,1,dy))
U_test = np.zeros((num_test,m,du))

for i in range(num_test):
    a, b = output_construction(S_test[i,:,:,:], X,Y, P=P,Nt=Nt, Nx=Nx, Ny=Ny, ds=ds, dy=dy)
    s_test[i,:,:], y_test[i,:,:] = output_construction(S_test[i,:,:,:], X,Y, P=P,Nt=Nt, Nx=Nx, Ny=Ny, ds=ds, dy=dy)
    U_test[i,:,:] = u_test[i,:,:,:].reshape(Nx*Ny,du)

U_test = jnp.reshape(U_test,(num_test,m,du))
y_test = jnp.reshape(y_test,(num_test,P,dy))
s_test = jnp.reshape(s_test,(num_test,P,ds))
y_train_posT = y_test
pos_encodingyt = PositionalEncodingY(y_train_posT,int(y_train_posT.shape[1]*y_train_posT.shape[2]), max_len = P, H=H)
y_test   = pos_encodingyt.forward(y_test)
del pos_encodingyt

tag = "CN"
in_noise_test  = 0.15*np.random.normal(loc=0.0, scale=1.0, size=(u_test.shape))
u_test = u_test + in_noise_test

U_test = np.zeros((num_test,m,du))

for i in range(num_test):
    U_test[i,:,:] = u_test[i,:,:,:].reshape(Nx*Ny,du)

inputs_testxu = np.zeros((num_test,3332,du))
inputs_testxu[:,:,0:1] = jnp.asarray(scattering(U_test[:,:,0:1], l=L, m=m, training_batch_size=num_test))
inputs_testxu[:,:,1:2] = jnp.asarray(scattering(U_test[:,:,1:2], l=L, m=m, training_batch_size=num_test))
inputs_testxu = jnp.array(inputs_testxu)

Z = np.concatenate((Z_1.flatten()[:,None], Z_2.flatten()[:,None]), axis=-1)
Z = np.tile(Z,(training_batch_size,1,1))

W = np.outer(w1, w2).flatten()[:,None]
W = np.tile(W,(training_batch_size,1,1))

z = jnp.asarray(Z)
w = jnp.asarray(W)
z = jnp.reshape(z,(training_batch_size,polypoints,dy))
w = jnp.reshape(w,(training_batch_size,polypoints,1))

pos_encodingy  = PositionalEncodingY(z,int(z.shape[1]*z.shape[2]), max_len = polypoints, H=H)
z  = pos_encodingy.forward(z)
del pos_encodingy

s_super_all_test = np.zeros((num_test, Nx*Ny*Nt, ds))

print("Predicting the solution for the full resolution")
s_super_all_test, X, Y = predict_function(U_test, Y_test_in, model=model, P=P, Nx=Nx, Ny=Ny, Nt=Nt, params=params, L=L,mode="test",  num_test=num_test, training_batch_size=training_batch_size, H=H, z=z, w=w)
absolute_error_test, mean_test_error_u, mean_test_error_v, test_error_u, test_error_v  = error_full_resolution(s_super_all_test,S_test,tag='test',P=P,Nx=Nx, Ny=Ny, Nt=Nt, idx = None, num_train=num_test)

s_super_all_test = np.asarray(s_super_all_test)
S_test = np.asarray(S_test)
absolute_error_test, mean_test_error_u, mean_test_error_v, test_error_u, test_error_v  = error_full_resolution(s_super_all_test,S_test,tag='test',P=P,Nx=Nx, Ny=Ny, Nt=Nt, idx = None, num_train=num_test)
