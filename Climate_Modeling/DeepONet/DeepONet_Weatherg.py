from jax.core import as_named_shape
from scipy import linalg, interpolate
from sklearn import gaussian_process as gp
from jax.example_libraries.stax import Dense, Gelu, Relu
from jax.example_libraries import stax
import os

import timeit

from jax.example_libraries import optimizers

from absl import app
from jax import vjp
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import norm

from jax import random, grad, jit
from functools import partial 

from torch.utils import data

from scipy import interpolate

from tqdm import trange
from math import sqrt

import itertools


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmax(memory_available))

os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()

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
        X1 = jnp.take(self.Y, 0, axis=2)[:,:,None]
        X2 = jnp.take(self.Y, 1, axis=2)[:,:,None]
        positionX1 = jnp.tile(X1,(1,1,self.H))
        positionX2 = jnp.tile(X2,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,0::2], jnp.cos(positionX1[:,:,0::2] * div_term))
        pex = jax.ops.index_update(pex, jax.ops.index[:,:,1::2], jnp.sin(positionX1[:,:,1::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,0::2], jnp.cos(positionX2[:,:,0::2] * div_term))
        pey = jax.ops.index_update(pey, jax.ops.index[:,:,1::2], jnp.sin(positionX2[:,:,1::2] * div_term))
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
                                                                      decay_rate=0.95))
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
        test_error_u.append(norm(s[i,:,0]- z[i,:,0], 2)/norm(s[i,:,0], 2))
    print("The average "+tag+" u error for the super resolution is %e, the standard deviation %e, the minimum error is %e and the maximum error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
    absolute_error = np.abs(z-s)
    return absolute_error, np.mean(test_error_u), test_error_u

def minmax(a, mean):
    minpos = a.index(min(a))
    maxpos = a.index(max(a)) 
    meanpos = min(range(len(a)), key=lambda i: abs(a[i]-mean))

    print("The maximum is at position", maxpos)  
    print("The minimum is at position", minpos)
    print("The mean is at position", meanpos)
    return minpos,maxpos,meanpos


def main(_):
    TRAINING_ITERATIONS = 100000
    P = 144
    m = int(72*72)
    num_train = 1825
    num_test  = 1825
    training_batch_size = 100
    du = 1
    dy = 2
    ds = 1
    n_hat  = 100
    Nx = 72
    Ny = 72
    H_y = 10
    H_u = 10

    d = np.load("../Data/weather_dataset.npz")
    u_train = d["U_train"][:num_train,:]
    S_train = d["S_train"][:num_train,:]/1000.
    Y_train = d["Y_train"]

    d = np.load("../Data/weather_dataset.npz")
    u_test = d["U_train"][-num_test:,:]
    S_test = d["S_train"][-num_test:,:]/1000.
    Y_test = d["Y_train"]

    Y_train_in = Y_train
    Y_test_in = Y_test

    s_all_test = S_test
    s_all_train = S_train

    s_train = np.zeros((num_train,P,ds))
    y_train = np.zeros((num_train,P,dy))
    U_train = np.zeros((num_train,m,du))

    s_test = np.zeros((num_test,P,ds))
    y_test = np.zeros((num_test,P,dy))
    U_test = np.zeros((num_test,m,du))
    
    for i in range(0,num_train):
        s_train[i,:,:], y_train[i,:,:] = output_construction(S_train[i,:], Y_train, Nx=Nx, Ny=Ny, P=P, ds=ds)
        U_train[i,:,:] = u_train[i,:][:,None]

    for i in range(num_test):
      s_test[i,:,:], y_test[i,:,:] = output_construction(S_test[i,:], Y_test, Nx=Nx, Ny=Ny, P=P, ds=ds)
      U_test[i,:,:] = u_test[i,:][:,None]

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

    print(U_test[0,0:20,:])

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
    
    branch_layers = [m*(du*H_u+du), 100, 100, 100, 100, ds*n_hat]
    trunk_layers  = [H_y*dy + dy, 100, 100, 100, 100, ds*n_hat]

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

    print(np.max(absolute_error_test), np.max(absolute_error_train))
    np.savez_compressed("Error_Weather_DeepONet_P%d"%(P), test_error = test_error)


if __name__ == '__main__':
  app.run(main)