from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy.polynomial import polyutils

from jax.experimental.stax import Dense, Gelu
from jax.experimental import stax
import os

from scipy.integrate import solve_ivp

import timeit

from jax.experimental import optimizers

from absl import app
import jax
import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import norm

from jax import random, grad, vmap, jit, vjp
from functools import partial 

from torch.utils import data

from tqdm import trange

import itertools

import scipy.signal as signal
from kymatio.numpy import Scattering1D

from jax.experimental.ode import odeint
from jax.config import config
from numpy.polynomial.legendre import leggauss

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return str(np.argmax(memory_available))

os.environ['CUDA_VISIBLE_DEVICES']= get_freer_gpu()
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"

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
        u  = self.u[idx,:,:]
        y = self.y[idx,:,:]
        inputs = (u, y)
        return inputs, s

class PositionalEncodingY:
    def __init__(self, Y, d_model, max_len = 100,H=20):
        self.d_model = d_model
        self.Y = Y
        self.max_len = max_len
        self.H = H

    @partial(jit, static_argnums=(0,))
    def forward(self, x):
        self.pe = np.zeros((x.shape[0], self.max_len, self.H))
        T = jnp.asarray(self.Y[:,:,0:1])
        position = jnp.tile(T,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        self.pe = jax.ops.index_update(self.pe, jax.ops.index[:,:,0::2], jnp.cos(position[:,:,0::2] * div_term))
        self.pe = jax.ops.index_update(self.pe, jax.ops.index[:,:,1::2], jnp.sin(position[:,:,1::2] * div_term))
        x =  jnp.concatenate([x, self.pe],axis=-1)
        return x

class PositionalEncodingU:
    def __init__(self, Y, d_model, max_len = 100,H=20):
        self.d_model = d_model
        self.Y = Y
        self.max_len = max_len
        self.H = H

    @partial(jit, static_argnums=(0,))
    def forward(self, x):
        self.pe = np.zeros((x.shape[0], self.max_len, self.H))
        T = jnp.asarray(self.Y[:,:,0:1])
        position = jnp.tile(T,(1,1,self.H))
        div_term = 2**jnp.arange(0,int(self.H/2),1)*jnp.pi
        self.pe = jax.ops.index_update(self.pe, jax.ops.index[:,:,0::2], jnp.cos(position[:,:,0::2] * div_term))
        self.pe = jax.ops.index_update(self.pe, jax.ops.index[:,:,1::2], jnp.sin(position[:,:,1::2] * div_term))
        x =  jnp.concatenate([x, self.pe],axis=-1)
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
            for i in range(0, num_layers-1):
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
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predictT(self, params, inputs):
        s_pred = self.DON(params,inputs)
        return s_pred

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


# Define RBF kernel
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)

# Geneate training data corresponding to one input sample
def generate_one_training_data(key, m=100, P=1):
    # Sample GP prior at a fine grid
    N = 512
    length_scale = 0.9
    gp_params = (1.0, length_scale)
    # key1, key2 = random.split(key,num=2)
    # z = random.uniform(key1, minval=-2, maxval=2)
    # output_scale = 10**z
    # z = random.uniform(key2, minval=-2, maxval=0)
    # length_scale = 10**z
    # gp_params = (output_scale, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(key, (N,)))

    # Create a callable interpolation function  
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), gp_sample)

    # Ijnput sensor locations and measurements
    x = jnp.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Output sensor locations and measurements
    y = jnp.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)
    return u, y, s

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, m=100, P=100):
    # Sample GP prior at a fine grid
    N = 512
    length_scale = 0.1
    gp_params = (1.0, length_scale)
    # key1, key2 = random.split(key,num=2)
    # z = random.uniform(key1, minval=-2, maxval=2)
    # output_scale = 10**z
    # z = random.uniform(key2, minval=-2, maxval=0)
    # length_scale = 10**z
    # gp_params = (output_scale, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)[:,None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(key, (N,)))
    # Create a callable interpolation function  
    u_fn = lambda x, t: jnp.interp(t, X.flatten(), gp_sample)
    # Input sensor locations and measurements
    x = jnp.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)
    # Output sensor locations and measurements
    y = jnp.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)
    return u, y, s 

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, m, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_training_data(key, m, P))
    u_train, y_train, s_train = vmap(gen_fn)(keys)
    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train

# Geneate test data corresponding to N input sample
def generate_test_data(key, N, m, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_test_data(key, m, P))
    u, y, s = vmap(gen_fn)(keys)
    config.update("jax_enable_x64", False)
    return u, y, s

TRAINING_ITERATIONS = 50000
P = 100
m = 1000
num_train = 1000
num_test  = 1000
training_batch_size = 100
du = 1
dy = 1
ds = 1
n_hat  = 100
Nx = P
index = 9
length_scale = 0.9
H_y = 2
H_u = 2

# Create the dataset 
key_train = random.PRNGKey(0)
U_train, y_train, s_train = generate_training_data(key_train, num_train, m, Nx)
key_test = random.PRNGKey(12345)
U_test, y_test, s_test = generate_test_data(key_test, num_test, m, Nx)

# Make all array to be jax numpy format
y_train = jnp.asarray(y_train)
s_train = jnp.asarray(s_train)
U_train = jnp.asarray(U_train)

y_test = jnp.asarray(y_test)
s_test = jnp.asarray(s_test)
U_test = jnp.asarray(U_test)

U_train = jnp.reshape(U_train,(num_test,m,du))
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
s_train_std  = jnp.std(s_train,axis=0) + 1e-03

s_train = (s_train - s_train_mean)/s_train_std

# Perform the scattering transform for the inputs yh
train_dataset = DataGenerator(U_train, y_train, s_train, training_batch_size)
train_dataset = iter(train_dataset)

test_dataset = DataGenerator(U_test, y_test, s_test, training_batch_size)
test_dataset = iter(test_dataset)

branch_layers = [m*(du*H_u+du), 512, 512, ds*n_hat]
trunk_layers  = [H_y*dy + dy, 512, 512, ds*n_hat]

model = DON(branch_layers, trunk_layers, m=m, P=P, mn=s_train_mean,  std=s_train_std)

model.count_params(model.get_params(model.opt_state))

start_time = timeit.default_timer()
model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

params = model.get_params(model.opt_state)

uCNN_test = model.predictT(params, (U_test, y_test))
test_error_u = []
for i in range(0,num_train):
    test_error_u.append(norm(s_test[i,:,0]- uCNN_test[i,:,0],2)/norm(s_test[i,:,0],2))
print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))

uCNN_train = model.predict(params, (U_train, y_train))
train_error_u = []
for i in range(0,num_test):
    train_error_u.append(norm(s_train[i,:,0]- uCNN_train[i,:,0],2)/norm(s_train[i,:,0],2))
print("The average train u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_u),np.std(train_error_u),np.min(train_error_u),np.max(train_error_u)))


np.savez_compressed("/scratch/gkissas/Antiderivative/DON/Antiderivative_test_P%d_m%d_ls%f_id%d_DON.npz"%(P,m,length_scale,index), uCNN_super_all_test=uCNN_test, U_test=U_test, s_all_test=s_test, test_error=test_error_u)