import jax
import jax.numpy as jnp
from jax.example_libraries.stax import Dense, Gelu
from jax.example_libraries import stax
from jax.example_libraries import optimizers
from jax.example_libraries.ode import odeint
from jax.config import config

import timeit
import numpy as np
from jax.numpy.linalg import norm

from jax import random, grad, vmap, jit
from functools import partial 

from torch.utils import data
from jax.flatten_util import ravel_pytree

from tqdm import trange

import itertools

from kymatio.numpy import Scattering1D

from numpy.polynomial.legendre import leggauss
import os 

def pairwise_distances(dist,**arg):
    return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))

def euclid_distance(x,y):
    XX=jnp.dot(x,x)
    YY=jnp.dot(y,y)
    XY=jnp.dot(x,y)
    return XX+YY-2*XY

class DataGenerator(data.Dataset):
    def __init__(self, inputsxu, y, s, z, w,
                 batch_size=100, rng_key=random.PRNGKey(1234)):
        'Initialization'
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

def scatteringTransform(sig, l=100, m=100, training_batch_size = 100):
    J = 4
    Q = 8
    T = sig.shape[1]
    scattering = Scattering1D(J, T, Q)
    sig = np.asarray(sig)
    sctcoef = np.zeros((training_batch_size, 1550, 1))
    for i in range(0,training_batch_size):
        sctcoef[i,:,:] = scattering(sig[i,:,0]).flatten()[:,None]
    return sctcoef

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
        diff_norms = jnp.linalg.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = jnp.linalg.norm(y.reshape(num_examples,-1), self.p, 1)
        return jnp.mean(diff_norms/y_norms)

    def __call__(self, x, y):
        return self.rel(x, y)


class LOCA:
    def __init__(self, q_layers, g_layers, v_layers, jac_det=None):    
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

        # RBF kernel parameters
        beta = [1.]
        gamma = [1.]
        # Model parameters
        params = (beta, gamma,q_params, g_params, v_params)

        self.opt_init,self.opt_update,self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3, 
                                                                      decay_steps=100, 
                                                                      decay_rate=0.99))
        self.opt_state = self.opt_init(params)
        self.itercount = itertools.count()
        self.loss_log = []

        self.l2loss = LpLoss(size_average=False)

        self.jac_det = jac_det
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

    def LOCA_net(self, params, inputs, ds=1):
        beta, gamma, q_params, g_params, v_params = params
        inputsxu, inputsy, inputsz, w = inputs
        inputsy  = self.q_apply(q_params,inputsy)
        inputsz  = self.q_apply(q_params,inputsz)

        d = self.vdistance_function(inputsz, inputsz)
        K =  beta[0]*jnp.exp(-gamma[0]*d)
        Kzz =  jnp.sqrt(self.jac_det*jnp.einsum("ijk,ikl->ijl",K,w))

        d = self.vdistance_function(inputsy, inputsz)
        K =  beta[0]*jnp.exp(-gamma[0]*d)
        Kyz =  jnp.sqrt(self.jac_det*jnp.einsum("ijk,ikl->ijl",K,w))
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
                loss_test  = self.loss(params, test_batch)

                errorTrain = self.L2error(params, train_batch)
                errorTest  = self.L2error(params, test_batch)

                self.loss_log.append(loss_train)

                pbar.set_postfix({'Training loss': loss_train, 
                                  'Testing loss' : loss_test,
                                  'Test error':    errorTest,
                                  'Train error':   errorTrain})

    @partial(jit, static_argnums=(0,))
    def predict(self, params, inputs):
        s_pred = self.LOCA_net(params,inputs)
        return s_pred

    def count_params(self, params):
        params_flat, _ = ravel_pytree(params)
        print("The number of model parameters is:",params_flat.shape[0])

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
    length_scale = 0.3
    gp_params = (1.0, length_scale)
    key1= random.split(key,num=2)
    z = random.uniform(key1[0], minval=-2, maxval=2)
    output_scale = 10**z
    z = random.uniform(key1[1], minval=-2, maxval=0)
    length_scale = 10**z
    gp_params = (output_scale, length_scale)
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
    y_train = random.uniform(key, (P,)).sort() 
    s_train = odeint(u_fn, 0.0, jnp.hstack((0.0, y_train)))[1:] # JAX has a bug and always returns s(0), so add a dummy entry to y and return s[1:]
    return u, y_train, s_train, length_scale

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, m=100, P=100):
    # Sample GP prior at a fine grid
    N = 512
    length_scale = 0.3
    gp_params = (1.0, length_scale)
    key1, key2 = random.split(key,num=2)
    z = random.uniform(key1, minval=-2, maxval=2)
    output_scale = 10**z
    z = random.uniform(key2, minval=-2, maxval=0)
    length_scale = 10**z
    gp_params = (output_scale, length_scale)
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
    return u, y, s, length_scale

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, m, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_training_data(key, m, P))
    u_train, y_train, s_train, l_train = vmap(gen_fn)(keys)
    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train, l_train

# Geneate test data corresponding to N input sample
def generate_test_data(key, N, m, P):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_test_data(key, m, P))
    u, y, s, l = vmap(gen_fn)(keys)
    config.update("jax_enable_x64", False)
    return u, y, s, l

TRAINING_ITERATIONS = 50000
P = 100
m = 500
L = 1
T = 1
N_hat = 1
num_train = 1000
num_test  = 1000
training_batch_size = 100
du = 1
dy = 1
ds = 1
n_hat  = 100
l  = 100
Nx = P
H = 10
index = 0
length_scale = 1.1

# Number of GLL quadrature points, coordinates and weights
polypoints = 100
z, w = leggauss(polypoints)
lb = np.array([0.0])
ub = np.array([1.0])

# Map [-1,1] -> [0,1]
z = 0.5*(ub - lb)*(z + 1.0) + lb
jac_det = 0.5*(ub-lb)

# Reshape both weights and coordinates. We need them to have shape: (num_train, N, dy)
z = np.tile(np.expand_dims(z,0),(num_train,1))[:,:,None]
w = np.tile(np.expand_dims(w,0),(num_train,1))[:,:,None]

# Create the dataset 
key_train = random.PRNGKey(0)
U_train, y_train, s_train, l_train = generate_training_data(key_train, num_train, m, Nx)
key_test = random.PRNGKey(12345)
U_test, y_test, s_test, l_test = generate_test_data(key_test, num_test, m, Nx)

# Make all array to be jax numpy format
y_train = jnp.asarray(y_train)
s_train = jnp.asarray(s_train)

y_test = jnp.asarray(y_test)
s_test = jnp.asarray(s_test)

z = jnp.asarray(z)
w = jnp.asarray(w)

U_train = np.reshape(U_train,(num_test,m,du))
y_train = jnp.reshape(y_train,(num_train,P,dy))
s_train = jnp.reshape(s_train,(num_train,P,ds))

U_test = np.reshape(U_test,(num_test,m,du))
y_test = jnp.reshape(y_test,(num_test,P,dy))
s_test = jnp.reshape(s_test,(num_test,P,ds))

z = jnp.reshape(z,(num_test,polypoints,dy))
w = jnp.reshape(w,(num_test,polypoints,dy))

plot=False
if plot == True:
    import matplotlib.pyplot as plt
    pltN = 10
    for i in range(0,pltN-1):
        plt.plot(y_train[i,:,0], s_train[i,:,0], 'r-')
        plt.plot(y_test[i,:,0], s_test[i,:,0], 'b-')

    plt.plot(y_train[pltN,:,0], s_train[pltN,:,0], 'r-', label="Training output")
    plt.plot(y_test[pltN,:,0], s_test[pltN,:,0], 'b-', label="Testing output")
    plt.legend()
    plt.show()

    x = jnp.linspace(0,1,num=m)
    pltN = 10
    for i in range(0,pltN-1):
        plt.plot(x, np.asarray(U_train)[i,:,0], 'y-')
        plt.plot(x, np.asarray(U_test)[i,:,0], 'g-')

    plt.plot(x, np.asarray(U_train)[pltN,:,0], 'y-', label="Training input")
    plt.plot(x, np.asarray(U_test)[pltN,:,0], 'g-', label="Testing input")
    plt.legend()
    plt.show()

# Positionally encode y and z
y_train_pos = y_train

pos_encodingy  = PositionalEncodingY(y_train,int(y_train.shape[1]*y_train.shape[2]), max_len = P, H=H)
y_train  = pos_encodingy.forward(y_train)
del pos_encodingy

pos_encodingy  = PositionalEncodingY(z,int(z.shape[1]*z.shape[2]), max_len = polypoints, H=H)
z  = pos_encodingy.forward(z)
del pos_encodingy

pos_encodingyt = PositionalEncodingY(y_test,int(y_test.shape[1]*y_test.shape[2]), max_len = P, H=H)
y_test   = pos_encodingyt.forward(y_test)
del pos_encodingyt

start_time = timeit.default_timer()
inputs_trainxu = jnp.asarray(scatteringTransform(U_train, l=l, m=m, training_batch_size=num_train))
inputs_testxu  = jnp.asarray(scatteringTransform(U_test , l=l, m=m, training_batch_size=num_test))
elapsed = timeit.default_timer() - start_time
print("The wall-clock time for for loop is seconds is equal to %f seconds"%elapsed)
print(inputs_trainxu.shape, inputs_testxu.shape)

train_dataset = DataGenerator(inputs_trainxu, y_train, s_train, z, w, training_batch_size)
train_dataset = iter(train_dataset)

test_dataset = DataGenerator(inputs_testxu, y_test, s_test, z, w, training_batch_size)
test_dataset = iter(test_dataset)

q_layers = [L*dy+H*dy, 100, 100, l]
v_layers = [1550*du, 500, ds*n_hat]
g_layers  = [l, 100, 100, ds*n_hat]

model = LOCA(q_layers, g_layers, v_layers, jac_det=jac_det)

model.count_params(model.get_params(model.opt_state))

start_time = timeit.default_timer()
model.train(train_dataset, test_dataset, nIter=TRAINING_ITERATIONS)
elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

params = model.get_params(model.opt_state)

uCNN_test = model.predict(params, (inputs_testxu,y_test, z, w))
test_error_u = []
for i in range(0,s_test.shape[0]):
    test_error_u.append(jnp.linalg.norm(s_test[i,:,-1] - uCNN_test[i,:,-1], 2)/jnp.linalg.norm(s_test[i,:,-1], 2))
print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))

uCNN_train = model.predict(params, (inputs_trainxu, y_train, z, w))

train_error_u = [] 
for i in range(0,s_test.shape[0]):
    train_error_u.append(jnp.linalg.norm(s_train[i,:,-1] - uCNN_train[i,:,-1], 2)/jnp.linalg.norm(s_train[i,:,-1], 2))
print("The average train u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(train_error_u),np.std(train_error_u),np.min(train_error_u),np.max(train_error_u)))
