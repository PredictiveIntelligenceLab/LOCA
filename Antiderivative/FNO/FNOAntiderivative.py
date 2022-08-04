"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
from jax import random, vmap, jit

import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.config import config
import argparse

import os

seed = np.random.randint(10000)
torch.manual_seed(seed)
np.random.seed(seed)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

# Define RBF kernel
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs**2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)

# Geneate training data corresponding to one input sample
def generate_one_training_data(key, m=100, P=1, ls=1):
    # Sample GP prior at a fine grid
    N = 512
    # length_scale = ls
    # gp_params = (1.0, length_scale)
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

    # Ijnput sensor locations and measurements
    x = jnp.linspace(0, 1, m)
    u = vmap(u_fn, in_axes=(None,0))(0.0, x)

    # Output sensor locations and measurements
    y = jnp.linspace(0, 1, P)
    s = odeint(u_fn, 0.0, y)
    return u, y, s

# Geneate test data corresponding to one input sample
def generate_one_test_data(key, m=100, P=100, ls =0.1):
    # Sample GP prior at a fine grid
    N = 512
    # length_scale = ls
    # gp_params = (1.0, length_scale)
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
    return u, y, s 

# Geneate training data corresponding to N input sample
def generate_training_data(key, N, m, P, ls):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_training_data(key, m, P, ls))
    u_train, y_train, s_train = vmap(gen_fn)(keys)
    config.update("jax_enable_x64", False)
    return u_train, y_train, s_train

# Geneate test data corresponding to N input sample
def generate_test_data(key, N, m, P, ls):
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    gen_fn = jit(lambda key: generate_one_test_data(key, m, P, ls))
    u, y, s = vmap(gen_fn)(keys)
    config.update("jax_enable_x64", False)
    return u, y, s


################################################################
#  configurations
################################################################
def main(l,id):
    ntrain = 1000
    ntest = 1000
    m = 1000
    Nx = 1000


    h = 1000
    s = h

    batch_size = 100
    learning_rate = 0.001

    epochs = 500
    step_size = 100
    gamma = 0.5

    modes = 32
    width = 100
    length_scale = int(l)
    ind = id
    P = 100

    ################################################################
    # read data
    ################################################################

    # Data is of the shape (number of samples, grid size)
    print('The lengthscale is %.2f'%(0.1*l))
    key_train = random.PRNGKey(0)
    U_train, y_train, s_train = generate_training_data(key_train, ntrain, m, Nx, 0.1*l)
    key_test = random.PRNGKey(12345)
    U_test, y_test, s_test = generate_test_data(key_test, ntest, m, Nx, 0.1)

    dtype_double = torch.FloatTensor
    cdtype_double = torch.cuda.DoubleTensor
    x_train = torch.from_numpy(np.asarray(U_train)).type(dtype_double).reshape(ntrain,s,1)
    y_train = torch.from_numpy(np.asarray(s_train)).type(dtype_double).reshape(ntrain,s,1)

    x_test = torch.from_numpy(np.asarray(U_test)).type(dtype_double).reshape(ntrain,s,1)
    y_test = torch.from_numpy(np.asarray(s_test)).type(dtype_double).reshape(ntrain,s,1)

    ind_train = torch.randint(s, (ntrain, P))
    ind_test = torch.randint(s, (ntest, P))
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, ind_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, ind_test), batch_size=batch_size, shuffle=True)

    ################################################################
    # training and evaluation
    ################################################################

    batch_ind = torch.arange(batch_size).reshape(-1, 1).repeat(1, P)

    # model
    model = FNO1d(modes, width).cuda()
    print(count_params(model))

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y, idx in train_loader:
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            y = y[batch_ind, idx]
            out = out[batch_ind, idx]
            l2 = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            # l2.backward()

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y, idx in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                y = y[batch_ind, idx]
                out = out[batch_ind, idx]

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)

    x_test = torch.from_numpy(np.asarray(U_test)).type(dtype_double).reshape(ntrain,s,1)
    y_test = torch.from_numpy(np.asarray(s_test)).type(dtype_double).reshape(ntrain,s,1)

    pred_torch     = torch.zeros(y_test.shape)
    baseline_torch = torch.zeros(y_test.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    test_error_u = []
    test_error_u_np = []
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()

            out = model(x)
            pred_torch[index] = out
            baseline_torch[index,:,:] = y[:,:,:]

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            test_error_u.append(test_l2)
            test_error_u_np.append(np.linalg.norm(y.view(-1).cpu().numpy()- out.view(-1).cpu().numpy(),2)/np.linalg.norm(y.view(-1).cpu().numpy(),2))
            # print(index, test_l2)
            index = index + 1
    print("The average test u error (no noise) is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
    print("The average test u error (no noise) is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u_np),np.std(test_error_u_np),np.min(test_error_u_np),np.max(test_error_u_np)))

    # in_noise_test  = 0.05*np.random.normal(loc=0.0, scale=1.0, size=(U_test.shape))
    # U_test = U_test + in_noise_test

    x_test = torch.from_numpy(np.asarray(U_test)).type(dtype_double).reshape(ntrain,s,1)
    y_test = torch.from_numpy(np.asarray(s_test)).type(dtype_double).reshape(ntrain,s,1)

    pred_torch     = torch.zeros(y_test.shape)
    baseline_torch = torch.zeros(y_test.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    test_error_u = []
    test_error_u_np = []
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()

            out = model(x)
            pred_torch[index] = out
            baseline_torch[index,:,:] = y[:,:,:]

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            test_error_u.append(test_l2)
            test_error_u_np.append(np.linalg.norm(y.view(-1).cpu().numpy()- out.view(-1).cpu().numpy(),2)/np.linalg.norm(y.view(-1).cpu().numpy(),2))
            # print(index, test_l2)
            index = index + 1
    print("The average test u error (noise) is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
    print("The average test u error (noise) is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u_np),np.std(test_error_u_np),np.min(test_error_u_np),np.max(test_error_u_np)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('l', metavar='l', type=int, nargs='+', help='Lenghtscale of test dataset')
    parser.add_argument('id', metavar='id', type=int, nargs='+', help='Index of the run')

    args = parser.parse_args()
    l = args.l[0]
    id = args.id[0]

    main(l,id)