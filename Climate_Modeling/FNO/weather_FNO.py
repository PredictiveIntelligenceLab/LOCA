"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
from numpy.linalg import norm
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

torch.manual_seed(0)
np.random.seed(0)

import timeit


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


################################################################
# configs
################################################################
ntrain = 1825
ntest = 1825

batch_size = 73
learning_rate = 0.001

epochs = 400
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 1
Nx = 72
Ny = 72
h = Nx
s = h
P = 144

################################################################
# load data and data normalization
################################################################

d = np.load("../Data/weather_dataset.npz")
U_train = d["U_train"][:ntrain,:].reshape(ntrain,Nx,Ny)
S_train = d["S_train"][:ntrain,:].reshape(ntrain,Nx,Ny)/1000.
CX = d["X_train"]
CY = d["Y_train"]

d = np.load("../Data/weather_dataset.npz")
U_test = d["U_train"][ntest:,:].reshape(ntrain,Nx,Ny)
S_test = d["S_train"][ntest:,:].reshape(ntrain,Nx,Ny)/1000.
CX = d["X_train"]
CY = d["Y_train"]

dtype_double = torch.FloatTensor
cdtype_double = torch.cuda.DoubleTensor
U_train = torch.from_numpy(np.asarray(U_train)).type(dtype_double)
S_train = torch.from_numpy(np.asarray(S_train)).type(dtype_double)

U_test = torch.from_numpy(np.asarray(U_test)).type(dtype_double)
S_test = torch.from_numpy(np.asarray(S_test)).type(dtype_double)

x_train = U_train
y_train = S_train

x_test = U_test
y_test = S_test

grids = []
lontest = np.linspace(0,355,num=Nx)/360
lattest = (np.linspace(90,-87.5,num=Ny) + 90.)/180.

grids.append(lontest)
grids.append(lattest)
grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
grid = grid.reshape(1,s,s,2)
grid = torch.tensor(grid, dtype=torch.float)
x_train = torch.cat([x_train.reshape(ntrain,s,s,1), grid.repeat(ntrain,1,1,1)], dim=3)
x_test = torch.cat([x_test.reshape(ntest,s,s,1), grid.repeat(ntest,1,1,1)], dim=3)

ind_train = torch.randint(s*s, (ntrain, P))
ind_test = torch.randint(s*s, (ntest, P))
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, ind_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, ind_test), batch_size=batch_size, shuffle=True)

################################################################
# training and evaluation
################################################################

batch_ind = torch.arange(batch_size).reshape(-1, 1).repeat(1, P)
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
start_time = timeit.default_timer()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y, idx in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s*s)
        y = y.reshape(batch_size, s*s)
        y = y[batch_ind, idx]
        out = out[batch_ind, idx]

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y, idx in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s*s)
            y = y.reshape(batch_size, s*s,1)
            y = y[batch_ind, idx]
            out = out[batch_ind, idx]

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_l2/= ntrain
    test_l2 /= ntest

    t2 = default_timer()

    print(ep, t2-t1, train_l2, test_l2)#, np.mean(error_total))

elapsed = timeit.default_timer() - start_time
print("The training wall-clock time is seconds is equal to %f seconds"%elapsed)

pred_torch = torch.zeros(S_test.shape)
baseline_torch = torch.zeros(S_test.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
test_error_u = []
test_error_u_np = []
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).reshape(1, s, s)
        pred_torch[index,:,:] = out[:,:,:]
        baseline_torch[index,:,:] = y[:,:,:]

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_error_u.append(test_l2)
        test_error_u_np.append(np.linalg.norm(out.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1])- y.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[1]),2)/np.linalg.norm(out.cpu().numpy().reshape(S_test.shape[1]*S_test.shape[2]),2))
        index = index + 1

print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u),np.std(test_error_u),np.min(test_error_u),np.max(test_error_u)))
print("The average test u error is %e the standard deviation is %e the min error is %e and the max error is %e"%(np.mean(test_error_u_np),np.std(test_error_u_np),np.min(test_error_u_np),np.max(test_error_u_np)))