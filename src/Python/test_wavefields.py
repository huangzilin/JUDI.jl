from __future__ import print_function
import numpy as np
import psutil, os, gc
from numpy.random import randint
from sympy import solve, cos, sin
from sympy import Function as fint
from devito.logger import set_log_level
from devito import Eq, Function, TimeFunction, Dimension, Operator, clear_cache
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model
from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
import matplotlib.pyplot as plt
from JAcoustic_codegen import forward_modeling, adjoint_born, adjoint_modeling, forward_born

# Model
shape = (101, 101)
spacing = (10., 10.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 2.5
v0 = np.empty(shape, dtype=np.float32)
v0[:, :] = 1.5
model = Model(shape=shape, origin=origin, spacing=spacing, vp=v)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=v0)

# Time axis
t0 = 0.
tn = 1000.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time = np.linspace(t0,tn,nt)

# Source
f0 = 0.010
src = RickerSource(name='src', grid=model.grid, f0=f0, time=time)
src.coordinates.data[0,:] = np.array(model.domain_size) * 0.5
src.coordinates.data[0,-1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=101, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
rec_t.coordinates.data[:, 1] = 20.

# Observed data
d_obs, utrue = forward_modeling(model, src.coordinates.data, src.data, rec_t.coordinates.data)

##################################################################################################################

# Receiver for predicted data
rec = Receiver(name='rec', grid=model0.grid, npoint=101, ntime=nt)
rec.coordinates.data[:, 0] = np.linspace(0, model0.domain_size[0], num=101)
rec.coordinates.data[:, 1] = 20.

# Data in, data out
#d_pred, _ = forward_modeling(model0, src.coordinates.data, src.data, rec.coordinates.data, save=False, dt=dt)
#q_ad = adjoint_modeling(model0, src.coordinates.data, rec.coordinates.data, d_pred.data, dt=dt)

# Data in, wavefield out
u = forward_modeling(model0, src.coordinates.data, src.data, None, dt=dt)
v = adjoint_modeling(model0, None, rec.coordinates.data, d_obs.data, dt=dt)

# Wavefield in, wavefield out
ufull = forward_modeling(model0, None, v, None, dt=dt)
vfull = adjoint_modeling(model0, None, None, u, dt=dt)

