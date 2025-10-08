import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})
plt.interactive(True)


viscos=3.57E-5
datax= np.loadtxt("x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt("y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

y2d=np.zeros((ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))
x2d=np.zeros((ni+1,nj+1))
x2d=np.reshape(x,(ni+1,nj+1))

xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])
y=yp2d[1,:]
x=xp2d[:,1]
# make ii 2D

itstep,nk,dz=np.load('itstep.npy')
u2d=np.load('u_averaged.npy')/itstep
v2d=np.load('v_averaged.npy')/itstep
k2d=np.load('k_averaged.npy')/itstep
eps2d=np.load('eps_averaged.npy')/itstep
om2d=np.load('om_averaged.npy')/itstep
vis2d=np.load('vis_averaged.npy')/itstep



i=ni-10
u=u2d[i,:]
v=v2d[i,:]
k=k2d[i,:]
om=om2d[i,:]
vis=vis2d[i,:]

vist=vis-viscos
dudy=np.gradient(u,y)
uv=-vist*dudy

np.savetxt('y_u_v_k_om_uv_re-theta-2500.txt', np.c_[y,u,v,k,om,uv])

