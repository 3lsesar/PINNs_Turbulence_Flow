import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})

plt.interactive(True)

viscos=1/100000


# makes sure figures are updated when using ipython
display.clear_output(wait=True)

datax= np.loadtxt("x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt("y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

y=yp2d[1,:]
x=xp2d[:,1]

u2d=np.load('u2d_saved.npy')
p2d=np.load('p2d_saved.npy')
v2d=np.load('v2d_saved.npy')
k2d=np.load('k2d_saved.npy')
om2d=np.load('om2d_saved.npy')
vis2d=np.load('vis2d_saved.npy')

# compute ustar
# north
dy=y[0] # symmetric mesh
ustar_n=(viscos*(1-u2d[:,-1])/dy)**0.5
ustar_s=(viscos*abs(u2d[:,0])/dy)**0.5

yplus_n = ustar_n*dy/viscos
yplus_s = ustar_s*dy/viscos

dx=x[0] # symmetric mesh
ustar_e=(viscos*abs(v2d[-1,:])/dx)**0.5
ustar_w=(viscos*abs(v2d[0,:])/dx)**0.5

yplus_w = ustar_w*dx/viscos
yplus_e = ustar_e*dx/viscos



# load u vs iteration
u_iter= np.loadtxt("u-iter-history.dat")

iter=u_iter[:,0]
u5=u_iter[:,1]
u10=u_iter[:,2]
u20=u_iter[:,3]
u30=u_iter[:,4]
u40=u_iter[:,5]
u50=u_iter[:,6]


########################################## plot y+
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus_s,x,'b-',label='south wall')
plt.plot(yplus_s,x,'r--',label='north wall')
plt.plot(yplus_w,y,'k-.',label='west wall')
plt.plot(yplus_e,y,'r-',label='east wall')
plt.ylabel(r"$x^+, \quad y^+$")
plt.xlabel(r"$x, \quad y$")
plt.axis([0,2,0,1])
plt.legend(loc="upper right",prop=dict(size=18))
plt.savefig('yplus.png')

########################################## plot u along x=0.5
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i=int(ni/2)-1
u05=0.5*(u2d[i,:]+u2d[i+1,:])
plt.plot(u05,y,'b-')
plt.ylabel("$y$")
plt.xlabel("$U$")
plt.axis([-0.4,1,0,1])
plt.savefig('u_lid.png')


########################################## plot vis along x=0.5
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i=int(ni/2)-1
vis05=0.5*(vis2d[i,:]+vis2d[i+1,:])
plt.plot(vis05/viscos,y,'b-')
plt.ylabel("$y$")
plt.xlabel(r"$\nu_{tpt}/\nu$")
plt.axis([0,75,0,1])
plt.savefig('vis_lid.png')


########################################## plot u vs. iteration 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(iter,u5,'b-')
plt.plot(iter,u10,'r-')
plt.plot(iter,u20,'k-')
plt.plot(iter,u30,'b--')
plt.plot(iter,u40,'r--')
plt.plot(iter,u50,'k--')
plt.ylabel("$U$")
plt.xlabel("iteration")
plt.savefig('u_iteration.png')


