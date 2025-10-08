import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})

plt.interactive(True)
plt.close('all')

viscos=1/100


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

x=xp2d[:,0]
y=yp2d[0,:]


u2d=np.load('u2d_saved.npy')
p2d=np.load('p2d_saved.npy')
v2d=np.load('v2d_saved.npy')
k2d=np.load('k2d_saved.npy')
om2d=np.load('om2d_saved.npy')
vis2d=np.load('vis2d_saved.npy')

dudx2d,dudy2d=np.gradient(u2d,x,y)
uv2d=-vis2d*dudy2d


#load iteration history
u_iter= np.loadtxt("u-iteration.dat")

u5=u_iter[:,1]
u10=u_iter[:,2]
u20=u_iter[:,3]


########################################## U 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(y,u2d[i1,:],'b-',label="$x=2$")
i1 = 2
plt.plot(y,u2d[i1,:],'r-',label="$x=10$")
plt.legend(loc='best',prop=dict(size=13))
plt.ylabel("$U$")
plt.xlabel("$y$")
plt.axis([0, 1, 0, 58])
plt.savefig('u_100-channel.png',bbox_inches='tight')

########################################## uv 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
i1 = 0
plt.plot(y,uv2d[i1,:],'b-',label="$x=2$")
i1 = 2
plt.plot(y,uv2d[i1,:],'r-',label="$x=10$")
plt.legend(loc='best',prop=dict(size=13))
plt.ylabel(r"$\overline{u'v'}$")
plt.xlabel("$y$")
plt.axis([0, 1, -1, 0])
plt.savefig('uv_100-channel.png',bbox_inches='tight')


########################################## u iteration
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(u5,'b-',label="$j=5$")
plt.plot(u10,'r-',label="$j=10$")
plt.plot(u20,'k-',label="$j=15$")
plt.legend(loc='best',prop=dict(size=13))
plt.xlabel('iteration')
plt.ylabel('$U$')
plt.savefig('u_vs_iteration.png',bbox_inches='tight')




