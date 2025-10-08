import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
plt.rcParams.update({'font.size': 22})


# read data file
tec=np.genfromtxt("tec.dat", dtype=None,comments="%")

#text='VARIABLES = X Y P U V u2 v2 w2 uv mu_sgs prod'

x=tec[:,0]
y=tec[:,1]
p=tec[:,2]
u=tec[:,3]
v=tec[:,4]
uu=tec[:,5]
vv=tec[:,6]
ww=tec[:,7]
uv=tec[:,8]
k=0.5*(uu+vv+ww)

if max(y) == 1.:
   ni=170
   nj=194
   nu=1./10000.
else:
   nu=1./10595.
   if max(x) > 8.:
     nj=162
     ni=162
   else:
     ni=402
     nj=162

viscos=nu

u2d=np.reshape(u,(nj,ni))
v2d=np.reshape(v,(nj,ni))
p2d=np.reshape(p,(nj,ni))
x2d=np.reshape(x,(nj,ni))
y2d=np.reshape(y,(nj,ni))
uu2d=np.reshape(uu,(nj,ni)) #=mean{v'_1v'_1}
uv2d=np.reshape(uv,(nj,ni)) #=mean{v'_1v'_2}
vv2d=np.reshape(vv,(nj,ni)) #=mean{v'_2v'_2}
k2d=np.reshape(k,(nj,ni))

u2d=np.transpose(u2d)
v2d=np.transpose(v2d)
p2d=np.transpose(p2d)
x2d=np.transpose(x2d)
y2d=np.transpose(y2d)
uu2d=np.transpose(uu2d)
vv2d=np.transpose(vv2d)
uv2d=np.transpose(uv2d)



#************
# plot uv
fig3 = plt.figure("Figure 3")
i=10
plt.plot(uv2d[i,:],y2d[i,:],'b-')
plt.xlabel('$\overline{u^\prime v^\prime}$')
plt.ylabel('y/H')
plt.savefig('uv_python.eps')



