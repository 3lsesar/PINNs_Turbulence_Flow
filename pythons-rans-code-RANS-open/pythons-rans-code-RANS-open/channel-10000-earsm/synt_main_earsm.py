import scipy.io as sio
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 22})
plt.interactive(True)

visc=1./5200.

data = np.loadtxt('y_u_k_ed_dudy.txt')

y = data[:,0]
u = data[:,1]
rk_turb = data[:,2]
ed = data[:,3]
dudy = data[:,4]

nj = len(y)

uu=np.zeros(nj)
vv=np.zeros(nj)
ww=np.zeros(nj)
uv=np.zeros(nj)
p1=np.zeros(nj)
p2=np.zeros(nj)
un=np.zeros(nj)
beta1=np.zeros(nj)
beta4=np.zeros(nj)

# lowe Re EARSM by
# S. Wallin and A. V. Johansson,
#"A New Explicit Algebraic Reynolds Stress Model for Incompressible
#and Compressible Turbulent Flows", JFM,
#Vol. 403, pp. 89-132, 2000


for j in range (0,nj-1):

    diss=ed[j]
    rk=rk_turb[j]
    ttau=rk/diss

    om12=ttau*0.5*dudy[j]
    om21=-om12
    om22=0.
    om11=0.
    s11=0.
    s12=ttau*0.5*dudy[j]
    s21=s12
    s22=0.
    s33=0.
    vor=(-2.*om12**2)
    str1=(s11**2+s12**2+s21**2+s22**2)

      
    cpr1=9./4.*(1.8-1.)
    p1[j]=(1./27.*cpr1**2+9./20.*str1-2./3.*vor)*cpr1
    p2[j]=p1[j]**2-(1./9.*cpr1**2+9./10.*str1+2./3.*vor)**3
      
    if p2[j] >  0:
       if p1[j]-p2[j]**0.5 >= 0:
          sigg=1.
       else:
          sigg=-1.
       un[j]=cpr1/3.+(p1[j]+p2[j]**0.5)**(1./3.)+sigg*(abs(p1[j]-p2[j]**0.5))**(1./3.)
    else:
       un[j]=cpr1/3.+2.*(p1[j]**2-p2[j])**(1./6.)*np.cos(1./3.*np.arccos(p1[j]/(np.sqrt(p1[j]**2-p2[j]))))
      
    const=6./5.
    beta1[j]=-const*un[j]/(un[j]**2-2.*vor)
    beta4[j]=beta1[j]/un[j]

    uu[j]=2./3.*rk+rk*beta1[j]*s11+rk*beta4[j]*(s12*om21-om12*s21)
    vv[j]=2./3.*rk+rk*beta1[j]*s22+rk*beta4[j]*(s21*om12-om21*s12)
    ww[j]=2./3.*rk
    uv[j]=rk*(beta1[j]*s12+beta4[j]*(s11*om12-om12*s22))

np.savetxt('y_uu_vv_ww.txt', np.c_[y,uu, vv, ww, uv])
