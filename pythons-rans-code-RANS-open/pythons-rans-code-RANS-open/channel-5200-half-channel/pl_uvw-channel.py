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

viscos=1/5200


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

y=yp2d[0,:]


u2d=np.load('u2d_saved.npy')
p2d=np.load('p2d_saved.npy')
v2d=np.load('v2d_saved.npy')
k2d=np.load('k2d_saved.npy')
om2d=np.load('om2d_saved.npy')
vis2d=np.load('vis2d_saved.npy')

# average in x direction
u=np.mean(u2d,axis=0)
v=np.mean(v2d,axis=0)
k=np.mean(k2d,axis=0)
om=np.mean(om2d,axis=0)
vis=np.mean(vis2d,axis=0)

ustar=(viscos*u[1]/y[1])**0.5
yplus=y*ustar/viscos

vist=vis-viscos
dudy=np.gradient(u,y)
uv=-vist*dudy

np.savetxt('y_u_k_om_uv_5200-RANS-half-channel.txt', np.c_[y,u,k,om,uv])

DNS_mean=np.genfromtxt("LM_Channel_5200_mean_prof.dat",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];

DNS_stress=np.genfromtxt("LM_Channel_5200_vel_fluc_prof.dat",comments="%")
u2_DNS=DNS_stress[:,2];
v2_DNS=DNS_stress[:,3];
w2_DNS=DNS_stress[:,4];
uv_DNS=DNS_stress[:,5];

k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

dudy_DNS = np.gradient(u_DNS,y_DNS)

y_kom = yp2d[0,:]

k_DNS_on_kom_grid = np.interp(y_kom, y_DNS, k_DNS)

vist_DNS = abs(uv_DNS/dudy_DNS)


        #y/delta                    y^+                   Production          Turbulent_Transport        Viscous_Transport       Pressure_Strain         Pressure_Transport        Viscous_Dissipation           Balance
DNS_k_terms=np.genfromtxt("LM_Channel_5200_RSTE_k_prof.dat",comments="%")
diff_DNS=DNS_k_terms[:,3]
diss_DNS=DNS_k_terms[:,7]
diff_DNS_visc =DNS_k_terms[:,4]
Pk_DNS=DNS_k_terms[:,2]
diff_DNS_visc = diff_DNS_visc/viscos
diss_DNS=diss_DNS/viscos
diff_DNS=diff_DNS/viscos
Pk_DNS=Pk_DNS/viscos

k_terms =  np.loadtxt('terms_k_eq.txt')
#       np.c_[prod[0,:],diss[0,:],diff[0,:],diff_turb[0,:],diff_lam[0,:]])
dy = np.diff(y2d[0,:])
dx = x2d[1,0]-x2d[0,0]
Pk = k_terms[:,0]
diss_k = k_terms[:,1]
diff_k = k_terms[:,2]/dx/dy
diff_k_turb = k_terms[:,3]/dx/dy
diff_k_lam = k_terms[:,4]/dx/dy

# limit the ratio of k_DNS/k_kom. 
k_ratio = np.minimum(k_DNS_on_kom_grid/k,5)
# new k_DNS
k_DNS_on_kom_grid = k_ratio*k

vist_DNS_on_kom_grid = np.interp(y_kom, y_DNS, vist_DNS)

omega_DNS_from_vist =k_DNS/vist_DNS
omega_DNS_from_vist_on_kom_grid = k_DNS_on_kom_grid/vist


# find equi.distant DNS cells in log-scale
xx=0.
jDNS=[1]*40
for i in range (0,40):
   i1 = (np.abs(10.**xx-yplus_DNS)).argmin()
   jDNS[i]=int(i1)
   xx=xx+0.2

########################################## U 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.semilogx(yplus,u,'b-')
plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 28])
plt.savefig('u_log_5200-channel.png',bbox_inches='tight')


########################################## uv 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
# compute shear stress
plt.plot(yplus,uv,'b-')
plt.plot(yplus_DNS[jDNS],uv_DNS[jDNS],'bo')
plt.ylabel(r"$\overline{u'v'}$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, -1, 1])
plt.savefig('uv_5200-channel.png',bbox_inches='tight')


########################################## k 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,k,'b-')
plt.plot(yplus_DNS,k_DNS,'r--')
plt.ylabel(r"$k^+$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 7.5])
plt.savefig('k_5200-channel.png',bbox_inches='tight')

########################################## vis 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,vis/viscos,'b-')
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y^+$")
plt.axis([1, 5200, 0, 700])
plt.savefig('vis_5200-channel.png',bbox_inches='tight')



########################################## k  balance zoom from pyCALC
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
# compute shear stress
plt.plot(yplus,Pk,'b-',label= '$P^k$')
plt.plot(yplus,-diss_k,'r-',label= r'$\varepsilon$')
plt.plot(yplus,diff_k,'k-',label= r'$D^k$')
plt.plot(yplus_DNS,Pk_DNS,'b--')
plt.plot(yplus_DNS,-diss_DNS,'r--')
plt.plot(yplus_DNS,diff_DNS,'k--')
plt.legend(loc='best',fontsize=20)
plt.ylabel('term in $k$ eq.')
plt.xlabel("$y^+$")
plt.axis([0, 100, -1300, 1300])
plt.savefig('k-balance_5200-channel.png',bbox_inches='tight')

########################################## k  diff  + diss - diff_visc
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(yplus[1:],-diss_k[1:]+diff_k_lam[1:],'b-',label=r'$k-\omega:$ $ D^{k,\nu} -  \varepsilon$')
plt.plot(yplus[1:],-diss_k[1:],'b--',label=r'$k-\omega:$ $-  \varepsilon$')
plt.plot(yplus[1:],-0.09*k_DNS_on_kom_grid[1:]*omega_DNS_from_vist_on_kom_grid[1:],'g--',label=r'$-0.09 k_{DNS} \omega_{DNS}$')
diff_DNS_visc_kom = np.interp(yp2d[0,:], y_DNS, diff_DNS_visc)
plt.plot(yplus_DNS,-diss_DNS+diff_DNS_visc,'r--',label=r'DNS: $ D^{k,\nu} - \varepsilon$')
plt.legend(loc='best',fontsize=20)
plt.xlabel("$y^+$")
plt.axis([0, 200, -1550, 10])
plt.savefig('k-diss-visc-diff-channel.png',bbox_inches='tight')

########################################## omega  balance zoom from pyCALC
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

dy = np.diff(y2d[0,:])
dx = x2d[1,0]-x2d[0,0]
om_terms = np.loadtxt('terms_omega_eq.txt')
prod_py = om_terms[:,0]*dx*dy
Psi_py = om_terms[:,1]*dx*dy
diff_py = om_terms[:,2]
source = om_terms[:,3]
su_term = om_terms[:,4]
sp_term = om_terms[:,5]
diff_turb_py = om_terms[:,6]
diff_visc_py  = diff_py-diff_turb_py
diff_lam  =  om_terms[:,7]
prod_dy = prod_py/dx/dy
diff_turb = diff_turb_py/dx/dy
diff_visc_py = diff_visc_py/dx/dy
Psi_py = Psi_py/dx/dy

j=1
plt.plot(yplus[j:],prod_py[j:],'b-',label=r'$P_\omega$')
plt.plot(yplus[j:],diff_turb_py[j:],'k-',label='turb.diff')
plt.plot(yplus[j:],diff_visc_py[j:],'g-',label='visc.diff')
plt.plot(yplus[j:],diff_visc_py[j:],'go')
plt.plot(yplus[j:],-Psi_py[j:],'r-',label='-Psi')

imb_py = (prod_py + diff_py - Psi_py)/prod_py
imb_source = (source + diff_py)/prod_py
plt.axis([0, 50, -1e6, 1e6])
plt.ylabel(r"$\omega$ eq.")
plt.xlabel("$y^+$")
plt.title("from pyCALC")
plt.legend(loc="upper right",prop=dict(size=14))
plt.savefig('balance-omega-eq-5200-channel-zoom-from-pyCALC.png',bbox_inches='tight')


########################################## omega  balance zoom from pyCALC zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

j = 10

plt.plot(yplus[j:],prod_py[j:],'b-',label=r'$P_\omega$')
plt.plot(yplus[j:],diff_turb_py[j:],'k-',label='turb.diff')
plt.plot(yplus[j:],diff_visc_py[j:],'g-',label='visc.diff')
plt.plot(yplus[j:],diff_visc_py[j:],'go')
plt.plot(yplus[j:],-Psi_py[j:],'r-',label='-Psi')

plt.axis([0, 20, -5e8, 5e8])
plt.ylabel(r"$\omega$ eq.")
plt.xlabel("$y^+$")
plt.title("from pyCALC")
plt.legend(loc="upper right",prop=dict(size=14))
plt.savefig('balance-omega-eq-5200-channel-zoom-from-pyCALC-zoom.png',bbox_inches='tight')


