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

p2d=np.load('p2d_saved.npy')
u2d=np.load('u2d_saved.npy')
k2d=np.load('k2d_saved.npy')
vis2d=np.load('vis2d_saved.npy')
om2d=np.load('om2d_saved.npy')
k_model2d=np.load('k2d_saved.npy')

vis2d=vis2d/viscos

ustar=(viscos*u2d[:,0]/yp2d[1,0])**0.5


#  compute re_theta for boundary layer flow
dx=x[3]-x[2]
re_mom_bl=np.zeros(ni)
for i in range (0,ni-1):
   d_mom=0
   for j in range (1,nj-1):
      up=u2d[i,j]/u2d[i,-1]
      dy=y2d[i,j]-y2d[i,j-1]
      d_mom=d_mom+up*(1.-min(up,1.))*dy

   re_mom_bl[i]=d_mom*u2d[i,-1]/viscos

re_mom_bl[-1]=re_mom_bl[-1-1]

cf_exp_retheta=2*(1./0.384*np.log(re_mom_bl)+4.127)**(-2)



# compute cf
cf=np.zeros(ni)
yplus2d=np.zeros((ni,nj))
for i in range (0,ni-1):
   uwall=u2d[i,0]
   yp=yp2d[i,0]
   ustars=(uwall*viscos/yp)**0.5
   cf[i]=ustars**2/(0.5*u2d[i,-1]**2)
   yplus2d[i,:]=ustar[i]*yp2d[i,:]/viscos

cf[-1]=cf[-2]

# find boundary layer thickness
delta=np.zeros(ni)
for i in range(1,ni-1):
   for j in range(0,nj-2):
      up=u2d[i,j]/u2d[i,-1]
      up1=u2d[i,j+1]/u2d[i,-1]
      if up < 0.99 and up1 > 0.99:
         jj = j
         break
   up=u2d[i,jj]/u2d[i,-1]
   up1=u2d[i,jj+1]/u2d[i,-1]
   delta[i]=y[jj]+(0.99-up)*(y[jj+1]-y[jj])/(up1-up)
#  delta[i]=y[jj]+(0.99-u2d[i,jj])*(y[jj+1]-y[jj])/(u2d[i,jj+1]-u2d[i,jj])
#  delta[i]=y[jj]

delta_inlet=delta[1]


delta[0]=delta[1]
x_delta=np.zeros(ni)
x_delta=x/delta[0]

vel_DNS=np.genfromtxt("vel_2540_dns.prof", dtype=None,comments="%")

#y/\delta_{99}       y+          U+          urms+       vrms+       wrms+       uv+         prms+       pu+         pv+         S(u)        F(u)        dU+/dy+     V+

y_DNS=vel_DNS[:,0]
u_DNS=vel_DNS[:,2]
yplus_DNS=vel_DNS[:,1]
uu_DNS=vel_DNS[:,3]**2
vv_DNS=vel_DNS[:,4]**2
ww_DNS=vel_DNS[:,5]**2
uv_DNS=vel_DNS[:,6]

# find equi.distant DNS cells in log-scale
xx=0.
jDNS=[1]*20
for i in range (0,20):
   i1 = (np.abs(10.**xx-yplus_DNS)).argmin()
   jDNS[i]=int(i1)
   xx=xx+0.2


u_time= np.loadtxt("u-time-history.dat")

u5=u_time[:,1]
u10=u_time[:,2]
u20=u_time[:,3]
u30=u_time[:,4]
u40=u_time[:,5]
u50=u_time[:,6]
u60=u_time[:,7]


########################################## Ustar
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(xp2d[:,1]/delta_inlet,ustar,'b-')
plt.xlabel(r'$x/\delta$')
plt.ylabel(r"$u_\tau$")
plt.savefig('ustar-vs-x.png')


########################################## U 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 1
plt.semilogx(yplus2d[i1,:],u2d[i1,:]/ustar[i1],'b-',label="$x=0$")
plt.semilogx(yplus_DNS,u_DNS,'o',label="DNS")
xx=30
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.semilogx(yplus2d[i1,:],u2d[i1,:]/ustar[i1],'r-',label="$x=30$")
xx=50
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.semilogx(yplus2d[i1,:],u2d[i1,:]/ustar[i1],'k-.',label="$x=50$")
plt.legend(loc="upper left",prop=dict(size=18))
plt.ylabel("$U$")
plt.xlabel("$y^+$")
plt.axis([1, 1000, 0, 28])
plt.savefig('u_log_python.png',bbox_inches='tight')

########################################## vis 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus2d[i1,:],vis2d[i1,:],'b-',label="$x=0$")
xx=30
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yplus2d[i1,:],vis2d[i1,:],'r--',label="$x=30$")
xx=50
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yplus2d[i1,:],vis2d[i1,:],'k-.',label="$x=50$")
plt.legend(loc="upper right",prop=dict(size=18))
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y^+$")
plt.axis([1, 1000, 0,130])
plt.savefig('vis_python.png',bbox_inches='tight')

########################################## vis  vs y
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yp2d[i1,:],vis2d[i1,:],'b-',label="$x=0$")
xx=30
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yp2d[i1,:],vis2d[i1,:],'r--',label="$x=30$")
xx=50
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yp2d[i1,:],vis2d[i1,:],'k-.',label="$x=50$")
plt.legend(loc="upper right",prop=dict(size=18))
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y$")
plt.axis([0, 3, 0,130])
plt.savefig('vis_vs_y_python.png',bbox_inches='tight')

########################################## omega  vs y
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yp2d[i1,:],om2d[i1,:],'b-',label="$x=0$")
xx=30
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yp2d[i1,:],om2d[i1,:],'r--',label="$x=30$")
xx=50
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yp2d[i1,:],om2d[i1,:],'k-.',label="$x=50$")
plt.legend(loc="upper right",prop=dict(size=18))
plt.ylabel(r"$\omega$")
plt.xlabel("$y$")
plt.axis([0, 3, 0,2])
plt.savefig('omega_vs_y_python.png',bbox_inches='tight')


########################################## k
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus2d[i1,:],k_model2d[i1,:]/ustar[i]**2,'b-',label="$x=0$")
xx=30
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yplus2d[i1,:],k_model2d[i1,:]/ustar[i]**2,'r--',label="$x=30$")
xx=50
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yplus2d[i1,:],k_model2d[i1,:]/ustar[i]**2,'k-.',label="$x=50$")
plt.legend(loc="upper right",prop=dict(size=18))
plt.ylabel(r"$k$")
plt.xlabel("$y^+$")
plt.axis([1, 1000, 0,2.5])
plt.savefig('k_model_python.png',bbox_inches='tight')

########################################## k vs y
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yp2d[i1,:],k_model2d[i1,:]/ustar[i]**2,'b-',label="$x=0$")
xx=30
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yp2d[i1,:],k_model2d[i1,:]/ustar[i]**2,'r--',label="$x=30$")
xx=50
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(yp2d[i1,:],k_model2d[i1,:]/ustar[i]**2,'k-.',label="$x=50$")
plt.legend(loc="upper right",prop=dict(size=18))
plt.ylabel(r"$k$")
plt.xlabel("$y$")
plt.axis([0, 3, 0,2.5])
plt.savefig('k_model_python.png',bbox_inches='tight')

########################################## cf
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
plt.subplots_adjust(left=0.25, bottom=0.2, right=0.92, top=0.82, wspace=0, hspace=0.0)
ax1.plot(re_mom_bl,cf,'b-',label="$x=0$")
ax1.plot(re_mom_bl[0::5],cf_exp_retheta[0::5],'bo',label="$x=0$")
ax1.set_ylabel(r"$C_f$")
ax1.set_xlabel(r"$Re_\theta$")
ax1.axis([10,3000,0.003,0.01])
ax2.axis([0,x_delta[-1], 0.003,0.01])

ax2.set_xlabel(r"$x/\delta_{in}$")

plt.savefig('cf_vs_re_mom.png',bbox_inches='tight')

sys.exit()

########################################## u time
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(u5,'b-',label="$j=5$")
plt.plot(u10,'r-',label="$j=10$")
plt.plot(u20,'k-',label="$j=20$")
plt.plot(u30,'b--',label="$j=30$")
plt.plot(u40,'r--',label="$j=40$")
plt.plot(u50,'k--',label="$j=50$")
plt.plot(u60,'g-',label="$j=60$")
#plt.legend()
plt.xlabel('time step')
plt.ylabel('$U$')
plt.savefig('u_vs_time.png',bbox_inches='tight')





########################################## u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
# inlet
i1 = 0
plt.plot(u2d[i1,:],yp2d[i1,:],'b-',label="$x=0$")
xx=30
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(u2d[i1,:],yp2d[i1,:],'r--',label="$x=30$")
xx=50
i1 = (np.abs(xx-xp2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(u2d[i1,:],yp2d[i1,:],'k-.',label="$x=50$")
plt.legend(loc="upper left",prop=dict(size=18))
plt.ylabel("$y$")
plt.xlabel("$U$")
plt.axis([0, 1.1, 0,2.5])
plt.savefig('u_python.png',bbox_inches='tight')
