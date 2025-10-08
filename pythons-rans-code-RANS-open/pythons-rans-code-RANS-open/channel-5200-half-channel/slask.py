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


def compute_face_phi(phi2d,phi_bc_west,phi_bc_east,phi_bc_south,phi_bc_north,\
    phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type):
   import numpy as np

   phi2d_face_w=np.empty((ni+1,nj))
   phi2d_face_s=np.empty((ni,nj+1))
   phi2d_face_w[0:-1,:]=fx*phi2d+(1-fx)*np.roll(phi2d,1,axis=0)
   phi2d_face_s[:,0:-1]=fy*phi2d+(1-fy)*np.roll(phi2d,1,axis=1)

# west boundary 
   phi2d_face_w[0,:]=phi_bc_west
   if phi_bc_west_type == 'n': 
# neumann
      phi2d_face_w[0,:]=phi2d[0,:]
   if cyclic_x:
      phi2d_face_w[0,:]=0.5*(phi2d[0,:]+phi2d[-1,:])

# east boundary 
   phi2d_face_w[-1,:]=phi_bc_east
   if phi_bc_east_type == 'n': 
# neumann
      phi2d_face_w[-1,:]=phi2d[-1,:]
      phi2d_face_w[-1,:]=phi2d_face_w[-2,:]
   if cyclic_x:
      phi2d_face_w[-1,:]=0.5*(phi2d[0,:]+phi2d[-1,:])


# south boundary 
   phi2d_face_s[:,0]=phi_bc_south
   if phi_bc_south_type == 'n': 
# neumann
      phi2d_face_s[:,0]=phi2d[:,0]

# north boundary 
   phi2d_face_s[:,-1]=phi_bc_north
   if phi_bc_north_type == 'n': 
# neumann
      phi2d_face_s[:,-1]=phi2d[:,-1]
   
   return phi2d_face_w,phi2d_face_s


def dphidy(phi_face_w,phi_face_s):

   phi_w=phi_face_w[0:-1,:]*areawy[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawy[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasy[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasy[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol


viscos=1/2000


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

y_kom = y

xs=0.5*(x2d[0:-1,0:-1]+x2d[1:,0:-1])
ys=0.5*(y2d[0:-1,0:-1]+y2d[1:,0:-1])

del1y=((xs-xp2d)**2+(ys-yp2d)**2)**0.5
del2y=((xs-np.roll(xp2d,1,axis=1))**2+(ys-np.roll(yp2d,1,axis=1))**2)**0.5
fy=del2y/(del1y+del2y)
fx=0.5*np.ones((ni,nj))

areawy=np.diff(x2d,axis=1)
areawx=-np.diff(y2d,axis=1)

areasy=-np.diff(x2d,axis=0)
areasx=np.diff(y2d,axis=0)

areaw=(areawx**2+areawy**2)**0.5
areas=(areasx**2+areasy**2)**0.5

# volume approaximated as the vector product of two triangles for cells
ax=np.diff(x2d,axis=1)
ay=np.diff(y2d,axis=1)
bx=np.diff(x2d,axis=0)
by=np.diff(y2d,axis=0)

areaz_1=0.5*np.absolute(ax[0:-1,:]*by[:,0:-1]-ay[0:-1,:]*bx[:,0:-1])

ax=np.diff(x2d,axis=1)
ay=np.diff(y2d,axis=1)
#  areaz_2=0.5*np.absolute(ax[1:,:]*by[:,0:-1]-ay[1:,:]*bx[:,0:-1])
areaz_2=0.5*np.absolute(ax[1:,:]*by[:,1:]-ay[1:,:]*bx[:,1:])

vol=areaz_1+areaz_2

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
eps=0.09*k*om

k_kom = k

ustar=(viscos*u[1]/y[1])**0.5
yplus=y*ustar/viscos

vist=vis-viscos
dudy=np.gradient(u,y)
uv=-vist*dudy

np.savetxt('y_u_k_om_uv_2000-RANS-half-channel.txt', np.c_[y,u,k,om,uv])

data_LES = np.loadtxt('../diffuser-14-degrees-WF-reichardt/y_u_channel-2000-WALE-600-150-300-half-channel-xmax-6-2dt.txt')

y_LES = data_LES[:,0]
u_LES = data_LES[:,1]

        # y/h             y+              U+             u'+             v'+             w'+           -Om_z+          om_x'+           om_y'+           om_z'+         uv'+             uw'+           vw'+             pr'+            ps'+          psto'+            p'
# ----------------------------------------------------------------------------------------------------------------------------------------
DNS_mean=np.genfromtxt("Re2000_jimenez.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]
u2_DNS=DNS_mean[:,3]**2
v2_DNS=DNS_mean[:,5]**2
w2_DNS=DNS_mean[:,4]**2
uv_DNS=DNS_mean[:,10]

#    y/h               y+           dissip          produc         p-strain          p-diff          t-diff         v-diff            bal          tp-kbal

DNS_k_terms=np.genfromtxt("Re2000_bal_k.dat",comments="%")

diss_DNS=-DNS_k_terms[:,2]
Pk_DNS=DNS_k_terms[:,3]
diff_DNS=DNS_k_terms[:,6]
diff_DNS_visc =  DNS_k_terms[:,7]

diss_iso_DNS = np.maximum(diss_DNS-diff_DNS_visc,0)

diss_DNS=diss_DNS/viscos
diff_DNS=diff_DNS/viscos
diff_DNS_visc=diff_DNS_visc/viscos
Pk_DNS=Pk_DNS/viscos


k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)



k_DNS_on_kom_grid = np.interp(y_kom, y_DNS, k_DNS)

# limit the ratio of k_DNS/k_kom. 
k_ratio = np.minimum(k_DNS_on_kom_grid/k_kom,5)
# new k_DNS
k_DNS_on_kom_grid = k_ratio*k_kom 



omega_DNS=diss_DNS/k_DNS/0.09

dudy_DNS = abs(np.gradient(u_DNS,y_DNS))

vist_DNS=abs(uv_DNS)/dudy_DNS

vist_DNS_on_kom_grid = np.interp(y_kom, y_DNS, vist_DNS)

omega_DNS_from_vist =k_DNS/vist_DNS
omega_DNS_from_vist_on_kom_grid = k_DNS_on_kom_grid/vist
# viscos = m2/s, diss = m2/s3 => omega => diss/viscos)**0.5
# vist = cmu*k2/eps => k = (vist*eps/cmu)**0.5
# omega = eps/k = eps/(vist*eps/cmu)**0.5 = (eps/vist/cmu)**0.5
omega_DNS_from_vist_and_diss=(diss_DNS/0.09/vist_DNS)**0.5
cmu_DNS_from_vist_and_diss=vist_DNS*omega_DNS_from_vist_and_diss/k_DNS
cmu_DNS=vist_DNS*omega_DNS/k_DNS


# find equi.distant DNS cells in log-scale
xx=0.
jDNS=[1]*40
for i in range (0,40):
   i1 = (np.abs(10.**xx-yplus_DNS)).argmin()
   jDNS[i]=int(i1)
   xx=xx+0.2

k_terms =  np.loadtxt('terms_k_eq.txt')
#       np.c_[prod[0,:],diss[0,:],diff[0,:],diff_turb[0,:],diff_lam[0,:]])
dy = np.diff(y2d[0,:])
dx = x2d[1,0]-x2d[0,0]
Pk = k_terms[:,0]
diss_k = k_terms[:,1]
diff_k = k_terms[:,2]/dx/dy
diff_k_turb = k_terms[:,3]/dx/dy
diff_k_lam = k_terms[:,4]/dx/dy

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
plt.axis([0, 200, -500, 500])
plt.savefig('k-balance_2000-channel.png',bbox_inches='tight')

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
plt.axis([0, 100, -550, 10])
plt.savefig('k-diss-visc-diff-channel.png',bbox_inches='tight')

########################################## k  diff  + diss - diff_visc large  omega
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(yplus[1:],-diss_k[1:]+diff_k_lam[1:],'b-',label=r'$k-\omega:$ $ D^{k,\nu} -  \varepsilon$')
plt.plot(yplus[1:],-diss_k[1:],'b--',label=r'$k-\omega:$ $-  \varepsilon$')
plt.plot(yplus[1:],-0.09*k_DNS_on_kom_grid[1:]*omega_DNS_from_vist_on_kom_grid[1:],'g--',label=r'$-0.09 k_{DNS} \omega_{DNS}$')
diff_DNS_visc_kom = np.interp(yp2d[0,:], y_DNS, diff_DNS_visc)
plt.plot(yplus_DNS,-diss_DNS+diff_DNS_visc,'r--',label=r'DNS: $ D^{k,\nu} - \varepsilon$')
plt.legend(loc='best',fontsize=20)
plt.xlabel("$y^+$")
plt.axis([0, 100, -3550, 10])
plt.savefig('k-diss-visc-diff-large-values-channel.png',bbox_inches='tight')



########################################## U 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.semilogx(yplus,u,'b-',label='RANS')
plt.semilogx(y_LES/viscos,u_LES,'r-',label='LES')
plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo',label='DNS')
plt.legend(loc='best',fontsize=20)
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")
plt.axis([1, 2000, 0, 28])
plt.savefig('u_log_2000-channel.png',bbox_inches='tight')


########################################## uv 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
# compute shear stress
plt.plot(yplus,uv,'b-')
plt.plot(yplus_DNS[jDNS],uv_DNS[jDNS],'bo')
plt.ylabel(r"$\overline{u'v'}$")
plt.xlabel("$y^+$")
plt.axis([1, 2000, -1, 1])
plt.savefig('uv_2000-channel.png',bbox_inches='tight')


########################################## k 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,k,'b-')
plt.plot(yplus_DNS,k_DNS,'r--')
plt.ylabel(r"$k^+$")
plt.xlabel("$y^+$")
plt.axis([1, 2000, 0, 5])
plt.savefig('k_2000-channel.png',bbox_inches='tight')

########################################## k  zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
k_rat = np.minimum(k_DNS_on_kom_grid/k,3)
plt.plot(yplus,k,'b-')
plt.plot(yplus_DNS,k_DNS,'r--')
plt.plot(yplus,k*k_rat,'g--')
plt.ylabel(r"$k^+$")
plt.xlabel("$y^+$")
plt.axis([1, 50, 0, 6])
plt.savefig('k_2000-channel-zoom.png',bbox_inches='tight')


########################################## vis 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,vis/viscos,'b-')
plt.ylabel(r"$\nu_t/\nu$")
plt.xlabel("$y^+$")
plt.axis([1, 2000, 0, 700])
plt.savefig('vis_2000-channel.png',bbox_inches='tight')



########################################## omega 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus,om,'b-',label='RANS')
plt.plot(yplus_DNS,omega_DNS,'r--',label=r'DNS from $k$, $\varepsilon$')
plt.plot(yplus_DNS,omega_DNS_from_vist,'k--',label=r'DNS from $\nu_t$, $k$')
plt.plot(yplus_DNS,omega_DNS_from_vist_and_diss,'g--',label=r'DNS from $\nu_t$, $\varepsilon$')
plt.legend(loc="best",prop=dict(size=14))
plt.ylabel(r"$\omega$")
plt.xlabel("$y^+$")
plt.axis([0, 2000, 0, 700])
plt.savefig('omega_2000-channel.png',bbox_inches='tight')


########################################## omega  zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.semilogx(yplus,om,'b-',label='RANS')
plt.ylabel(r"$\omega$")
plt.semilogx(yplus_DNS,omega_DNS,'r--',label=r'DNS from $k$, $\varepsilon$')
plt.semilogx(yplus_DNS,omega_DNS_from_vist,'k--',label=r'DNS from $\nu_t$, $k$')
plt.semilogx(yplus_DNS,omega_DNS_from_vist_and_diss,'g--',label=r'DNS from $\nu_t$, $\varepsilon$')
plt.legend(loc="best",prop=dict(size=14))
plt.xlabel("$y^+$")
plt.axis([1, 100, 0, 1e5])
plt.savefig('omega_2000-channel-zoom.png',bbox_inches='tight')

########################################## omega  zoom linear
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.25,bottom=0.20)
plt.plot(yplus,om,'b-',label='RANS')
om_fix =  np.ones(nj)
for j in range(0,10):
   om_fix[j] = 6*viscos/0.075/yp2d[0,j]**2
plt.plot(yplus,om_fix,'ro',label='Fixed')
plt.ylabel(r"$\omega$")
plt.legend(loc="best",prop=dict(size=14))
plt.xlabel("$y^+$")
plt.axis([0, 10, 0, 1e5])
plt.savefig('omega_2000-channel-zoom-linear.png',bbox_inches='tight')


sys.exit()


########################################## diss  log
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.semilogx(yplus,eps,'b-',label='RANS')
plt.ylabel(r"$\varepsilon$")
plt.semilogx(yplus_DNS,diss_DNS,'r--',label='DNS')
plt.legend(loc="best",prop=dict(size=14))
plt.xlabel("$y^+$")
plt.axis([1, 2000, 0, 1700])
plt.savefig('diss_log-2000-channel.png',bbox_inches='tight')

########################################## diss 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus,eps,'b-',label='RANS')
plt.ylabel(r"$\varepsilon$")
plt.plot(yplus_DNS,diss_DNS,'r--',label='DNS')
plt.legend(loc="upper left",prop=dict(size=14))
plt.xlabel("$y^+$")
plt.axis([0, 2000, 0, 300])
axins1 = inset_axes(ax1, width="50%", height="50%", loc='upper right', borderpad=0.2)
axins1.tick_params(axis = 'both', which = 'major', labelsize = 10)
plt.semilogx(yplus,eps,'b-',label='RANS')
plt.ylabel(r"$\varepsilon$")
plt.semilogx(yplus_DNS,diss_DNS,'r--',label='DNS')
plt.xlabel("$y^+$")
axins1.yaxis.set_label_position("left")
axins1.yaxis.tick_left()
axins1.xaxis.set_label_position("bottom")
axins1.xaxis.tick_bottom()
plt.axis([1, 100, 0,600])
plt.savefig('diss_2000-channel.png',bbox_inches='tight')



########################################## diss  zoom y
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus,eps,'b-',label='RANS')
plt.ylabel(r"$\varepsilon$")
plt.plot(yplus_DNS,diss_DNS,'r--',label='DNS')
plt.legend(loc="best",prop=dict(size=14))
plt.xlabel("$y^+$")
plt.axis([0, 2000, 0, 500])
plt.savefig('diss_zoom-2000-channel.png',bbox_inches='tight')


########################################## k over omega 
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i1 = 0
plt.plot(yplus,k/om,'b-',label=r'$k/\omega$')
plt.plot(yplus_DNS,k_DNS/omega_DNS,'r--',label=r'$k/\omega$. DNS')
plt.plot(yplus_DNS,vist_DNS,'k--',label=r'$\nu_t$. DNS')
plt.ylabel(r"$k/\omega$")
plt.xlabel("$y^+$")
plt.legend(loc="best",prop=dict(size=14))
plt.axis([1, 2000, 0, 0.2])
plt.savefig('k-over-omega-2000-channel.png',bbox_inches='tight')

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



j=1

plt.plot(yplus[j:],prod_py[j:]/dx/dy[j:],'b-',label=r'$P_\omega$')
plt.plot(yplus[j:],diff_turb_py[j:]/dx/dy[j:],'k-',label='turb.diff')
plt.plot(yplus[j:],diff_visc_py[j:]/dx/dy[j:],'g-',label='visc.diff')
plt.plot(yplus[j:],-Psi_py[j:]/dx/dy[j:],'r-',label='Psi')

imb_py = (prod_py + diff_py - Psi_py)/prod_py
imb_source = (source + diff_py)/prod_py
plt.axis([0, 50, -1e6, 1e6])
plt.ylabel(r"$\omega$ eq.")
plt.xlabel("$y^+$")
plt.title("from pyCALC")
plt.legend(loc="upper right",prop=dict(size=14))
plt.savefig('balance-omega-eq-2000-channel-zoom-from-pyCALC.png',bbox_inches='tight')

########################################## omega  balance zoom from pyCALC zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

j = 1

plt.semilogy(yplus[j:],prod_py[j:]/dx/dy[j:],'b-',label=r'$P_\omega$')
plt.semilogy(yplus[j:],diff_turb_py[j:]/dx/dy[j:],'k-',label='turb.diff')
plt.semilogy(yplus[j:],diff_visc_py[j:]/dx/dy[j:],'g-',label='visc.diff')
plt.semilogy(yplus[j:],diff_visc_py[j:]/dx/dy[j:],'go')
plt.semilogy(yplus[j:],Psi_py[j:]/dx/dy[j:],'r-',label='Psi')

plt.axis([0, 1000, 10, 4e13])
plt.ylabel(r"$\omega$ eq.")
plt.xlabel("$y^+$")
plt.title("from pyCALC")
plt.legend(loc="upper right",prop=dict(size=14))
plt.savefig('balance-omega-eq-2000-channel-zoom-from-pyCALC-log.png',bbox_inches='tight')


########################################## omega  balance zoom from pyCALC zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

j = 8

plt.plot(yplus[j:],prod_py[j:]/dx/dy[j:],'b-',label=r'$P_\omega$')
plt.plot(yplus[j:],diff_turb_py[j:]/dx/dy[j:],'k-',label='turb.diff')
plt.plot(yplus[j:],diff_visc_py[j:]/dx/dy[j:],'g-',label='visc.diff')
plt.plot(yplus[j:],diff_visc_py[j:]/dx/dy[j:],'go')
plt.plot(yplus[j:],-Psi_py[j:]/dx/dy[j:],'r-',label='-Psi')

plt.axis([0, 20, -6e6, 4e6])
plt.ylabel(r"$\omega$ eq.")
plt.xlabel("$y^+$")
plt.title("from pyCALC")
plt.legend(loc="upper right",prop=dict(size=14))
plt.savefig('balance-omega-eq-2000-channel-zoom-from-pyCALC-zoom.png',bbox_inches='tight')



