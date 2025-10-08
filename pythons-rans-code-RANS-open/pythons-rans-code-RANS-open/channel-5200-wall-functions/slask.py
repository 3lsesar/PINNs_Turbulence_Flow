
def modify_init(u2d,v2d,k2d,om2d,vis2d):
   
# set inlet field in entre domain
#  u2d=np.repeat(u_bc_west[None,:], repeats=ni, axis=0)
   k2d=np.ones((ni,nj))
   om2d=np.ones((ni,nj))

   vis2d=k2d/om2d+viscos

   return u2d,v2d,k2d,om2d,vis2d

def modify_inlet():

   global y_rans,y_rans,u_rans,v_rans,k_rans,om_rans,uv_rans,k_bc_west,eps_bc_west,om_bc_west

   return u_bc_west,v_bc_west,k_bc_west,om_bc_west,u2d_face_w,convw

def modify_conv(convw,convs):

   return convw,convs

def modify_u(su2d,sp2d):

   global file1

# we know that the convection and diffusion term in the x direction are zero
   aw2d=np.zeros((ni,nj))
   ae2d=np.zeros((ni,nj))

# we know that for this flow the wall shear stress mustt be equal to one (since the driving pressure
# gradient is equal to one). We print it every iteration to see if it is one. When it reaches one it is
# a good indicator that the flow has converged

# D IRECT NUMERICAL SIMULATION AND THEORIES OF WALL TURBULENCE WITH A RANGE OF PRESSURE
# GRADIENTS G.N. Coleman1 , A. Garbaruk2 and P.R. Spalart3

   su2d=su2d+vol

# we know that the convection and diffusion term in the x direction are zero
   aw2d=np.zeros((ni,nj))
   ae2d=np.zeros((ni,nj))

# we know that for this flow the wall shear stress mustt be equal to one (since the driving pressure
# gradient is equal to one). We print it every iteration to see if it is one. When it reaches one it is
# a good indicator that the flow has converged

# south wall
# take old ustar from k=cmu**(-0.5)*ustar**2
   ustar=cmu**0.25*k2d[:,0]**0.5
   tauw=ustar**2
   su2d[:,0]=su2d[:,0]-tauw*areas[:,0]

   print('south: tauw',tauw)

# north wall
   ustar=cmu**0.25*k2d[:,-1]**0.5
   tauw=ustar**2
   su2d[:,-1]=su2d[:,-1]-tauw*areas[:,0]

   print('north: tauw',tauw)

   print('south: np.sum(su2d[:,0])',np.sum(su2d[:,0]))
   print('north: np.sum(su2d[:,-1],np.sum(vol)',np.sum(su2d[:,-1]),np.sum(vol))

   if iter == 0:
      np.savetxt('u-iter.dat', np.c_[u2d[0,3],u2d[0,11],u2d[0,22],u2d[0,28]])
   else:
      with open('u-iter.dat','ab') as f:
        np.savetxt(f, np.c_[u2d[0,3],u2d[0,11],u2d[0,22],u2d[0,28]])

   return su2d,sp2d

def modify_v(su2d,sp2d):

   return su2d,sp2d

def modify_p(su2d,sp2d):

   return su2d,sp2d

def modify_k(su2d,sp2d):

# we know that the convection and diffusion term in the x direction are zero
   aw2d=np.zeros((ni,nj))
   ae2d=np.zeros((ni,nj))

   return su2d,sp2d

def modify_om(su2d,sp2d):

# we know that the convection and diffusion term in the x direction are zero
   aw2d=np.zeros((ni,nj))
   ae2d=np.zeros((ni,nj))

   return su2d,sp2d

def modify_outlet(convw):

# since we are solving for fully-developed channel flow, we know that the convection terms are zero
   convw=np.zeros((ni+1,nj))

   return convw

def fix_omega():

# take old ustar from k=cmu**(-0.5)*ustar**2
   ustar=cmu**0.25*k2d[:,0]**0.5
   dy=yp2d[0,0]

# south wall
   aw2d[:,0]=0
   ae2d[:,0]=0
   as2d[:,0]=0
   an2d[:,0]=0
   ap_max=np.max(ap2d)
   ap2d[:,0]=ap_max
# eps = ustar**3/kappa/dy
# k =cmu**(-0.5)*ustar**2
# omega =eps/k/cmu =  ustar**3/kappa/dy/cmu**(-0.5)/ustar**2/cmu = cmu**-0.5*ustar/kappa/dy
   su2d[:,0]=ap_max*cmu**(-0.5)*ustar/kappa/dy
   ustar_mean_s=np.mean(ustar)

# north wall
   ustar=cmu**0.25*k2d[:,-1]**0.5
   dy=y2d[0,-1]-yp2d[0,-1]
   aw2d[:,-1]=0
   ae2d[:,-1]=0
   as2d[:,-1]=0
   an2d[:,-1]=0
   ap_max=np.max(ap2d)
   ap2d[:,-1]=ap_max
   su2d[:,0]=ap_max*cmu**(-0.5)*ustar/kappa/dy
   ustar_mean_n=np.mean(ustar)

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d

def fix_k():

# south wall
# take old ustar from k=cmu**(-0.5)*ustar**2
   ustar=cmu**0.25*k2d[:,0]**0.5
   ustar_mean_s =  np.mean(ustar)
   dy=yp2d[0,0]
   print('south: dy',dy)
   velabs=abs(u2d[:,0]-u_bc_south)

# count u2d values smaller than u_bc_south
#  ustar=compute_ustar_reich_solve(dy/viscos,velabs,ustar)
   ustar=compute_ustar(dy,velabs,ustar,9,4)
   kwall=cmu**(-0.5)*ustar**2

   ustar_min =np.min(ustar)
   ustar_max =np.max(ustar)
   print('south wall: dy,ustar_min,max,mean',dy,ustar_min,ustar_max,ustar_mean_s)

   kwall_min =np.min(kwall)
   kwall_max =np.max(kwall)
   print('south wall: kwall_min,max',kwall_min,kwall_max)

   aw2d[:,0]=0
   ae2d[:,0]=0
   as2d[:,0]=0
   an2d[:,0]=0
   ap_max=np.max(ap2d)
   su2d[:,0]=ap_max*kwall
   ap2d[:,0]=ap_max

# north wall
   ustar=cmu**0.25*k2d[:,-1]**0.5
   ustar_mean_n =  np.mean(ustar)
   dy=y2d[0,-1]-yp2d[0,-1]
   print('north: dy',dy)
   velabs=abs(u2d[:,-1]-u_bc_north)
# count u2d values larger than u_bc_north
   number_n= (u2d[:,-1]-u_bc_north > 0).sum()
#  ustar=compute_ustar_reich_solve(dy/viscos,velabs,ustar)
   ustar=compute_ustar(dy,velabs,ustar,9,4)
   kwall=cmu**(-0.5)*ustar**2

   ustar_min =np.min(ustar)
   ustar_max =np.max(ustar)
   print('north wall: dy,ustar_min,max,mean',dy,ustar_min,ustar_max,ustar_mean_n)

   kwall_min =np.min(kwall)
   kwall_max =np.max(kwall)
   print('north wall: kwall_min,max',kwall_min,kwall_max)

   aw2d[:,-1]=0
   ae2d[:,-1]=0
   as2d[:,-1]=0
   an2d[:,-1]=0
   ap_max=np.max(ap2d)
   ap2d[:,-1]=ap_max
   su2d[:,-1]=ap_max*kwall

   if iter == 0:
      np.savetxt('ustar-history.dat', np.c_[ustar_mean_s,ustar_mean_n])
   else:
      print('fix_k saved')
      print('ustar_mean_s',ustar_mean_s)
      with open('ustar-history.dat','ab') as f:
            np.savetxt(f,np.c_[ustar_mean_s,ustar_mean_n])

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def compute_ustar_reich_solve(wdist,velabs,ustar):

   from scipy.optimize import fsolve,root,newton

   yplus=ustar*wdist/viscos

   ustar=ustar.flatten()
   velabs=velabs.flatten()
   #ustar = fsolve(solve_ustar_reich,x0=ustar,args=(velabs,wdist),xtol=1)
   ustar = newton(solve_ustar_reich,x0=ustar,args=(velabs,wdist))

   ustar=np.reshape(ustar,(ni,nk))

   return ustar

def compute_ustar(wdist,velabs,ustar,elog,n):

   for i in range(0,n):
      arg=np.maximum(elog*ustar*wdist/viscos,10.)
      ustar=kappa*velabs/np.log(arg)

   xyplus=ustar*wdist/viscos
   ustar=np.where(xyplus <= 11.69,(viscos*velabs/wdist)**0.5,ustar)

   return ustar

