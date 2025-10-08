def modify_init(u2d,v2d,k2d,om2d,vis2d):

# set inlet field in entre domain
   u2d=np.repeat(u_bc_west[None,:], repeats=ni, axis=0)
   k2d=np.ones((ni,nj))
   om2d=np.ones((ni,nj))

   vis2d=k2d/om2d+viscos

   return u2d,v2d,k2d,om2d,vis2d,dist

def modify_PINN():
   
# load vist/viscos predicted by PINN
   vist_ML_5200 = np.loadtxt('../PINN/vist_pred-PINN-from-vist-diffusion-pinn-5200-plus-units-load-5-cells-fixed.txt')

   viscos_5200 = 1/5200

   vist_ML_5200 = vist_ML_5200*viscos_5200

   data_5200= np.loadtxt('y_u_k_om_uv_5200-RANS-half-channel.txt')
   y_5200 = data_5200[:,0]
   y1d= yp2d[0,:]

   prand_k_ML_5200 = np.loadtxt('../PINN/prand_k_5200-plus-units-from-balance.txt')

   print('prand_k_ML_5200',prand_k_ML_5200)

   vist_ML = np.interp(y1d, y_5200, vist_ML_5200)
   prand_k = np.interp(y1d, y_5200, prand_k_ML_5200)

# make it 2D
   prand_k = np.repeat(prand_k[None,:], repeats=ni, axis=0)

   print('vist_ML',vist_ML)

   vist = k2d/om2d

   print('prand_k',prand_k)

# load c_k taken from balance of k eq.
   c_k_ML_5200= np.loadtxt('../PINN/c_k_pred_5200-plus-units-from-balance.txt')
   c_k_ML = np.interp(y1d, y_5200, c_k_ML_5200)
# make it 2D
   c_k = np.repeat(c_k_ML[None,:], repeats=ni, axis=0)

# load c_omega_2 taken from balance of omega eq.
   c_omega_2_ML_5200 = np.loadtxt('../PINN/c_omega_2_pred_5200-plus-units-from-balance.txt')
   c_omega_2_ML = np.interp(y1d, y_5200, c_omega_2_ML_5200)
# make it 2D
   c_omega_2 = np.repeat(c_omega_2_ML[None,:], repeats=ni, axis=0)

   return prand_k, c_k, c_omega_2

def modify_inlet():

   global y_rans,y_rans,u_rans,v_rans,k_rans,om_rans,uv_rans,k_bc_west,eps_bc_west,om_bc_west

   return u_bc_west,v_bc_west,k_bc_west,om_bc_west,u2d_face_w,convw

def modify_conv(convw,convs):

# since we are solving for fully-developed channel flow, we know that the convection terms are zero
   convs=np.zeros((ni,nj))
   convw=np.zeros((ni,nj))

   return convw,convs

def modify_u(su2d,sp2d):

   global file1

# add a driving pressure gradient term
   su2d= su2d+vol

# we know that the convection and diffusion term in the x direction are zero
   aw2d=np.zeros((ni,nj))
   ae2d=np.zeros((ni,nj))

# we know that for this flow the wall shear stress mustt be equal to one (since the driving pressure
# gradient is equal to one). We print it every iteration to see if it is one. When it reaches one it is
# a good indicator that the flow has converged

   tauw_south=viscos*np.sum(as_bound*u2d[:,0])/x2d[-1,0]
   tauw_north=viscos*np.sum(an_bound*u2d[:,-1])/x2d[-1,0]

   print(f"{'tau wall, south: '} {tauw_south:.3f},{'  tau wall, north: '} {tauw_north:.3f}")


   return su2d,sp2d

def modify_v(su2d,sp2d):

   return su2d,sp2d

def modify_p(su2d,sp2d):

   return su2d,sp2d

def modify_k(su2d,sp2d):

   if iter == 0:
      np.savetxt('k-iteration.dat', np.c_[iter,k2d[-1,5],k2d[-1,10],k2d[-1,20],k2d[-1,30],k2d[-1,40],\
           k2d[-1,50],k2d[-1,58]])
   else:
      with open('k-iteration.dat','ab') as f:
         np.savetxt(f,np.c_[iter,k2d[-1,5],k2d[-1,10],k2d[-1,20],k2d[-1,30],k2d[-1,40],\
           k2d[-1,50],k2d[-1,58]])

   return su2d,sp2d

def modify_om(su2d,sp2d):

   return su2d,sp2d

def modify_outlet(convw):

# since we are solving for fully-developed channel flow, we know that the convection terms are zero
   convw=np.zeros((ni+1,nj))

   return convw

def fix_omega():

   aw2d[:,0]=0
   ae2d[:,0]=0
   as2d[:,0]=0
   an2d[:,0]=0
   ap2d[:,0]=1
   su2d[:,0]=om_bc_south

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d


def fix_k():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d
