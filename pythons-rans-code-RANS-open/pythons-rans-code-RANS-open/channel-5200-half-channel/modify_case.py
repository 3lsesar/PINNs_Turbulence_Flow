
def modify_init(u2d,v2d,k2d,om2d,vis2d):
   
# set inlet field in entre domain
#  u2d=np.repeat(u_bc_west[None,:], repeats=ni, axis=0)
   k2d=np.ones((ni,nj))
   om2d=np.ones((ni,nj))

   vis2d=k2d/om2d+viscos

   return u2d,v2d,k2d,om2d,vis2d,dist

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

   if iter == 100:
      vist = vis2d - viscos
      prod = gen*vist
      diss = cmu*k2d*om2d
#    ae2d=np.roll(aw2d,-1,axis=0)
      diff = as2d*(np.roll(k2d,1,axis=1)-k2d) + an2d*(np.roll(k2d,-1,axis=1)-k2d)

# turb. diff
      prand_k = 2
      viss=np.zeros((ni,nj+1))
      vis_turb=(vis2d-viscos)/prand_k

      viss[:,0:-1]=fy*vis_turb+(1-fy)*np.roll(vis_turb,1,axis=1)

      vols=np.ones((ni,nj+1))*1e-10
      vols[:,1:]=0.5*np.roll(vol,-1,axis=1)+0.5*vol
      diffs=viss[:,0:-1]*areas[:,0:-1]**2/vols[:,0:-1]

      as2d_t=diffs
      an2d_t=np.roll(diffs,-1,axis=1)

      diff_turb = as2d_t*(np.roll(k2d,1,axis=1)-k2d) + an2d_t*(np.roll(k2d,-1,axis=1)-k2d)


# visc. diff
      viss=np.ones((ni,nj+1))*viscos

      vols=np.ones((ni,nj+1))*1e-10
      vols[:,1:]=0.5*np.roll(vol,-1,axis=1)+0.5*vol
      diffs=viss[:,0:-1]*areas[:,0:-1]**2/vols[:,0:-1]
      diffn=viss[:,1:]*areas[:,1:]**2/vols[:,1:]

      as2d_lam=diffs
      an2d_lam=np.roll(diffs,-1,axis=1)

      diff_lam = as2d_lam*(np.roll(k2d,1,axis=1)-k2d) + an2d_lam*(np.roll(k2d,-1,axis=1)-k2d)

      np.savetxt('terms_k_eq.txt', \
        np.c_[prod[0,:],diss[0,:],diff[0,:],diff_turb[0,:],diff_lam[0,:]])
\


   return su2d,sp2d

def modify_om(su2d,sp2d):

   if iter == 100:
      vist = vis2d - viscos
      prod = c_omega_1*gen*vist*om2d/k2d
      psi = c_omega_2*om2d**2
#    ae2d=np.roll(aw2d,-1,axis=0)
      diff = as2d*(np.roll(om2d,1,axis=1)-om2d) + an2d*(np.roll(om2d,-1,axis=1)-om2d)

      source = su2d + sp2d*om2d

# turb. diff
      prand_om = 2
      viss=np.zeros((ni,nj+1))
      vis_turb=(vis2d-viscos)/prand_om

      viss[:,0:-1]=fy*vis_turb+(1-fy)*np.roll(vis_turb,1,axis=1)

      vols=np.ones((ni,nj+1))*1e-10
      vols[:,1:]=0.5*np.roll(vol,-1,axis=1)+0.5*vol
      diffs=viss[:,0:-1]*areas[:,0:-1]**2/vols[:,0:-1]

      as2d_t=diffs
      an2d_t=np.roll(diffs,-1,axis=1)

      diff_turb = as2d_t*(np.roll(om2d,1,axis=1)-om2d) + an2d_t*(np.roll(om2d,-1,axis=1)-om2d)

# visc. diff
      viss=np.ones((ni,nj+1))*viscos

      vols=np.ones((ni,nj+1))*1e-10
      vols[:,1:]=0.5*np.roll(vol,-1,axis=1)+0.5*vol
      diffs=viss[:,0:-1]*areas[:,0:-1]**2/vols[:,0:-1]
      diffn=viss[:,1:]*areas[:,1:]**2/vols[:,1:]

      as2d_lam=diffs
      an2d_lam=diffn

      diff_lam_north = an2d_lam*(np.roll(om2d,-1,axis=1)-om2d)
      diff_lam = as2d_lam*(np.roll(om2d,1,axis=1)-om2d) + an2d_lam*(np.roll(om2d,-1,axis=1)-om2d)

      print('an2d_lam/dx',an2d_lam[0,:]/(x2d[1,0]-x2d[0,0]))
      print('1/dy',1/(yp2d[0,1:]-yp2d[0,0:-1]))
      print('domdy_north',diff_lam_north[0,:]/(x2d[1,0]-x2d[0,0]))
      print('diff_lam',diff_lam[0,:]/(x2d[1,0]-x2d[0,0])/(y2d[0,1:]-y2d[0,0:-1]))

      np.savetxt('terms_omega_eq.txt', \
        np.c_[prod[0,:],psi[0,:],diff[0,:],source[0,:],su2d[0,:],sp2d[0,:]*om2d[0,:],diff_turb[0,:],diff_lam[0,:]])

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
