
def modify_init(u2d,v2d,k2d,om2d,vis2d):
   
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

# add a driving pressure gradient term
   su2d= su2d+vol

# we know that for this flow the wall shear stress mustt be equal to one (since the driving pressure
# gradient is equal to one). We print it every iteration to see if it is one. When it reaches one it is
# a good indicator that the flow has converged

   tauw_south=viscos*np.sum(as_bound*u2d[:,0])/x2d[-1,0]
   tauw_north=viscos*np.sum(an_bound*u2d[:,-1])/x2d[-1,0]

   print(f"{'tau wall, south: '} {tauw_south:.3f},{'  tau wall, north: '} {tauw_north:.3f}")

   if iter == 0:
      np.savetxt('u-iteration.dat', np.c_[iter,u2d[ni-5,5],u2d[ni-5,10],u2d[ni-5,15]])
   else:
      with open('u-iteration.dat','ab') as f:
         np.savetxt(f, np.c_[iter,u2d[ni-5,5],u2d[ni-5,10],u2d[ni-5,15]])

   return su2d,sp2d

def modify_v(su2d,sp2d):

   return su2d,sp2d

def modify_p(su2d,sp2d):

   return su2d,sp2d

def modify_k(su2d,sp2d):

   return su2d,sp2d

def modify_om(su2d,sp2d):

   return su2d,sp2d

def modify_outlet(convw):

# since we are solving for fully-developed channel flow, we know that the convection terms are zero
   convw=np.zeros((ni+1,nj))

   return convw

def fix_omega():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d


def fix_k():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d
