
def modify_init(u2d,v2d,k2d,om2d,vis2d):
   
# read inlet data. DNS data found in my eBook, Assignment 1 in the course MTF271 Turbulence Modeling

# set inlet b.c. in the entire domain
   data = np.loadtxt('yp-u-k-omega.dat')
   yin = data[:,0]
   uin = data[:,1]
   kin = data[:,2]
   omin = data[:,3]
   uin_interp=np.interp(yp2d[0,:], yin, uin)
   kin_interp=np.interp(yp2d[0,:], yin, kin)
   omin_interp=np.interp(yp2d[0,:], yin, omin)

# set initial field in entire domain
   y0=y2d[0,-1]-y2d[0,0]
   for i in range(0,ni):
      yi=y2d[i,-1]-y2d[i,0]
      u2d[i,:]=uin_interp*y0/yi
      k2d[i,:]=kin_interp
      om2d[i,:]=omin_interp

   vis2d=k2d/om2d+viscos

   return u2d,v2d,k2d,om2d,vis2d,dist

def modify_inlet():

# read inlet data. DNS data found in my eBook, Assignment 1 in the course MTF271 Turbulence Modeling
   data = np.loadtxt('yp-u-k-omega.dat')
   yin = data[:,0]
   uin = data[:,1]
   kin = data[:,2]
   omin = data[:,3]
   uin_interp=np.interp(yp2d[0,:], yin, uin)
   kin_interp=np.interp(yp2d[0,:], yin, kin)
   omin_interp=np.interp(yp2d[0,:], yin, omin)

# interpolate to the CFD grid

   u_bc_west = uin_interp
   k_bc_west = kin_interp
   om_bc_west = omin_interp

   return u_bc_west,v_bc_west,k_bc_west,om_bc_west,u2d_face_w,convw

def modify_conv(convw,convs):

   return convw,convs

def modify_u(su2d,sp2d):

   su2d[0,:]= su2d[0,:]+convw[0,:]*u_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:,]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*u_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound

   if iter == 0:
      print('u(5)=%7.3E,u(10)=%7.3E,u(20)=%7.3E,u(30)=%7.3E,u(40)=%7.3E,u(50)=%7.3E,u(60)=%7.3E'\
 %(u2d[ni-5,5],u2d[ni-5,10],u2d[ni-5,20],u2d[ni-5,30],u2d[ni-5,40],u2d[ni-5,50],u2d[ni-5,60]))

   if iter == 0:
      np.savetxt('u-iteratiom.dat', np.c_[iter,u2d[ni-5,5],u2d[ni-5,10],u2d[ni-5,20],u2d[ni-5,30],u2d[ni-5,40],\
           u2d[ni-5,50],u2d[ni-5,60]])
   else:
      with open('u-iteratiom.dat','ab') as f:
         np.savetxt(f,np.c_[iter,u2d[ni-5,5],u2d[ni-5,10],u2d[ni-5,20],u2d[ni-5,30],u2d[ni-5,40],\
           u2d[ni-5,50],u2d[ni-5,60]])


   return su2d,sp2d

def modify_v(su2d,sp2d):

   su2d[0,:]= su2d[0,:]+convw[0,:]*v_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:,]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*v_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound


   return su2d,sp2d

def modify_p(su2d,sp2d):

   return su2d,sp2d

def modify_k(su2d,sp2d):

   su2d[0,:]= su2d[0,:]+convw[0,:]*k_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:,]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*k_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound


   return su2d,sp2d

def modify_om(su2d,sp2d):

   su2d[0,:]= su2d[0,:]+convw[0,:]*om_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:,]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*om_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound

   return su2d,sp2d

def modify_outlet(convw):

# inlet
   flow_in=np.sum(convw[0,:])
#  flow_out=np.sum(convw[-1,:])
   flow_out=np.sum(convw[-2,:])
   area_out=np.sum(areaw[-1,:])

   uinc=(flow_in-flow_out)/area_out
   ares=areaw[-1,:]
#  convw[-1,:]=convw[-1,:]+uinc*ares
   convw[-1,:]=convw[-2,:]+uinc*ares

   print('area_out',area_out)

   flow_out_new=np.sum(convw[-1,:])

   print(f"{'flow_in: '} {flow_in:.3e},{'  flow_out: '} {flow_out:.3e},{'  flow_out_new: '} {flow_out_new:.3e},{'  uinc: '} {uinc:.3e}")

   return convw

def fix_omega():

   aw2d[:,0]=0
   ae2d[:,0]=0
   as2d[:,0]=0
   an2d[:,0]=0
   ap2d[:,0]=1
   su2d[:,0]=6*viscos/0.075/dist[:,0]**2

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d


def fix_k():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d
