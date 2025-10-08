from scipy import sparse
import numpy as np
import sys
import time
from scipy.sparse import spdiags,linalg,eye

def setup_case():
   global  c_omega_1, c_omega_2, cmu, convergence_limit_eps, convergence_limit_k, convergence_limit_om, convergence_limit_pp, \
   convergence_limit_u, convergence_limit_v, convergence_limit_w, dist,fx, fy,imon,jmon,kappa,k_bc_east,k_bc_east_type, \
   k_bc_north,k_bc_north_type,k_bc_south, k_bc_south_type,k_bc_west,k_bc_west_type,kom,maxit, \
   ni,nj,nsweep_kom, nsweep_pp, nsweep_vel,  om_bc_east, om_bc_east_type, om_bc_north, om_bc_north_type, \
   om_bc_south, om_bc_south_type, om_bc_west, om_bc_west_type, p_bc_east, p_bc_east_type, \
   p_bc_north, p_bc_north_type, p_bc_south, p_bc_south_type, p_bc_west, p_bc_west_type, \
   prand_k,prand_omega,resnorm_p,resnorm_vel,restart,save,save_vtk_movie,scheme,scheme_turb,solver_pp,solver_vel, \
   solver_turb,sormax, u_bc_east, u_bc_east_type, u_bc_north, u_bc_north_type, u_bc_south, u_bc_south_type, u_bc_west, \
   u_bc_west_type, urfvis, urf_vel, urf_k, urf_p,urf_omega,v_bc_east, v_bc_east_type, v_bc_north, v_bc_north_type, \
   v_bc_south, v_bc_south_type,v_bc_west, v_bc_west_type,viscos, vol,vtk,vtk_save,vtk_file_name,x2d, xp2d, y2d, yp2d


   import numpy as np
   import sys


########### section 1 choice of differencing scheme ###########
   scheme='h'  #hybrid
   scheme_turb='h'  #hybrid upwind-central 

########### section 2 turbulence models ###########
   cmu=0.09
   kom = False
   c_omega_1= 5./9.
   c_omega_2=3./40.
   prand_omega=2.0
   prand_k=2.0

########### section 3 restart/save ###########
   restart = False
   save = True

########### section 4 fluid properties ###########
   viscos=1/1000

########### section 5 relaxation factors ###########
   urfvis=0.5
   urf_vel=0.5
   urf_k=0.5
   urf_p=1.0
   urf_omega=0.5

########### section 6 number of iteration and convergence criterira ###########
   maxit=200
   min_iter=1
   sormax=1e-20

   solver_vel='lgmres'
   solver_pp='lgmres'
   solver_vel='direct'
   solver_pp='direct'
   solver_turb='gmres'
   nsweep_vel=50
   nsweep_pp=50
   nsweep_kom=1
   convergence_limit_vel=1e-6
   convergence_limit_u=1e-6
   convergence_limit_v=1e-6
   convergence_limit_w=1e-6
   convergence_limit_k=1e-10
   convergence_limit_om=1e-10
   convergence_limit_pp=5e-4

########### section 7 all variables are printed during the iteration at node ###########
   imon=ni-10
   jmon=int(nj/2)

########### section 8 time-averaging ###########
   vtk=False
   save_all_files=False
   vtk_file_name='bound'

########### section 9 residual scaling parameters ###########
   uin=1
   resnorm_p=uin*y2d[1,-1]
   resnorm_vel=uin**2*y2d[1,-1]


########### Section 10 boundary conditions ###########

# boundary conditions for u
   u_bc_west=np.zeros(nj)
   u_bc_east=np.zeros(nj)
   u_bc_south=np.zeros(ni)
   u_bc_north=np.ones(ni)

   u_bc_west_type='d' 
   u_bc_east_type='d' 
   u_bc_south_type='d'
   u_bc_north_type='d'

# boundary conditions for v
   v_bc_west=np.zeros(nj)
   v_bc_east=np.zeros(nj)
   v_bc_south=np.zeros(ni)
   v_bc_north=np.zeros(ni)

   v_bc_west_type='d' 
   v_bc_east_type='d' 
   v_bc_south_type='d'
   v_bc_north_type='d'

# boundary conditions for p
   p_bc_west=np.zeros(nj)
   p_bc_east=np.zeros(nj)
   p_bc_south=np.zeros(ni)
   p_bc_north=np.zeros(ni)

   p_bc_west_type='n'
   p_bc_east_type='n'
   p_bc_south_type='n'
   p_bc_north_type='n'

# boundary conditions for k
   k_bc_west=np.ones(nj)*1e-2
   k_bc_west[10:]=1e-5
   k_bc_east=np.zeros(nj)
   k_bc_south=np.zeros(ni)
   k_bc_north=np.ones(ni)*1e-5

   k_bc_west_type='d'
   k_bc_east_type='d'
   k_bc_south_type='d'
   k_bc_north_type='n'

# boundary conditions for omega
   om_bc_west=np.ones(nj)
   om_bc_east=np.zeros(nj)
   om_bc_south=np.zeros(ni)
   om_bc_north=np.zeros(ni)

   xwall_s=0.5*(x2d[0:-1,0]+x2d[1:,0])
   ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
   dist2_s=(yp2d[:,0]-ywall_s)**2+(xp2d[:,0]-xwall_s)**2
   om_bc_south=10*6*viscos/0.075/dist2_s

   om_bc_west_type='d'
   om_bc_east_type='n'
   om_bc_south_type='d'
   om_bc_north_type='n'

   return 



def modify_init(u2d,v2d,k2d,om2d,vis2d):
   
   return u2d,v2d,k2d,om2d,vis2d

def modify_inlet():

   global y_rans,y_rans,u_rans,v_rans,k_rans,om_rans,uv_rans,k_bc_west,eps_bc_west,om_bc_west

   return u_bc_west,v_bc_west,k_bc_west,om_bc_west,u2d_face_w,convw

def modify_conv(convw,convs):

   convs[:,0,:]=0
   convs[:,-1,:]=0

   return convw,convs

def modify_u(su2d,sp2d):

   global file1

   su2d[0,:]= su2d[0,:]+convw[0,:]*u_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:,]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*u_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound

   return su2d,sp2d

def modify_v(su2d,sp2d):
   su2d[0,:]= su2d[0,:]+convw[0,:]*v_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*v_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound

   return su2d,sp2d

def modify_k(su2d,sp2d,gen):

   su2d[0,:]= su2d[0,:]+np.maximum(convw[0,:],0)*k_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*k_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound


   return su2d,sp2d

def modify_om(su2d,sp2d):


   su2d[0,:]= su2d[0,:]+np.maximum(convw[0,:],0)*om_bc_west
   sp2d[0,:]= sp2d[0,:]-convw[0,:]
   vist=vis2d[0,:]-viscos
   su2d[0,:]=su2d[0,:]+vist*aw_bound*om_bc_west
   sp2d[0,:]=sp2d[0,:]-vist*aw_bound

   return su2d,sp2d

def modify_outlet(convw):

# inlet
   flow_in=np.sum(convw[0,:])
   flow_out=np.sum(convw[-1,:])
   area_out=np.sum(areaw[-1,:])

   uinc=(flow_in-flow_out)/area_out
   ares=areaw[-1,:]
   convw[-1,:]=convw[-1,:]+uinc*ares

   print('area_out',area_out)

   flow_out_new=np.sum(convw[-1,:])

   print('flow_in',flow_in,'flow_out',flow_out,'area_out',area_out,'flow_out_new',flow_out_new,'uinc:',uinc)

   return convw

def fix_omega():

#  aw2d[:,0,:]=0
#  ae2d[:,0,:]=0
#  as2d[:,0,:]=0
#  an2d[:,0,:]=0
#  al2d[:,0,:]=0
#  ah2d[:,0,:]=0
#  ap2d[:,0,:]=1
#  su2d[:,0,:]=om_bc_south

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d


def fix_k():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d
