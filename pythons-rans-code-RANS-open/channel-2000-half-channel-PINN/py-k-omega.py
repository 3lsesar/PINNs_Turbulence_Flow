from scipy import sparse
import numpy as np
import sys
import time
import pyamg
from scipy.sparse import spdiags,linalg,eye
import socket

def setup_case():

   global  c_omega_1, c_omega_2, cmu, convergence_limit_k, convergence_limit_om,  \
   convergence_limit_u, dist,fx, fy,imon,jmon,kappa,k_bc_east,k_bc_east_type, \
   k_bc_north,k_bc_north_type,k_bc_south, k_bc_south_type,k_bc_west,k_bc_west_type,kom,maxit, \
   ni,nj,nsweep_kom, nsweep_u,  om_bc_east, om_bc_east_type, om_bc_north, om_bc_north_type, \
   om_bc_south, om_bc_south_type, om_bc_west, om_bc_west_type, \
   prand_k,prand_omega,restart,save, \
   solver_turb, solver_u, sormax, u_bc_east, u_bc_east_type, u_bc_north, u_bc_north_type, u_bc_south, u_bc_south_type, u_bc_west, \
   u_bc_west_type, urfvis, urf_k, urf_u,urf_omega, viscos, vol,vtk,vtk_save,vtk_file_name,x2d, xp2d, y2d, yp2d


   import numpy as np
   import sys

########### section 2 turbulence models ###########
   cmu=0.09
   kom = True
   c_omega_1= 5./9.
   c_omega_2=3./40.
   prand_omega=2.0*np.ones((ni,nj))
   prand_k=2.0*np.ones((ni,nj))

########### section 3 restart/save ###########
   restart = True
   save = True

########### section 4 fluid properties ###########
   viscos=1/2000

########### section 5 relaxation factors ###########
   urf_u=0.5
   urfvis=0.5
   urf_k=0.5
   urf_omega=0.5

########### section 6 number of iteration and convergence criterira ###########
   maxit=50000
   min_iter=1
   sormax=4e-7

   solver_u='lgmres'
   solver_u='direct'
   nsweep_u=50
   convergence_limit_u=1e-6
   solver_turb='direct'

   nsweep_kom=1
   convergence_limit_u=1e-6
   convergence_limit_k=1e-7
   convergence_limit_om=1e-7

########### section 7 all variables are printed during the iteration at node ###########
   imon=0
   jmon=int(nj/2)

########### section 9 residual scaling parameters ###########
########### Section 10 boundary conditions ###########

# boundary conditions for u
   u_bc_west=np.zeros(nj)
   u_bc_east=np.zeros(nj)
   u_bc_south=np.zeros(ni)
   u_bc_north=np.zeros(ni) 

   u_bc_west_type='n' 
   u_bc_east_type='n' 
   u_bc_south_type='d'
   u_bc_north_type='n'

# boundary conditions for k
   k_bc_west=np.zeros(nj)
   k_bc_east=np.zeros(nj)
   k_bc_south=np.zeros(ni)
   k_bc_north=np.zeros(ni)

   k_bc_west_type='n'
   k_bc_east_type='n'
   k_bc_south_type='d'
   k_bc_north_type='n'

# boundary conditions for omega
   om_bc_west=np.zeros(nj)
   om_bc_east=np.zeros(nj)

   xwall_s=0.5*(x2d[0:-1,0]+x2d[1:,0])
   ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
   dist2_s=(yp2d[:,0]-ywall_s)**2+(xp2d[:,0]-xwall_s)**2
   om_bc_south=10*6*viscos/0.075/dist2_s

   xwall_n=0.5*(x2d[0:-1,-1]+x2d[1:,-1])
   ywall_n=0.5*(y2d[0:-1,-1]+y2d[1:,-1])
   dist2_n=(yp2d[:,-1]-ywall_n)**2+(xp2d[:,-1]-xwall_n)**2
   om_bc_north=6*viscos/0.075/dist2_n

   om_bc_west_type='n'
   om_bc_east_type='n'
   om_bc_south_type='d'
   om_bc_north_type='n'

   return 

def modify_init(u2d,k2d,om2d,vis2d):

# set inlet field in entre domain
#  u2d=np.repeat(u_bc_west[None,:], repeats=ni, axis=0)
   k2d=np.ones((ni,nj))
   om2d=np.ones((ni,nj))

   vis2d=k2d/om2d+viscos

   return u2d,k2d,om2d,vis2d


def modify_u(su2d,sp2d):

   su2d = su2d + vol

   tauw_south=viscos*np.sum(as_bound*u2d[:,0])/x2d[-1,0]
   tauw_north=viscos*np.sum(an_bound*u2d[:,-1])/x2d[-1,0]

   print(f"{'tau wall, south: '} {tauw_south:.3f},{'  tau wall, north: '} {tauw_north:.3f}")

   print('as_bound[0],an',as_bound[0],an_bound[0])

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

   if iter == 0:
      np.savetxt('u-iteration.dat', np.c_[iter,k2d[-1,5],k2d[-1,10],k2d[-1,20],k2d[-1,30],k2d[-1,40],\
           k2d[-1,50],k2d[-1,55]])
   else:
      with open('u-iteration.dat','ab') as f:
         np.savetxt(f,np.c_[iter,k2d[-1,5],k2d[-1,10],k2d[-1,20],k2d[-1,30],k2d[-1,40],\
           k2d[-1,50],k2d[-1,55]])


   return su2d,sp2d

def fix_omega():

#  j=7
#  aw2d[0:j,0]=0
#  ae2d[0:j,0]=0
#  as2d[0:j,0]=0
#  an2d[0:j,0]=0
#  ap2d[0:j,0]=1
#  su2d[0:j,0]=om_bc_south

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d

def init():
   print('hostname: ',socket.gethostname())

# distance to nearest wall
   ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
   dist_s=yp2d-ywall_s[:,None]
   ywall_n=0.5*(y2d[0:-1,-1]+y2d[1:,-1])
   dist_n=ywall_n[:,None] -yp2d
   dist=np.minimum(dist_s,dist_n)

#  west face coordinate
   xw=0.5*(x2d[0:-1,0:-1]+x2d[0:-1,1:])
   yw=0.5*(y2d[0:-1,0:-1]+y2d[0:-1,1:])

   del1x=((xw-xp2d)**2+(yw-yp2d)**2)**0.5
   del2x=((xw-np.roll(xp2d,1,axis=0))**2+(yw-np.roll(yp2d,1,axis=0))**2)**0.5
   fx=del2x/(del1x+del2x)

#  south face coordinate
   xs=0.5*(x2d[0:-1,0:-1]+x2d[1:,0:-1])
   ys=0.5*(y2d[0:-1,0:-1]+y2d[1:,0:-1])

   del1y=((xs-xp2d)**2+(ys-yp2d)**2)**0.5
   del2y=((xs-np.roll(xp2d,1,axis=1))**2+(ys-np.roll(yp2d,1,axis=1))**2)**0.5
   fy=del2y/(del1y+del2y)

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

# coeff at south wall (without viscosity)
   as_bound=areas[:,0]**2/(0.5*vol[:,0])

# coeff at north wall (without viscosity)
   an_bound=areas[:,-1]**2/(0.5*vol[:,-1])

# coeff at west wall (without viscosity)
   aw_bound=areaw[0,:]**2/(0.5*vol[0,:])

   ae_bound=areaw[-1,:]**2/(0.5*vol[-1,:])

   return areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,aw_bound,ae_bound,as_bound,an_bound,dist

def read_restart_data():

   print('read_restart_data called')

   u2d=np.load('u2d_saved.npy')
   k2d=np.load('k2d_saved.npy')
   om2d=np.load('om2d_saved.npy')
   vis2d=np.load('vis2d_saved.npy')
   ap2d_vel=np.load('ap2d_vel_saved.npy')

   return u2d,k2d,om2d,vis2d,ap2d_vel

def save_data(u2d,k2d,om2d,vis2d,ap2d_vel):

   print('save_data called')
   np.save('u2d_saved', u2d)   # don't save u. It is not solved for
   np.save('k2d_saved', k2d)
   np.save('om2d_saved', om2d)
   np.save('vis2d_saved', vis2d)
   np.save('ap2d_vel_saved',ap2d_vel)

   return


def print_indata():

   print('////////////////// Start of input data ////////////////// \n\n\n')

   print('\n\n########### section 2 turbulence models ###########')

   print(f"{'cmu: ':<29} {cmu}")
   print(f"{'c_omega_1: ':<29} {c_omega_1:.3f}")
   print(f"{'c_omega_2: ':<29} {c_omega_2}")
   print(f"{'prand_k: ':<29} {prand_k[0,0]}")
   print(f"{'prand_omega: ':<29} {prand_omega[0,0]}")

   print('\n\n########### section 3 restart/save ###########')
   print(f"{'restart: ':<29} {restart}")
   print(f"{'save: ':<29} {save}")


   print('\n\n########### section 4 fluid properties ###########')
   print(f"{'viscos: ':<29} {viscos:.2e}")

   print('\n\n########### section 6 number of iteration and convergence criterira ###########')
   print(f"{'sormax: ':<29} {sormax}")
   print(f"{'maxit: ':<29} {maxit}")
   print(f"{'solver_u: ':<29} {solver_u}")
   print(f"{'solver_turb: ':<29} {solver_turb}")
   print(f"{'nsweep_u: ':<29} {nsweep_u}")
   print(f"{'nsweep_kom: ':<29} {nsweep_kom}")
   print(f"{'convergence_limit_u: ':<29} {convergence_limit_u}")
   print(f"{'convergence_limit_k: ':<29} {convergence_limit_k}")
   print(f"{'convergence_limit_om: ':<29} {convergence_limit_om}")


   print('\n\n########### section 7 all variables are printed during the iteration at node ###########')
   print(f"{'imon: ':<29} {imon}")
   print(f"{'jmon: ':<29} {jmon}")

   print('\n\n########### Section 10 grid and boundary conditions ###########')
   print(f"{'ni: ':<29} {ni}")
   print(f"{'nj: ':<29} {nj}")
   print('\n')
   print('\n')

   print('------boundary conditions for u')
   print(f"{' ':<5}{'u_bc_west_type: ':<29} {u_bc_west_type}")
   print(f"{' ':<5}{'u_bc_east_type: ':<29} {u_bc_east_type}")
   if u_bc_west_type == 'd':
      print(f"{' ':<5}{'u_bc_west[0]: ':<29} {u_bc_west[0]}")
   if u_bc_east_type == 'd':
      print(f"{' ':<5}{'u_bc_east[0]: ':<29} {u_bc_east[0]}")


   print(f"{' ':<5}{'u_bc_south_type: ':<29} {u_bc_south_type}")
   print(f"{' ':<5}{'u_bc_north_type: ':<29} {u_bc_north_type}")

   if u_bc_south_type == 'd':
      print(f"{' ':<5}{'u_bc_south[0]: ':<29} {u_bc_south[0]}")
   if u_bc_north_type == 'd':
      print(f"{' ':<5}{'u_bc_north[0]: ':<29} {u_bc_north[0]}")

   print('------boundary conditions for k')
   print(f"{' ':<5}{'k_bc_west_type: ':<29} {k_bc_west_type}")
   print(f"{' ':<5}{'k_bc_east_type: ':<29} {k_bc_east_type}")
   if k_bc_west_type == 'd':
      print(f"{' ':<5}{'k_bc_west[0]: ':<29} {k_bc_west[0]}")
   if k_bc_east_type == 'd':
      print(f"{' ':<5}{'k_bc_east[0]: ':<29} {k_bc_east[0]}")

   print(f"{' ':<5}{'k_bc_south_type: ':<29} {k_bc_south_type}")
   print(f"{' ':<5}{'k_bc_north_type: ':<29} {k_bc_north_type}")

   if k_bc_south_type == 'd':
      print(f"{' ':<5}{'k_bc_south[0]: ':<29} {k_bc_south[0]}")
   if k_bc_north_type == 'd':
      print(f"{' ':<5}{'k_bc_north[0]: ':<29} {k_bc_north[0]}")

   print('------boundary conditions for omega')
   print(f"{' ':<5}{'om_bc_west_type: ':<29} {om_bc_west_type}")
   print(f"{' ':<5}{'om_bc_east_type: ':<29} {om_bc_east_type}")
   if om_bc_west_type == 'd':
      print(f"{' ':<5}{'om_bc_west[0]: ':<29} {om_bc_west[0]:.1f}")
   if om_bc_east_type == 'd':
      print(f"{' ':<5}{'om_bc_east[0]: ':<29} {om_bc_east[0]:.1f}")

   print(f"{' ':<5}{'om_bc_south_type: ':<29} {om_bc_south_type}")
   print(f"{' ':<5}{'om_bc_north_type: ':<29} {om_bc_north_type}")

   if om_bc_south_type == 'd':
      print(f"{' ':<5}{'om_bc_south[0]: ':<29} {om_bc_south[0]:.1f}")
   if om_bc_north_type == 'd':
      print(f"{' ':<5}{'om_bc_north[0]: ':<29} {om_bc_north[0]:.1f}")

   print('\n\n\n ////////////////// End of input data //////////////////\n\n\n')

   return 

def coeff(vis2d,prand):

   visw=np.zeros((ni+1,nj))
   viss=np.zeros((ni,nj+1))
   vis_turb=(vis2d-viscos)/prand

   visw[0:-1,:]=fx*vis_turb+(1-fx)*np.roll(vis_turb,1,axis=0)+viscos
   viss[:,0:-1]=fy*vis_turb+(1-fy)*np.roll(vis_turb,1,axis=1)+viscos

   volw=np.ones((ni+1,nj))*1e-10
   vols=np.ones((ni,nj+1))*1e-10
   volw[1:,:]=0.5*np.roll(vol,-1,axis=0)+0.5*vol
   diffw=visw[0:-1,:]*areaw[0:-1,:]**2/volw[0:-1,:]
   vols[:,1:]=0.5*np.roll(vol,-1,axis=1)+0.5*vol
   diffs=viss[:,0:-1]*areas[:,0:-1]**2/vols[:,0:-1]

   aw2d=diffw
   ae2d=np.roll(diffw,-1,axis=0)

   as2d=diffs
   an2d=np.roll(diffs,-1,axis=1)

   as2d[:,0]=0
   an2d[:,-1]=0

   aw2d[0,:]=0
   ae2d[-1,:]=0

   return aw2d,ae2d,as2d,an2d,su2d,sp2d

def bc(su2d,sp2d,phi_bc_west,phi_bc_east,phi_bc_south,phi_bc_north\
     ,phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type):

   su2d=np.zeros((ni,nj))
   sp2d=np.zeros((ni,nj))

#south
   if phi_bc_south_type == 'd':
      sp2d[:,0]=sp2d[:,0]-viscos*as_bound
      su2d[:,0]=su2d[:,0]+viscos*as_bound*phi_bc_south

#north
   if phi_bc_north_type == 'd':
      sp2d[:,-1]=sp2d[:,-1]-viscos*an_bound
      su2d[:,-1]=su2d[:,-1]+viscos*an_bound*phi_bc_north

#west
   if phi_bc_west_type == 'd':
      sp2d[0,:]=sp2d[0,:]-viscos*aw_bound
      su2d[0,:]=su2d[0,:]+viscos*aw_bound*phi_bc_west
#east
   if phi_bc_east_type == 'd':
      sp2d[-1,:]=sp2d[-1,:]-viscos*ae_bound
      su2d[-1,:]=su2d[-1,:]+viscos*ae_bound*phi_bc_east

   return su2d,sp2d

def solve_2d(phi2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,tol_conv,nmax,solver_local):
   if iter == 0:
      print('solve_2d called')
      print('nmax',nmax)

   aw=np.matrix.flatten(aw2d)
   ae=np.matrix.flatten(ae2d)
   as1=np.matrix.flatten(as2d)
   an=np.matrix.flatten(an2d)
   ap=np.matrix.flatten(ap2d)
  
   m=ni*nj

   A = sparse.diags([ap, -an[0:-1], -as1[1:], -ae, -aw[nj:]], [0, 1, -1, nj, -nj], format='csr')

   su=np.matrix.flatten(su2d)
   phi=np.matrix.flatten(phi2d)

   res_su=np.linalg.norm(su)
   resid_init=np.linalg.norm(A*phi - su)

   phi_org=phi

   resid=np.linalg.norm(A*phi - su)
   tol=tol_conv
   if tol_conv < 0:
# use absolute convergence criterium
      tol=1e-10
      tol_conv=abs(tol_conv)*resid
# bicg (BIConjugate Gradient)
# bicgstab (BIConjugate Gradient STABilized)
# cg (Conjugate Gradient) - symmetric positive definite matrices only
# cgs (Conjugate Gradient Squared)
# gmres (Generalized Minimal RESidual)
# minres (MINimum RESidual)
# qmr (Quasi
   if solver_local == 'direct':
      if iter == 0:
         print('solver in solve_2d: direct solver')
      info=0
      resid=np.linalg.norm(A*phi - su)
      phi = linalg.spsolve(A,su)
   if solver_local == 'pyamg':
      if iter == 0:
         print('solver in solve_2d: pyamg solver')
      App = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
      res_amg = []
      phi = App.solve(su, tol=tol, x0=phi, residuals=res_amg)
      info=0
      print('Residual history in pyAMG', ["%0.4e" % i for i in res_amg])
   if solver_local == 'cgs':
      if iter == 0:
         print('solver in solve_2d: cgs')
      phi,info=linalg.cgs(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'cg':
      if iter == 0:
         print('solver in solve_2d: cg')
      phi,info=linalg.cg(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'gmres':
      if iter == 0:
         print('solver in solve_2d: gmres')
      phi,info=linalg.gmres(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'qmr':
      if iter == 0:
         print('solver in solve_2d: qmr')
      phi,info=linalg.qmr(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'lgmres':
      if iter == 0:
         print('solver in solve_2d: lgmres')
      phi,info=linalg.lgmres(A,su,x0=phi, tol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if info > 0:
      print('warning in module solve_2d: convergence in sparse matrix solver not reached')

# compute residual without normalizing with |b|=|su2d|
   if solver_local != 'direct':
      resid=np.linalg.norm(A*phi - su)

   delta_phi=np.max(np.abs(phi-phi_org))

   phi2d=np.reshape(phi,(ni,nj))
   phi2d_org=np.reshape(phi_org,(ni,nj))

   if solver_local != 'pyamg':
      print(f"{'residual history in solve_2d: initial residual: '} {resid_init:.2e}{'final residual: ':>30}{resid:.2e}\
      {'delta_phi: ':>25}{delta_phi:.2e}")

# we return the initial residual; otherwise the solution is always satisfied (but the non-linearity is not accounted for)
   return phi2d,resid_init

def calcu(su2d,sp2d):
   if iter == 0:
      print('calcu called')
# b.c., sources, coefficients

# add sources
#  su2d=su2d+vol

# modify su & sp
   su2d,sp2d=modify_u(su2d,sp2d)
 
   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_u
   su2d=su2d+(1-urf_u)*ap2d*u2d

   return su2d,sp2d,ap2d


def vist_kom(vis2d,k2d,om2d):
   if iter == 0:
      print('vist_kom called')

   visold= vis2d
   vis2d= k2d/om2d+viscos

# modify viscosity
   vis2d=modify_vis(vis2d)

#            under-relax viscosity
   vis2d= urfvis*vis2d+(1.-urfvis)*visold

   return vis2d

def dphidx(phi_face_w,phi_face_s):

   phi_w=phi_face_w[0:-1,:]*areawx[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawx[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasx[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasx[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def dphidy(phi_face_w,phi_face_s):

   phi_w=phi_face_w[0:-1,:]*areawy[0:-1,:]
   phi_e=-phi_face_w[1:,:]*areawy[1:,:]
   phi_s=phi_face_s[:,0:-1]*areasy[:,0:-1]
   phi_n=-phi_face_s[:,1:]*areasy[:,1:]
   return (phi_w+phi_e+phi_s+phi_n)/vol

def calck(su2d,sp2d,k2d,om2d,vis2d,u2d_face_w,u2d_face_s,v2d_face_w,v2d_face_s):
# b.c., sources, coefficients 
   if iter == 0:
      print('calck_kom called')

# production term
   dudx=dphidx(u2d_face_w,u2d_face_s)
   dvdx=dphidx(v2d_face_w,v2d_face_s)

   dudy=dphidy(u2d_face_w,u2d_face_s)
   dvdy=dphidy(v2d_face_w,v2d_face_s)

   gen= (2.*(dudx**2+dvdy**2)+(dudy+dvdx)**2)
   vist=np.maximum(vis2d-viscos,1e-10)
   su2d=su2d+vist*gen*vol

   sp2d=sp2d-cmu*c_k_ML*om2d*vol  

# modify su & sp
   su2d,sp2d=modify_k(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_k
   su2d=su2d+(1-urf_k)*ap2d*k2d

   return su2d,sp2d,gen,ap2d

def compute_face_phi(phi2d,phi_bc_west,phi_bc_east,phi_bc_south,phi_bc_north,\
    phi_bc_west_type,phi_bc_east_type,phi_bc_south_type,phi_bc_north_type):
   import numpy as np

   if iter == 0:
      print('compute_face_phi called')

   phi2d_face_w=np.empty((ni+1,nj))
   phi2d_face_s=np.empty((ni,nj+1))
   phi2d_face_w[0:-1,:]=fx*phi2d+(1-fx)*np.roll(phi2d,1,axis=0)
   phi2d_face_s[:,0:-1]=fy*phi2d+(1-fy)*np.roll(phi2d,1,axis=1)

# west boundary 
   phi2d_face_w[0,:]=phi_bc_west
   if phi_bc_west_type == 'n':
# neumann
      phi2d_face_w[0,:]=phi2d[0,:]

# east boundary 
   phi2d_face_w[-1,:]=phi_bc_east
   if phi_bc_east_type == 'n':
# neumann
      phi2d_face_w[-1,:]=phi2d[-1,:]
      phi2d_face_w[-1,:]=phi2d_face_w[-2,:]

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

def calcom(su2d,sp2d,om2d,gen):
   if iter == 0:
      print('calcom called')


#--------production term
   su2d=su2d+c_omega_1*gen*vol

#--------dissipation term
   sp2d=sp2d-c_omega_2_ML*om2d*vol

# modify su & sp
   su2d,sp2d=modify_om(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_omega
   su2d=su2d+(1-urf_omega)*ap2d*om2d

   return su2d,sp2d,ap2d

######################### the execution of the code starts here #############################

########### grid specification ###########
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

# initialize geometric arrays

vol=np.zeros((ni,nj))
areas=np.zeros((ni,nj+1))
areasx=np.zeros((ni,nj+1))
areasy=np.zeros((ni,nj+1))
areaw=np.zeros((ni+1,nj))
areawx=np.zeros((ni+1,nj))
areawy=np.zeros((ni+1,nj))
areaz=np.zeros((ni,nj))
as_bound=np.zeros((ni))
an_bound=np.zeros((ni))
aw_bound=np.zeros((nj))
ae_bound=np.zeros((nj))
az_bound=np.zeros((ni,nj))
fx=np.zeros((ni,nj))
fy=np.zeros((ni,nj))

# default values
# boundary conditions for u
u_bc_west=np.ones(nj)
u_bc_east=np.zeros(nj)
u_bc_south=np.zeros(ni)
u_bc_north=np.zeros(ni)

u_bc_west_type='d'
u_bc_east_type='n'
u_bc_south_type='d'
u_bc_north_type='d'

setup_case()

print_indata()

areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,aw_bound,ae_bound,as_bound,an_bound,dist=init()


# initialization
u2d=np.ones((ni,nj))*1e-20
k2d=np.ones((ni,nj))
vis2d=np.ones((ni,nj))*viscos
om2d=np.ones((ni,nj))

aw2d=np.ones((ni,nj))*1e-20
ae2d=np.ones((ni,nj))*1e-20
as2d=np.ones((ni,nj))*1e-20
an2d=np.ones((ni,nj))*1e-20
al2d=np.ones((ni,nj))*1e-20
ah2d=np.ones((ni,nj))*1e-20
ap2d=np.ones((ni,nj))*1e-20
ap2d_vel=np.ones((ni,nj))*1e-20
su2d=np.ones((ni,nj))*1e-20
sp2d=np.ones((ni,nj))*1e-20
ap2d=np.ones((ni,nj))*1e-20

# initialize
u2d,k2d,om2d,vis2d=modify_init(u2d,k2d,om2d,vis2d)

# read data from restart
if restart:
   u2d,k2d,om2d,vis2d,ap2d_vel= read_restart_data()

iter=0

vist_ML_5200 = np.loadtxt('vist_pref-from-vist-diffusion-pinn-5200-plus-units-load-5-cells-half-channel-sigmoid-fist-and-2nd-and-3rd-layer-fixed.txt')

vist_ML_5200 = vist_ML_5200/5200

data_5200= np.loadtxt('/chalmers/users/lada/pythons-rans-code-RANS/channel-5200-half-channel/y_u_k_om_uv_5200-RANS-half-channel.txt')
y_5200 = data_5200[:,0]
y1d= yp2d[0,:]

prand_k_ML_5200 = np.loadtxt('prand_k_5200-plus-units-from-balance.txt')

print('prand_k_ML_5200',prand_k_ML_5200)

vist_ML = np.interp(y1d, y_5200, vist_ML_5200)
prand_k = np.interp(y1d, y_5200, prand_k_ML_5200)


# make it 2D
prand_k = np.repeat(prand_k[None,:], repeats=ni, axis=0)

print('vist_ML',vist_ML)

vist = k2d/om2d

print('prand_k',prand_k)

c_k_ML_5200= np.loadtxt('c_k_pred_5200-plus-units-from-balance.txt')
c_k_ML = np.interp(y1d, y_5200, c_k_ML_5200)
# make it 2D
c_k_ML = np.repeat(c_k_ML[None,:], repeats=ni, axis=0)

c_omega_2_ML_5200 = np.loadtxt('c_omega_2_pred_5200-plus-units-from-balance.txt')
c_omega_2_ML = np.interp(y1d, y_5200, c_omega_2_ML_5200)
# make it 2D
c_omega_2_ML = np.repeat(c_omega_2_ML[None,:], repeats=ni, axis=0)

u2d_face_w,u2d_face_s=compute_face_phi(u2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north,\
    u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)

v2d_face_w=np.zeros((ni+1,nj))
v2d_face_s=np.zeros((ni,nj+1))

# compute vis
urf_temp=urfvis # no under-relaxation
urfvis=1
vis2d=vist_kom(vis2d,k2d,om2d)
urfvis=urf_temp

######################### start of global iteration process #############################

for iter in range(0,abs(maxit)):

      start_time_iter = time.time()
# coefficients for velocities
      start_time = time.time()

# compute coefficient matrix
      aw2d,ae2d,as2d,an2d,su2d,sp2d=coeff(vis2d,np.ones((ni,nj)))
# boundary conditions for u2d
      su2d,sp2d=bc(su2d,sp2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north, \
                   u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)
      su2d,sp2d,ap2d=calcu(su2d,sp2d)
      ap2d_vel = ap2d

      u2d,residual_u=solve_2d(u2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_u,nsweep_u,solver_u)
#     residual_u=0

      k2d=np.maximum(k2d,1e-10)
      print(f"{'time k: '}{time.time()-start_time:.2e}")

      vis2d=vist_kom(vis2d,k2d,om2d)
# coefficients
      start_time = time.time()
      vist = k2d/om2d
      aw2d,ae2d,as2d,an2d,su2d,sp2d=coeff(vis2d,prand_k)

# case values
      u2d_face_w,u2d_face_s=compute_face_phi(u2d,u_bc_west,u_bc_east,u_bc_south,u_bc_north,\
        u_bc_west_type,u_bc_east_type,u_bc_south_type,u_bc_north_type)

# k
# boundary conditions for k2d
      su2d,sp2d=bc(su2d,sp2d,k_bc_west,k_bc_east,k_bc_south,k_bc_north, \
                   k_bc_west_type,k_bc_east_type,k_bc_south_type,k_bc_north_type)
      su2d,sp2d,gen,ap2d=calck(su2d,sp2d,k2d,om2d,vis2d,u2d_face_w,u2d_face_s,v2d_face_w,v2d_face_s)

      k2d,residual_k=solve_2d(k2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_k,nsweep_kom,solver_turb)

      k2d=np.maximum(k2d,1e-10)
      print(f"{'time k: '}{time.time()-start_time:.2e}")

      start_time = time.time()
# omega
# boundary conditions for om2d
      aw2d,ae2d,as2d,an2d,su2d,sp2d=coeff(vis2d,prand_omega)
      su2d,sp2d=bc(su2d,sp2d,om_bc_west,om_bc_east,om_bc_south,om_bc_north,\
                   om_bc_west_type,om_bc_east_type,om_bc_south_type,om_bc_north_type)
      su2d,sp2d,ap2d= calcom(su2d,sp2d,om2d,gen)

      aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d=fix_omega()

      om2d,residual_om=solve_2d(om2d,aw2d,ae2d,as2d,an2d,su2d,ap2d,convergence_limit_om,nsweep_kom,solver_turb)
#     residual_om=0
      om2d=np.maximum(om2d,1e-10)

      print(f"{'time om: '}{time.time()-start_time:.2e}")

      umax=np.max(u2d.flatten())

      resmax=np.max([residual_u ,residual_k,residual_om])

      print(f"\n{'--iter:'}{iter:d}, {'max residual:'}{resmax:.2e}, {'u:'}{residual_u:.2e}, {'k:'}{residual_k:.2e}, {'om:'}{residual_om:.2e}\n")

      print(f"\n{'monitor iteration:'}{iter:4d}, {'u:'}{u2d[imon,jmon]: .2e}, {'k:'}{k2d[imon,jmon]: .2e}, {'om:'}{om2d[imon,jmon]: .2e}\n")

      vismax=np.max(vis2d.flatten())/viscos
      umax=np.max(u2d.flatten())
      ommin=np.min(om2d.flatten())
      kmin=np.min(k2d.flatten())

      print(f"\n{'---iter: '}{iter:2d}, {'umax: '}{umax:.2e},{'vismax: '}{vismax:.2e}, {'kmin: '}{kmin:.2e}, {'ommin: '}{ommin:.2e}\n")

      print(f"{'time one iteration: '}{time.time()-start_time_iter:.2e}")

      if resmax < sormax:

         break

######################### end of global iteration process #############################
      
# save data for restart
if save:
   save_data(u2d,k2d,om2d,vis2d,ap2d_vel)

print('program reached normal stop')

