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

   if cyclic_x:
# A = sparse.diags([ap, -ah[:-1], -al[1:], -an[0:-nk], -as1[nk:], -ae, -aw[nj*nk:],-aw,-ae[nj*nk*(ni-1):]], \
#           [0, 1, -1, nk,-nk, nk*nj, -nk*nj, nj*nk*(ni-1), -nj*nk*(ni-1)], format='csr')
      A = sparse.diags([ap, -an[:-1], -as1[1:], -ae, -aw[nj:],-aw,-ae[nj*(ni-1):]],\
            [0, 1, -1, nj, -nj,nj*(ni-1), -nj*(ni-1)], format='csr')
   else:
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
      phi = App.solve(su, rtol=tol, x0=phi, residuals=res_amg)
      info=0
      print('Residual history in pyAMG', ["%0.4e" % i for i in res_amg])
   if solver_local == 'cgs':
      if iter == 0:
         print('solver in solve_2d: cgs')
      phi,info=linalg.cgs(A,su,x0=phi, rtol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'cg':
      if iter == 0:
         print('solver in solve_2d: cg')
      phi,info=linalg.cg(A,su,x0=phi, rtol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'gmres':
      if iter == 0:
         print('solver in solve_2d: gmres')
      phi,info=linalg.gmres(A,su,x0=phi, rtol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'qmr':
      if iter == 0:
         print('solver in solve_2d: qmr')
      phi,info=linalg.qmr(A,su,x0=phi, rtol=tol, atol=tol_conv,  maxiter=nmax)  # good
   if solver_local == 'lgmres':
      if iter == 0:
         print('solver in solve_2d: lgmres')
      phi,info=linalg.lgmres(A,su,x0=phi, rtol=tol, atol=tol_conv,  maxiter=nmax)  # good
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
