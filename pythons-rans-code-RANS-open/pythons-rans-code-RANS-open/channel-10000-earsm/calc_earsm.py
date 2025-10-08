def calc_earsm(k2d,om2d,u2d_face_w,u2d_face_s,v2d_face_w,v2d_face_s,uu2d,uv2d,vv2d,ww2d,vis2d_earsm):
# EARSM without NN


   global p1, p2, if_true, if_false, str1, vor, ttau, un, uu, vv, uv, beta4

   if iter == 0:
      print('standard calc_earsm called')



   dudx=dphidx(u2d_face_w,u2d_face_s)
   dudy=dphidy(u2d_face_w,u2d_face_s)


   dvdx=dphidx(v2d_face_w,v2d_face_s)
   dvdy=dphidy(v2d_face_w,v2d_face_s)

   diss=0.09*k2d*om2d

   rk=k2d
   ttau=np.maximum(rk/diss,6*(viscos/diss)**0.5)
   om12=ttau*0.5*(dudy-dvdx)
   om21=-om12
   om22=0.
   om11=0.
   s11=ttau*dudx
   s12=ttau*0.5*(dudy+dvdx)
   s21=s12
   s22=ttau*dvdy
   vor=(-2*om12**2)  # om_ij om_ji = om_12*om_21 + om_21*om_12 = -2*om_12*om_12
   str1=(s11**2+s12**2+s21**2+s22**2)  # s_ij s_ji = s_ij s_ij
   II_S=(s11**2+s12**2+s21**2+s22**2)  # s_ij s_ji = s_ij s_i

   A1 = 1.54
   A2 = 0.37
   A3 = 1.45
   A4 = 2.89

   p1=(1/27*A3**2+(A1*A4/6-2/9*A2**2)*str1-2/3*vor)*A3
   p2=p1**2-(A3**2/9+(A1*A4/3+2/9*A2**2)*str1+2/3*vor)**3

   if_true  = A3/3+(p1+p2**0.5)**(1/3)+np.sign(p1-p2**0.5)*(abs(p1-p2**0.5))**(1/3)
   if_false = A3/3+2*(p1**2-p2)**(1/6)*np.cos(1/3*np.arccos(p1/(np.sqrt(p1**2-p2))))


   un = np.where(p2>0,if_true,if_false)

   Q = un**2-2*vor-2/3*A2**2*str1

   beta1=-A1*un/Q
   beta2=2*A1*A2/Q
   beta4=-A1/Q

   uu=rk*(beta4*(s12*om21-om12*s21)+beta2*(s11**2+s12**2-II_S/3)) 
   vv=rk*(beta4*(s21*om12-om21*s12)+beta2*(s22**2+s12**2-II_S/3))
   ww=-rk*beta2*II_S/3
   uv = rk*beta4*(s11*om12-om12*s22)

   vis2d_earsm_old= vis2d_earsm
   vis2d_earsm=-0.5*rk*beta1*ttau+viscos

   uu2d = urfvis*uu+(1-urfvis)*uu2d
   uv2d = urfvis*uv+(1-urfvis)*uv2d
   vv2d = urfvis*vv+(1-urfvis)*vv2d
   vis2d_earsm=urfvis*vis2d_earsm+(1-urfvis)*vis2d_earsm_old

   if iter % 100 == 0:
      viss=np.zeros((ni,nj+1))
      uv_turb_s=np.zeros((ni,nj+1))
      vist = vis2d_earsm - viscos
      viss[:,0:-1]=fy*vist+(1-fy)*np.roll(vist,1,axis=1)
      uv2d_face_w,uv2d_face_s=compute_face_phi(uv2d,uv_bc_west,uv_bc_east,uv_bc_south,uv_bc_north,\
           uv_bc_west_type,uv_bc_east_type,uv_bc_south_type,uv_bc_north_type)
      dudy_s = (u2d[:,1:]-u2d[:,0:-1])/np.diff(yp2d,axis=1)
      uv_turb_s[:,1:-1] = uv2d_face_s[:,1:-1] - viss[:,1:-1]*dudy_s
      uv_turb_s[:,0] = uv_turb_s[:,1]
      uv_turb_s[:,-1] = uv_turb_s[:,-2]
      np.save('uv2d_s_saved', uv_turb_s)

      np.save('beta1_saved', beta1)
      np.save('beta4_saved', beta4)
      np.save('diss_saved', diss)


   return uu2d,vv2d,ww2d,uv2d,vis2d_earsm

