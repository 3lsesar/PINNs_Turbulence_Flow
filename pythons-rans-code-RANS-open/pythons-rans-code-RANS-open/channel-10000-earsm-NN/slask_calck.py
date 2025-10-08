def calck(su2d,sp2d,k2d,om2d,vis2d,u2d_face_w,u2d_face_s,v2d_face_w,v2d_face_s):
# b.c., sources, coefficients 
   if iter == 0:
      print('calck_kom earsm called')

# production term
   dudx=dphidx(u2d_face_w,u2d_face_s)
   dvdx=dphidx(v2d_face_w,v2d_face_s)

   dudy=dphidy(u2d_face_w,u2d_face_s)
   dvdy=dphidy(v2d_face_w,v2d_face_s)


   if earsm:
      vist = vis2d_earsm - viscos
      uu_tot = uu2d# - vist*dudx
      vv_tot = vv2d#- vist*dvdy
      uv_tot = uv2d - vist*(dudy+dvdx)
      gen = -uu_tot*dudx-uv_tot*(dudy+dvdx)-vv_tot*dvdy
      su2d=su2d+gen*vol
   else:
      gen= (2.*(dudx**2+dvdy**2)+(dudy+dvdx)**2)
      vist=np.maximum(vis2d-viscos,1e-10)
      su2d=su2d+vist*gen*vol

   sp2d=sp2d-cmu*om2d*vol

# modify su & sp
   su2d,sp2d=modify_k(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_k
   su2d=su2d+(1-urf_k)*ap2d*k2d

   return su2d,sp2d,gen,ap2d

