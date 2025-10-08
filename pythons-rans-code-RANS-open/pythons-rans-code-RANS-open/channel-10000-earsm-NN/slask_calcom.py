def calcom(su2d,sp2d,om2d,gen):
   if iter == 0:
      print('calcom earsm called')

   vist = vis2d - viscos
#--------production term
   if earsm:
      su2d=su2d+c_omega_1*gen*vol*om2d/k2d
   else:
      su2d=su2d+c_omega_1*gen*vist*om2d/k2d*vol

#--------dissipation term
   sp2d=sp2d-c_omega_2*om2d*vol

# modify su & sp
   su2d,sp2d=modify_om(su2d,sp2d)

   ap2d=aw2d+ae2d+as2d+an2d-sp2d

# under-relaxation
   ap2d=ap2d/urf_vel
   su2d=su2d+(1-urf_omega)*ap2d*om2d

   return su2d,sp2d,ap2d
