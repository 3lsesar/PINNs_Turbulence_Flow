def vist_kom(vis2d,k2d,om2d):
   if iter == 0:
      print('vist_kom earsm peng called')

   visold= vis2d

   vis2d= k2d/om2d+viscos

# modify viscosity
   vis2d=modify_vis(vis2d)

#            under-relax viscosity
   vis2d= urfvis*vis2d+(1.-urfvis)*visold

   return vis2d
