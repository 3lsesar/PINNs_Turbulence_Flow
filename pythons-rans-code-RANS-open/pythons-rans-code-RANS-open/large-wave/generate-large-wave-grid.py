import numpy as np
import matplotlib.pyplot as plt
import sys

plt.interactive(True)

plt.close('all')


# Number of cells: 
ni=300
dx=0.01
x_wall=np.linspace(0, (ni+1)*dx, ni+1)
y_wall=np.zeros(ni+1)
a=0.28
x0=5.907
delta=1.8219
y_wall=a*np.exp(-((7*x_wall-x0)**2)/(2*delta**2))
# make y = 0 at inlet
y_wall = y_wall - y_wall[0]


# set y


viscos=1/550
nj=150
y2d=np.zeros((ni+1,nj+1))
yy=np.zeros((ni+1,nj+1))
yfac=1.05 # stretching
yc=np.zeros(nj+1)
dx=x_wall[1]-x_wall[0]
for i in range(0,ni+1):
   dy= 0.1
   yc[0]=y_wall[i]
   for j in range(1,nj+1):
      yc[j]=yc[j-1]+dy
      if j < 70:
         dy=yfac*dy

   ymax_scale=yc[-1]
#  print('i,ymax_scale',i,ymax_scale)
# check yfac
   #if i%30 == 0:
     #print('i,yfcac',i,np.diff(yy[i,1:])/np.diff(yy[i,:-1]))

#  print('i,ymax_scale',i,ymax_scale)

# cell faces
   for j in range(0,nj+1):
      y2d[i,j]=y_wall[i]+yc[j]/ymax_scale*(1-y_wall[i])

#  print('y2d[i,-1]',y2d[i,-1])
yplus=0.5*y2d[0,1]/viscos
print('yplus',yplus)

# make it 2D
x2d=np.repeat(x_wall[:,None], repeats=nj+1, axis=1)

# add 50 cells upstream
dx_in=x2d[1,0]-x2d[0,0]
x2d_old=x2d
y2d_old=y2d

di=50
ni=ni+di
del x2d, y2d

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))
x2d[di:,:]=x2d_old
y2d[di:,:]=y2d_old
y2d[:di,:]=y2d_old[0,:]

for i in range (di,0,-1):
   x2d[i,:]=x2d[i+1,:]-dx_in

x2d[0,:]=x2d[1,:]-dx_in



#%%%%%%%%%%%%%%%%%%%%% grid
fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
for i in range(0,ni+1,5):
   plt.plot(x2d[i,:],y2d[i,:])
plt.plot(x2d[-1,:],y2d[-1,:])

for j in range(0,nj+1,5):
   plt.plot(x2d[:,j],y2d[:,j])
plt.plot(x2d[:,-1],y2d[:,-1])

plt.savefig('grid.png')

x2d_org=np.copy(x2d)
y2d_org=np.copy(y2d)

print('y+',0.5*yc[1]/viscos)
# make it 2D
y2d=np.append(y2d,nj)
np.savetxt('y2d.dat', y2d)

x2d=np.append(x2d,ni)
np.savetxt('x2d.dat', x2d)

# check it
datay= np.loadtxt("y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

y2=np.zeros((ni+1,nj+1))
y2=np.reshape(y,(ni+1,nj+1))

datax= np.loadtxt("x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])

x2=np.zeros((ni+1,nj+1))
x2=np.reshape(x,(ni+1,nj+1))

