import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from random import randrange
from joblib import dump, load


class ThePredictionMachine(nn.Module):

    def __init__(self):
        
        super(ThePredictionMachine, self).__init__()

# 2 hidden layers
        self.input   = nn.Linear(2, 50) # axis 0: dimension of X; axis 1: number of neurons
        self.hidden1 = nn.Linear(50, 50)
        self.hidden2 = nn.Linear(50, 3) # axis 1: dimension of y; axis 0: number of neurons




    def forward(self, x):
        x = nn.functional.relu(self.input(x))
        x = nn.functional.relu(self.hidden1(x))
        x = self.hidden2(x)

        return x


def modify_init(u2d,v2d,k2d,om2d,vis2d):
   
# set inlet field in entre domain
#  u2d=np.repeat(u_bc_west[None,:], repeats=ni, axis=0)
   k2d=np.ones((ni,nj))
   om2d=np.ones((ni,nj))

   vis2d=k2d/om2d+viscos

   return u2d,v2d,k2d,om2d,vis2d, dist

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
      np.savetxt('u-iteration.dat', np.c_[iter,u2d[-1,5],u2d[-1,10],u2d[-1,20],u2d[-1,30],u2d[-1,40],\
           u2d[-1,50],u2d[-1,60]])
   else:
      with open('u-iteration.dat','ab') as f:
         np.savetxt(f,np.c_[iter,u2d[-1,5],u2d[-1,10],u2d[-1,20],u2d[-1,30],u2d[-1,40],\
           u2d[-1,50],u2d[-1,60]])

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
   convs=np.zeros((ni,nj+1))
   convw=np.zeros((ni+1,nj))


   return convw

def fix_omega():

   aw2d[:,0]=0
   ae2d[:,0]=0
   as2d[:,0]=0
   an2d[:,0]=0
   ap2d[:,0]=1
   su2d[:,0]=om_bc_south


   aw2d[:,-1]=0
   ae2d[:,-1]=0
   as2d[:,-1]=0
   an2d[:,-1]=0
   ap2d[:,-1]=1
   su2d[:,-1]=om_bc_south


   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d

def modify_vis(vis2d):

   return vis2d


def fix_k():

   return aw2d,ae2d,as2d,an2d,ap2d,su2d,sp2d
