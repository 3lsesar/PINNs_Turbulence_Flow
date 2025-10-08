import torch.nn as nn

class ThePredictionMachine(nn.Module):

    def __init__(self):

        super(ThePredictionMachine, self).__init__()

        self.input   = nn.Linear(3, 50)
        self.hidden1 = nn.Linear(50, 50)
        self.hidden2 = nn.Linear(50, 3)

    def forward(self, x):

        x = nn.functional.relu(self.input(x))
        x = nn.functional.relu(self.hidden1(x))
        x = self.hidden2(x)


        return x

def calc_earsm(k2d,om2d,u2d_face_w,u2d_face_s,v2d_face_w,v2d_face_s,uu2d,uv2d,vv2d,vis2d_earsm):

    import torch
    import sys
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
    from random import randrange
    from joblib import dump, load

    global  dudy2_scaled_min, dudy2_scaled_max, N2_min, N2_max, scaler_dudy2, scaler_N2 , neural_net

    if iter == 0:
      print('calc_earsm called')
# load pytorch model
      folder='/chalmers/users/lada/noback/pycalc-les/pytorch-diffusor/'

      filename=str(folder)+'model-earsm-DNS-N2-with-d2kdy2-and-dudy2-2-hidden-1-yplus-4200-dudy-scale-with-k-eps-units-channel-5200-LRR.pth'
      neural_net = torch.load(filename)
      print('model',neural_net)
      scaler_dudy2 = load(str(folder)+'model-earsm-DNS-N2-with-d2kdy2-and-dudy2_scaler-dudy2-2-hidden-1-yplus-4200-dudy-scale-with-k-eps-units-channel-5200-LRR.bin')
      scaler_N2 = load(str(folder)+'model-earsm-DNS-N2-with-d2kdy2-and-dudy2_scaler-N2-with-d2kdy2-2-hidden-1-yplus-4200-dudy-scale-with-k-eps-units-channel-5200-LRR.bin')

      dudy2_scaled_min, dudy2_scaled_max, b1_min, b1_max, b2_min, b2_max, b4_min, b4_max, N2_min, N2_max  = np.loadtxt(str(folder)+'min-max-model-earsm-DNS-N2-with-d2kdy2-and-dudy2-2-hidden-1-yplus-4200-dudyale-with-k-eps-units-channel-5200-LRR.txt')
      #dudy2_scaled_min, dudy2_scaled_max, b1_min, b1_max, b4_min, b4_max, N2_min, N2_max = np.loadtxt(str(folder)+'min-max-model-earsm-DNS-N2-with-d2kdy2-and-dudy2-2-hidden-1-yplus-4200-dudyale-with-k-eps-units-channel-5200-LRR.txt')



    dudx=dphidx(u2d_face_w,u2d_face_s)
    dudy=dphidy(u2d_face_w,u2d_face_s)

    dvdx=dphidx(v2d_face_w,v2d_face_s)
    dvdy=dphidy(v2d_face_w,v2d_face_s)

    diss=0.09*k2d*om2d

#   np.savetxt('y_u_k_ed_dudy.txt', np.c_[yp2d[0,:],u2d[0,:], k2d[0,:], diss[0,:], dudy[0,:]])
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
    two_S=(s11**2+s12**2+s21**2+s22**2)  # s_ij s_ji = s_ij s_ij
      
    vist = vis2d_earsm - viscos
    uu_tot = uu2d - vist*dudx
    vv_tot = vv2d- vist*dvdy
    uv_tot = uv2d - vist*(dudy+dvdx)
    Pk = -uu_tot*dudx-uv_tot*(dudy+dvdx)-vv_tot*dvdy
    N2 = (Pk/diss)**2

    dudy_squared  = dudy**2
    dudy_squared_scaled = dudy_squared*ttau**2

# limit min/max
# count values larger/smaller than max/min
    dudy2_min_number= (dudy_squared_scaled < dudy2_scaled_min).sum()
    dudy2_max_number= (dudy_squared_scaled > dudy2_scaled_max).sum()
    print('dudy2_min_number',dudy2_min_number)
    print('dudy2_max_number',dudy2_max_number)

    N2_min_number= (N2 < N2_min).sum()
    N2_max_number= (N2 > N2_max).sum()
    print('N2_min_number',N2_min_number)
    print('N2_max_number',N2_max_number)

# set limits
    dudy_squared_scaled=np.minimum(dudy_squared_scaled,dudy2_scaled_max)
    dudy_squared_scaled=np.maximum(dudy_squared_scaled,dudy2_scaled_min)

    N2=np.minimum(N2,N2_max)
    N2=np.maximum(N2,N2_min)

    dudy_squared_scaled = dudy_squared_scaled.reshape(-1,1)
    N2= N2.reshape(-1,1)
# use standard scaler
    X=np.zeros((len(N2),2))
    X[:,0] = scaler_dudy2.transform(dudy_squared_scaled)[:,0]
    X[:,1] = scaler_N2.transform(N2)[:,0]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    preds = neural_net(X_tensor)
#transform from tensor to numpy
    c_NN = preds.detach().numpy()

    beta1=c_NN[:,0]
    beta2=c_NN[:,1]
    beta4=c_NN[:,2]

#   uu=2/3*rk\
    uu=rk*(beta4*(s12*om21-om12*s21)+beta2*(s11**2+s12**2-two_S/3))   # +rk*beta1*s11 # this is included via vis_earsm
#   vv=2/3*rk  \
    vv=rk*(beta4*(s21*om12-om21*s12)+beta2*(s22**2+s12**2-two_S/3))   # +rk*beta1*s22+  # this is included via vis_earsm
    uv = rk*beta4*(s11*om12-om12*s22) #+rk*beta1*s12 # this is included via vis_earsm

    vis2d_earsm_old= vis2d_earsm
    vis2d_earsm=-0.5*rk*beta1*ttau+viscos

    uu2d = urfvis*uu+(1-urfvis)*uu2d
    uv2d = urfvis*uv+(1-urfvis)*uv2d
    vv2d = urfvis*vv+(1-urfvis)*vv2d
    vis2d_earsm=urfvis*vis2d_earsm+(1-urfvis)*vis2d_earsm_old

    return uu2d,vv2d,uv2d,vis2d_earsm
