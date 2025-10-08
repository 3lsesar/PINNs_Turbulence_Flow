
############################ 10000
# load DNS data
# %     y/h             y+             U+           u'+           v'+          w'+           uv'+         dU/dy+
DNS_mean=np.genfromtxt("/chalmers/users/lada/DNS_channel_re10000/P10k.txt",comments="%")
y_DNS=DNS_mean[:,0];
yplus_DNS=DNS_mean[:,1];
u_DNS=DNS_mean[:,2];
u2_DNS=DNS_mean[:,3]**2;
v2_DNS=DNS_mean[:,4]**2;
w2_DNS=DNS_mean[:,5]**2;
uv_DNS=DNS_mean[:,6];
dudy_DNS= np.gradient(u_DNS,yplus_DNS)
k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)
# %      y/h            y+         dissip        prod         p-strain       p-diff        T-diff        V-diff
DNS_uu = np.genfromtxt("/chalmers/users/lada/DNS_channel_re10000/P10k.uu.txt",comments="%")
eps_DNS_uu = abs(DNS_uu[:,2])
visc_diff_uu =  DNS_uu[:,7]

DNS_vv = np.genfromtxt("/chalmers/users/lada/DNS_channel_re10000/P10k.vv.txt",comments="%")
eps_DNS_vv = abs(DNS_vv[:,2])
visc_diff_vv =  DNS_vv[:,7]

DNS_ww = np.genfromtxt("/chalmers/users/lada/DNS_channel_re10000/P10k.uu.txt",comments="%")
eps_DNS_ww = abs(DNS_ww[:,2])
visc_diff_ww =  DNS_ww[:,7]

diss_DNS= (eps_DNS_uu +eps_DNS_vv +eps_DNS_ww)/2
visc_diff = (visc_diff_uu +visc_diff_vv +visc_diff_ww)/2


# fix wall
diss_DNS[0]=diss_DNS[1]
k_DNS[0]=k_DNS[1]

dudy_DNS  = np.gradient(u_DNS,yplus_DNS)

pk_DNS = -uv_DNS*dudy_DNS
dkdy=np.gradient(k_DNS,yplus_DNS)
d2kdy2=np.gradient(dkdy,yplus_DNS)
diss_DNS_org = diss_DNS
diss_DNS = np.maximum(diss_DNS-visc_diff,0)

tau_DNS = k_DNS/diss_DNS

dudy_squared_DNS  = dudy_DNS**2
dudy_squared_scaled_DNS  = dudy_DNS**2*tau_DNS**2



vist_DNS=abs(uv_DNS)/dudy_DNS

omega_DNS_from_vist=k_DNS/vist_DNS


