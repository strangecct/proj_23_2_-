from scipy.fftpack import fft,ifft
from vmdpy import VMD
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']#['STsong']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

situation1 = pd.read_csv('breaker/fault1/backup_147.xlsx', header=None)
#数据的预处理
situation1 = situation1.iloc[:,:]
situation1 = situation1.drop(0, axis=1)
situation1.columns = ['a', 'b', 'c', 'd','e', 'f', 'g', 'h','i','j','l','m']
situation1.index = np.arange(200_000)
situation1 = situation1.replace([np.inf, -np.inf], np.nan).fillna(situation1.mean()) # 缺失值处理

situation2 = pd.read_csv('breaker/fault1/backup_147.xlsx', skiprows= 1, header=None)
situation2 = situation2.iloc[:,:]
situation2 = situation2.drop(0, axis=1)
situation1.columns = ['a', 'b', 'c', 'd','e', 'f', 'g', 'h','i','j','l','m']
situation2.index = np.arange(200_000)
situation2 = situation2.replace([np.inf, -np.inf], np.nan).fillna(situation2.mean()) # 缺失值处理

alpha =  972 # moderate bandwidth constraint
tau = 0.  # noise-tolerance (no strict fidelity enforcement)
K = 6  # 3 modes
DC = 0  # no DC part imposed
init = 1  # initialize omegas uniformly
tol = 1e-7

u, *_ = VMD(situation2.iloc[:,1].values, alpha, tau, K, DC, init, tol)
np.savetxt('vmd_i.xls', u, delimiter=',')
# u - the collection of decomposed modes
# u_hat - spectra of the modes
# omega - estimated mode center - frequencies



# plt.style.use('seaborn')
fig1 ,axes = plt.subplots(1,1,figsize=(6,6))
axes.plot(u.T,linewidth= 0.1)
axes.set_title('Decomposed modes')
plt.savefig("VMD分解图")

fig2 ,axes = plt.subplots(1,1,figsize=(6,6))
axes.scatter(np.arange(situation2.shape[0]),situation2.iloc[:,1], s = 0.1)
axes.set_title('Original input signal')
plt.savefig("未处理图像")


fig3 ,axes = plt.subplots(K,1,figsize=(8,16/3),sharex=False)
fig3.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
for i in range(K):
    axes[i].scatter(np.arange(len(u[i, :])),u[i, :], s=0.04/(i + 1), c='b')
    axes[i].set_ylabel('IMF{}'.format(i + 1))
    axes[i].set_xticks([0, 2_000, 4_000, 6_000, 8_000, 10_000])
    axes[i].set_xticklabels([0, 0.004, 0.008, 0.012, 0.016, 0.02])
    #axes[i].set_ylim([0, 0.5])
    #axes[i].xaxis.set_ticks([0, 0.2, 0.4, 0.6, 0.8])
axes[5].set_xlabel('时间($s$)', fontsize=10)
plt.savefig("VMD_6_972_ 时域")
# plt.show()
# 中心模态的图像

fig4 ,axes = plt.subplots(K,1,figsize=(8,16/3))
fig4.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
for i in range(K):
    a = abs(fft(u[i, :])).shape
    # axes[i].plot(abs(fft(u[i, :])[:30000]),linewidth = 0.1,c = 'r')
    axes[i].plot(abs(fft(u[i, :])[:5000]), linewidth=0.1, c='r')
    axes[i].set_ylabel('IMF{}'.format(i + 1))
    # axes[i].set_xticks([0, 5_000, 10_000, 15_000, 20_000, 25_000])
    # axes[i].set_xticklabels([0, 2, 4, 6, 8, 10])
    axes[i].set_xticks([0, 1_000, 2_000, 3_000, 4_000, 5_000])
    # axes[i].set_xticklabels([0, 0.004, 0.008, 0.012, 0.016, 0.02])

#axes[5].set_xlabel('时间 ($10^3/s$)', fontsize = 10)
axes[5].set_xlabel('频率', fontsize = 10)
plt.savefig("VMD_6_972")
plt.show()