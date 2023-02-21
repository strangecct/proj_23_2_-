
from vmdpy import VMD
from PyEMD import EEMD, EMD
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

matplotlib.rcParams['font.sans-serif'] = ['SimSun']#['STsong']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def get_data(file_path, skip = 1):
    data_ori = pd.read_excel(file_path, skiprows= skip, header=None)
    data_ori.drop(0, axis=1)
    data_len = data_ori.shape[0]
    data_ori.index = np.arange(data_len)
    data_ori = data_ori.replace([np.inf, -np.inf], np.nan).fillna(data_ori.mean())  # 缺失值处理
    return data_ori

def vmd_proc(file_path, save_name, csv_channel = 3, alp = 5_00, k = 6):
    data_ori = get_data(file_path)
    alpha = alp  # moderate bandwidth constraint
    tau = 0.  # noise-tolerance (no strict fidelity enforcement)
    K = k  # 3 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7
    u, *_ = VMD(data_ori.iloc[:, csv_channel].values, alpha, tau, K, DC, init, tol)
    # u - the collection of decomposed modes
    # u_hat - spectra of the modes
    # omega - estimated mode center - frequencies
    fig1, axes = plt.subplots(K, 1, figsize=(8, 16 / 3), sharex=False)
    fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    for i in range(K):
        axes[i].scatter(np.arange(len(u[i, :])), u[i, :], s=0.08 / (i + 1), c='b')
        axes[i].set_ylabel('IMF{}'.format(i + 1))
        # axes[i].set_ylim([-0.5, 0.5])
        # axes[i].yaxis.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    plt.savefig("./pig_save/" + f"VMD_time_{K}_{alpha}_" + save_name)
    plt.show()
    # 中心模态的图像
    # plt.style.use('seaborn')
    fig2, axes = plt.subplots(K, 1, figsize=(8, 16 / 3))
    fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    for i in range(K):
        a = abs(fft(u[i, :])).shape
        axes[i].plot(abs(fft(u[i, :])[:5000]), linewidth=0.1)
        axes[i].set_ylabel('IMF{}'.format(i + 1))
    plt.savefig("./pig_save/" + f"VMD_freq_{K}_{alpha}_" + save_name)
    plt.show()

def emd_proc(file_path, save_name, csv_channel = 3):
    data_ori = get_data(file_path)
    data_len = data_ori.shape[0]
    s = np.array(data_ori.iloc[:, csv_channel])
    t = np.arange(data_ori.shape[0])
    # Execute EEMD on S
    IMF = EMD().emd(s, t)
    N = IMF.shape[0]

    fig1, axes = plt.subplots(6, 1, figsize=(8, 8), sharex=False)
    fig1.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    for n, imf in enumerate(IMF):
        axes[n].scatter(t, imf, s=0.1, c='b')
        axes[n].set_ylabel('IMF{}'.format(n + 1))
        # axes[n].set_ylim([-0.5, 0.5])
        # axes[n].yaxis.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
        # axes[n].set_xticks([0, 10_000, 20_000, 30_000, 40_000, 10_000])
        # axes[n].set_xticklabels([0, 0.02, 0.04, 0.06, 0.08, 0.1])
        axes[n].set_xticks([0, data_len/5, data_len * 0.4, data_len * 0.6, data_len * 0.8, data_len])
        axes[n].set_xticklabels([0, 0.004, 0.008, 0.012, 0.016, 0.02])
        if (n == 5):
            break
    # axes[0].set_title("EMD")
    axes[5].set_xlabel('时间($/s$)', fontsize=10)
    plt.savefig("./pig_save/" + f"EMD_time_" + save_name)
    plt.show()
    # 中心模态的图像
    fig2, axes = plt.subplots(6, 1, figsize=(8, 8))
    fig2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    for i, imf in enumerate(IMF):
        a = abs(fft(imf)).shape
        axes[i].plot(abs(fft(imf))[5000:], linewidth=0.1, c='r')
        axes[i].set_ylabel('IMF{}'.format(i + 1))
        # axes[i].set_xticks([0, 5_000, 10_000, 15_000, 20_000, 25_000])
        # axes[i].set_xticklabels([0, 2, 4, 6, 8, 10])
        axes[i].set_xticks([0, 1_000, 2_000, 3_000, 4_000, 50_00])
        # axes[i].set_xticklabels([0, 0.004, 0.008, 0.012, 0.016, 0.02])
        if (i == 5):
            break
    # axes.set_ylabel('幅值($m/s^2$)', fontsize = 10)
    # axes[5].set_xlabel('Time ($10^3/s$)', fontsize = 10)
    axes[5].set_xlabel('频率', fontsize=10)
    plt.savefig("./pig_save/" + f"EMD_freq_" + save_name)
    plt.show()

if __name__ == "__main__":
    file_dir = './switch7to8/'
    file_dir = './all_datail/'
    file_name = '7_8_3.csv'
    file_name = "5_4.csv"
    # plot_original(file_dir + file_name, "channel_3",3)
    # plot_original(file_dir + file_name, "channel_5",5)

    vmd_proc(file_dir + file_name, "switch5_4",3,alp = 5_000, k = 6)
    # emd_proc(file_dir + file_name, "switch5_4")



# fig1 ,axes = plt.subplots(2,1,figsize=(8,8))
# axes[1].plot(u.T,linewidth= 0.1)
# axes[1].set_title('Decomposed modes')
#
# axes[0].scatter(np.arange(data_site1.shape[0]),data_site1.iloc[:,0], s = 0.1)
# axes[0].set_title('Original input signal')



