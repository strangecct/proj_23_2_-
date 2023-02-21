import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pandas as pd
from scipy.fftpack import fft,ifft
from plt_decompose import get_data


matplotlib.rcParams['font.sans-serif'] = ['SimSun']#['STsong']
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



def plot_original(file_path, save_name,csv_channel = 4):
    data_ori = get_data(file_path)
    data_len = data_ori.shape[0]
    print(data_len)

    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=0.8)
    axes.plot(np.arange(data_len), data_ori.iloc[:, csv_channel].values, linewidth=0.15)

    axes.set_xticks([0, data_len/5, data_len * 0.4, data_len * 0.6, data_len * 0.8, data_len])
    # axes.set_xticklabels([0, 0.04, 0.08, 0.12, 0.16, 0.2])
    axes.set_ylabel('value($m/s^2$)', fontsize=10)
    axes.set_xlabel('Time($s$)', fontsize=10)
    axes.set_title('original signal of ' + save_name + f" channel {csv_channel}")
    # plt.savefig("./pig_save/"+"fault1/" + save_name + "channel3")
    plt.savefig("./pig_save/"+"fault1/" + save_name + f" channel {csv_channel}")
    # plt.show()

def plot_freq(file_path, save_name, csv_channel = 4):
    data_ori = get_data(file_path)
    data_len = data_ori.shape[0]

    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    fig.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=0.8)
    axes.plot(np.arange(data_len), np.abs(fft(data_ori.iloc[:, csv_channel].values))/data_len*2, linewidth=0.15)


    # axes.set_ylabel('value($m/s^2$)', fontsize=10)
    axes.set_xlabel('freq($Hz$)', fontsize=10)
    axes.set_title('frequency of input signal')
    plt.savefig("./pig_save/"+"freq_of_" + save_name)
    plt.show()

def plt_spectrogram(file_path, save_name, csv_channel = 4):
    data_ori = pd.read_csv(file_path, skiprows=1, header=None)
    data_len = data_ori.shape[0]
    data_ori.index = np.arange(data_len)
    data_ori = data_ori.replace([np.inf, -np.inf], np.nan).fillna(data_ori.mean())  # 缺失值处理

    freqs, times, Sxx = signal.spectrogram(data_ori.iloc[:,csv_channel], fs=50_000, window='hanning',
                                          nperseg=64, noverlap=0.5*64,
                                          detrend=False, scaling='spectrum')
    plt.figure()
    plt.pcolormesh(times, freqs, 20 * np.log10(Sxx/1e-06), shading='auto', cmap='inferno')

    # plt.clim(70,100)
    # plt.ylim(0,1000)
    plt.colorbar()
    plt.ylabel('Frequency($Hz$)')
    plt.xlabel('Time($s$)');
    plt.savefig("./pig_save/"+"spectrogram_of_" + save_name)
    # plt.show()



if __name__ == "__main__":

    file_dir = 'breaker/正常/带电流(uesful)/'
    # file_name = 'backup_229.xlsx'
    file_name = 'backup_230.xlsx'
    for ind in range(4, 13):
        plot_original(file_dir + file_name, file_name.split('.',1)[0],ind)
    file_name = 'backup_231.xlsx'
    for ind in range(4, 13):
        plot_original(file_dir + file_name, file_name.split('.', 1)[0], ind)
    # plot_original(file_dir + file_name, file_name.split('.', 1)[0], ind)

    # plt_spectrogram(file_dir + file_name, "channel_3",3)
    # plot_freq(file_dir + file_name, "channel_5", 5)

