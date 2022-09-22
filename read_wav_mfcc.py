"""
    批量读取wav文件
"""
import os
from numpy import size
import re
from PyEMD import EMD
import numpy as np 
from scipy.fftpack import dct
from python_speech_features import *
import numpy as np
import scipy.io.wavfile
from matplotlib import pyplot as plt
import librosa
import csv
import time as tt

starttime1 = tt.time()
project_path = os.path.abspath('.')
data_path = os.path.abspath(".\ICBHI_final_database")
filenames = os.listdir(data_path)
print(project_path)
print(data_path)
# 文件分类存储
txt_file = []
wav_file = []
for file in filenames:
    nPos =file.find('.')
    file_class = file[nPos:nPos+4]
    if file_class == '.txt':
        txt_file.append(file)
    elif file_class == '.wav':
        wav_file.append(file)
txt_file = txt_file[0:-2]
print("共需要读取{}个txt文件".format(size(txt_file)))
print("共需要读取{}个wav文件".format(size(wav_file)))

# txt文件读取与切分
start_time = []
end_time = []
iscrackles = []
iswheezes = []
for index, file in enumerate(txt_file):
    start = []
    end = []
    crackles = []
    wheezes = []
    with open('.\ICBHI_final_database\{}'.format(txt_file[index]),'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            start.append(line[0])
            end.append(line[1])
            crackles.append(line[2])
            wheezes.append(line[3])
    start_time.append(start)
    end_time.append(end)
    iscrackles.append(crackles)
    iswheezes.append(wheezes)


"""
    MFCC特征提取
"""
MFCC = []
N_start = 0
N = 919
frame_len = 25 # each frame length (ms)
frame_shift = 10 # frame shift length (ms)
num_ceps = 13 # MFCC feature dims, usually between 2-13.

for index, wav in enumerate(wav_file[N_start:N]):
    wav_path = '.\ICBHI_final_database\{}'.format(wav_file[index]) 
    sample_rate, signal = scipy.io.wavfile.read(wav_path)
    times = librosa.get_duration(filename=wav_path,sr =sample_rate)
    
    for s in range(len(start_time[index])):
        start = float(start_time[index][s])
        end = float(end_time[index][s])
        # emd = EMD()
        wavdata= signal[int(start*sample_rate):int(end*sample_rate)]
        # wavdata = []
        # for i in range(wavdata_emd.shape[1]):
        #     wavdata.append(sum(wavdata_emd[:,i])/wavdata_emd.shape[0])
        wavdata = np.array(wavdata)
        # 预加重
        fs = sample_rate
        frame_len_samples = frame_len*fs//1000 # each frame length (samples)
        frame_shift_samples = frame_shift*fs//1000 # frame shifte length (samples)
        total_frames = int(np.ceil((len(wavdata) - frame_len_samples)/float(frame_shift_samples)) + 1) # total frames will get
        padding_length = int((total_frames-1)*frame_shift_samples + frame_len_samples - len(wavdata)) # how many samples last frame need to pad     
        pad_data = np.pad(wavdata,(0,padding_length),mode='constant') # pad last frame with zeros
        frame_data = np.zeros((total_frames,frame_len_samples)) # where we save the frame results
        pre_emphasis_coeff = 0.97 # Pre-emphasis coefficient
        pad_data = np.append(pad_data[0],pad_data[1:]-pre_emphasis_coeff*pad_data[:-1]) # Pre-emphasis

        # 分帧与加窗
        window_func = np.hamming(frame_len_samples) # hamming window
        for i in range(total_frames):
            single_frame = pad_data[i*frame_shift_samples:i*frame_shift_samples+frame_len_samples] # original frame data
            single_frame = single_frame * window_func # add window function
            frame_data[i,:] = single_frame
        
        # DFT
        K = 512 # length of DFT
        freq_domain_data = np.fft.rfft(frame_data,K) # DFT

        # 能量谱
        power_spec = np.absolute(freq_domain_data) ** 2 * (1/K) # power spectrum

        # Mel滤波
        low_frequency = 20 # We don't use start from 0 Hz because human ear is not able to perceive low frequency signal.
        high_frequency = fs//2 # if the speech is sampled at f Hz then our upper frequency is limited to 2/f Hz.
        low_frequency_mel = 2595 * np.log10(1 + low_frequency / 700)
        high_frequency_mel = 2595 * np.log10(1 + high_frequency / 700)
        n_filt = 40 # number of mel-filters (usually between 22-40)
        mel_points = np.linspace(low_frequency_mel, high_frequency_mel, n_filt + 2) # Make the Mel scale spacing equal.
        hz_points = (700 * (10**(mel_points / 2595) - 1)) # convert back to Hz scale.
        bins = np.floor((K + 1) * hz_points / fs) # round those frequencies to the nearest FFT bin.

        fbank = np.zeros((n_filt, int(np.floor(K / 2 + 1))))
        for m in range(1, n_filt + 1):
            f_m_minus = int(bins[m - 1])   # left point
            f_m = int(bins[m])             # peak point
            f_m_plus = int(bins[m + 1])    # right point

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
        filter_bank = np.matmul(power_spec,fbank.T) # This is known as fbank feature.
        filter_bank = np.where(filter_bank == 0,np.finfo(float).eps,filter_bank) # Repalce 0 to a small constant or it will cause problem to log.

        # 取log
        log_fbank = np.log(filter_bank)

        # DCT
        # feature from other dims are dropped beacuse they represent rapid changes in filter bank coefficients and they are not helpful for speech models.
        mfcc = dct(log_fbank, type=2, axis=1, norm="ortho")[:, 1 : (num_ceps + 1)]
        mfcc_1 = []
        mfcc = np.array(mfcc)
        for i in range(num_ceps):
            mean_mfcc = sum(mfcc[:][i])/len(mfcc[:][i])
            mfcc_1.append(mean_mfcc)
        MFCC.append(mfcc_1)
MFCC = np.array(MFCC)
print("MFCC特征提取完毕,特征矩阵为:",MFCC.shape)


"""
短时过零率特征提取、短时能量提取
"""
def sgn(data):
    if data >= 0 :
        return 1
    else :
        return 0
#计算过零率
def calZeroCrossingRate(wave_data) :
    zeroCrossingRate = []
    sum = 0
    for i in range(len(wave_data)) :
        if i % 256 == 0:
            continue
        sum = sum + np.abs(sgn(wave_data[i]) - sgn(wave_data[i - 1]))
        if (i + 1) % 256 == 0 :
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(wave_data) - 1 :
            zeroCrossingRate.append(float(sum) / 255)
    return zeroCrossingRate
# 计算每一帧的能量 256个采样点为一帧
def calEnergy(wave_data) :
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0 :
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1 :
            energy.append(sum)
    return energy

zeroCrossingRate_array = []
energy_array = []
wavdatalist_array = []
wavdatalist_w_array = []
# for index, wav in enumerate(wav_file[N_start:N]):
#     wav_path = '.\ICBHI_final_database\{}'.format(wav_file[index])
#     sample_rate, signal = scipy.io.wavfile.read(wav_path)
#     times = librosa.get_duration(filename=wav_path,sr =sample_rate)
    
#     for s in range(len(start_time[index])):
#         start = float(start_time[index][s])
#         end = float(end_time[index][s])
#         wavdata = signal[int(start*sample_rate):int(end*sample_rate)]

#         # 短时平均过零率
#         zeroCrossingRate = calZeroCrossingRate(wavdata)
#         zeroCrossingRate_mean = sum(zeroCrossingRate)/len(zeroCrossingRate)
#         zeroCrossingRate_array.append(zeroCrossingRate_mean)

#         # 短时能量
#         energy = calEnergy(wavdata)
#         energy_mean = sum(energy)/len(energy)
#         energy_array.append(energy_mean)

#         # 短时平均幅度
#         wavdatalist = []
#         ww = wavdata.tolist()
#         for j,data in enumerate(ww):
#             wavdatalist.append(abs(data))
#         wavdatalist_array.append(sum(wavdatalist))

#         # # 加权短时平均幅度
#         # wavdatalist_w = []
#         # wavsort = np.argsort(wavdata)+1
#         # for j,data in enumerate(ww):
#         #     wavdatalist_w.append(abs(data)*wavsort[j])
#         # wavdatalist_w_array.append(sum(wavdatalist_w))


zeroCrossingRate_array = np.array(zeroCrossingRate_array)
energy_array = np.array(energy_array)
wavdatalist_array = np.array(wavdatalist_array)
wavdatalist_w_array = np.array(wavdatalist_w_array)
print("短时平均过零率提取完毕,特征矩阵形状为:",zeroCrossingRate_array.shape)
print("短时能量提取完毕,特征矩阵形状为:",energy_array.shape)
print("短时平均幅度提取完毕,特征矩阵形状为:",wavdatalist_array.shape)


"""
制作数据集
"""
lenarray = []
for i in range(len(MFCC)):
    lenarray.append(len(MFCC[i]))
min_len = min(lenarray)

iscrackles_sum = []
iswheezes_sum = []
for i in range(len(iscrackles[0:N])):
    for j in range(len(iscrackles[i])):
        iscrackles_sum.append(iscrackles[i][j])
        iswheezes_sum.append(iswheezes[i][j])

for i in range(len(iswheezes_sum)):
    if iswheezes_sum[i] == '1':
        iswheezes_sum[i] = '2'

Write_data = MFCC.tolist()
for i in range(len(MFCC)):
    s = int(iscrackles_sum[i])+int(iswheezes_sum[i])
    # Write_data[i].append(zeroCrossingRate_array[i])
    # Write_data[i].append(energy_array[i])
    # Write_data[i].append(wavdatalist_array[i])
    # Write_data[i].append(wavdatalist_w_array[i])
    Write_data[i].append(s)


with open('data{}_{}.csv'.format(N_start, N - 1),'a+',newline='') as f:
        f = csv.writer(f)
        mfccname = ['mfcc_{}'.format(i+1) for i in range(num_ceps)]
        # mfccname.append('zero')
        # mfccname.append('energy')
        # mfccname.append('range')
        # mfccname.append('range_w')
        mfccname.append('target')
        f.writerow(mfccname)

for i in range(len(MFCC)):
    with open('data{}_{}.csv'.format(N_start, N - 1),'a+',newline='') as f:
        f = csv.writer(f)
        f.writerow(Write_data[i])
print("四分类数据集制作完毕！")

"""
制作'三分类'数据集
"""
import pandas as pd
data_read = pd.read_csv('data{}_{}.csv'.format(N_start, N - 1))
data_read_np = np.array(data_read)
with open('data3_{}_{}.csv'.format(N_start, N - 1),'a+',newline='') as f:
        f = csv.writer(f)
        mfccname = ['mfcc_{}'.format(i+1) for i in range(num_ceps)]
        mfccname.append('zero')
        mfccname.append('energy')
        mfccname.append('range')
        # mfccname.append('range_w')
        mfccname.append('target')
        f.writerow(mfccname)
for i in range(data_read_np.shape[0]):
    if data_read_np[i,data_read_np.shape[1] - 1] < 3:
        with open('data3_{}_{}.csv'.format(N_start, N - 1), 'a+', newline='') as f:
            f = csv.writer(f)
            data_read_list_i = data_read_np[i,:].tolist()
            f.writerow(data_read_list_i)
print("三分类数据集制作完毕！")

"""
制作'二分类'数据集
"""
with open('data2_{}_{}.csv'.format(N_start, N - 1),'a+',newline='') as f:
        f = csv.writer(f)
        mfccname = ['mfcc_{}'.format(i+1) for i in range(num_ceps)]
        mfccname.append('zero')
        mfccname.append('energy')
        mfccname.append('range')
        # mfccname.append('range_w')
        mfccname.append('target')
        f.writerow(mfccname)
for i in range(data_read_np.shape[0]):
    if data_read_np[i,data_read_np.shape[1] - 1] >= 1:
        data_read_np[i,data_read_np.shape[1] - 1] = 1
        with open('data2_{}_{}.csv'.format(N_start, N - 1), 'a+', newline='') as f:
            f = csv.writer(f)
            data_read_list_i = data_read_np[i,:].tolist()
            f.writerow(data_read_list_i)
    else:
        with open('data2_{}_{}.csv'.format(N_start, N - 1), 'a+', newline='') as f:
            f = csv.writer(f)
            data_read_list_i = data_read_np[i,:].tolist()
            f.writerow(data_read_list_i)
print("二分类数据集制作完毕！")

"""
计算程序运行时长
"""
endtime1 = tt.time()
programtime =  endtime1 - starttime1
minute = 0
second = 0
print(programtime)

if programtime <= 60:
    print("程序运行时长为：{}秒".format(programtime))
elif programtime >60  and programtime <= 3600:
    minute = int(programtime/60)
    second = programtime % 60
    print("程序运行时长为：{}分{}秒".format(minute,second))
elif programtime > 3600 and programtime <= 86400:
    hour  = int(programtime/3600)
    if (programtime % 3600) % 60 > 0 :
        minute = int((programtime % 3600)/60)
        second = (programtime % 3600) % 60
    else:
        second = (programtime % 3600)
    print("程序运行时长为：{}小时{}分{}秒".format(hour,minute,second))  
print("程序运行结束！")                                                                                                            