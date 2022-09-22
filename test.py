import imp
from PyEMD import EMD
import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np

a, data = scipy.io.wavfile.read('./ICBHI_final_database/101_1b1_Al_sc_Meditron.wav')
data = data[0:1000]

print("original data:",data)
emd = EMD()
data_emd = emd(data)
print("emd data:",data_emd)
print("origin data shape:",np.array(data).shape)
print("emd data shape:",np.array(data_emd).shape)
plt.figure()
plt.subplot(2,1,1)
plt.plot(data)
plt.title("original data")
plt.subplot(2,1,2)
plt.plot(data_emd)
plt.title("emd data")
plt.show()