import cv2
import os 
import numpy as np
import csv

project_path = os.path.abspath('.')
img_path = os.path.abspath(".\img")
filenames = os.listdir(img_path)
filenames = np.array(filenames)
print(filenames)
def img_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

data_len = filenames.shape[0]
rm = 30
rn = 30
target1 = []
target2 = []
img_data = np.zeros(shape=(data_len,30,90))
for i in range(data_len):
    path = './img/{}'.format(filenames[i])
    img_original = cv2.imread(path) # (656, 875, 3)
    img_1d = cv2.resize(img_original,(rm,rn)).reshape(1,30,-1)
    target1.append(filenames[i][3])
    target2.append(filenames[i][4])
    img_data[i,0:30,0:90] = img_1d
print(img_data.shape)
# target1 = np.array(target1)
# target2 = np.array(target2)
# target = []
# for i in range(len(target1)):
#     target.append(int(target1[i]) + int(target2[i]))
# target = np.array(target)
# print(target)
# x_train = img_data[0:int(data_len*0.8),:]
# x_test = img_data[int(data_len*0.8):data_len,:]
# y_train = target[0:int(data_len*0.8)]
# y_test = target[int(data_len*0.8):data_len]

path = './img/{}'.format(filenames[1])
img_original = cv2.imread(path) # (656, 875, 3)
img_1d = cv2.resize(img_original,(rm,rn)).reshape(1,30,-1)
print(img_1d.shape)