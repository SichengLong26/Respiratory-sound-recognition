import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
import time
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel("data.xlsx")
x = data.iloc[:,0:-1]
y = data["target"]

# 转化成字典
x = x.to_dict(orient="records")

# 3、数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

# 4、字典特征抽取
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
start5 = time.time()
estimator = AdaBoostClassifier(n_estimators=100)
estimator.fit(x_train, y_train)

# 模型评估
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("直接对比真实值和预测值：\n",y_test == y_predict)
score5 = estimator.score(x_test, y_test)
print("准确率为:",score5)
end5 = time.time()
print("AdaBoost模型运行时间为{}秒".format(end5-start5))

# 制作混淆矩阵
tgt0_pred = 0
tgt1_pred  = 0
tgt2_pred  = 0
tgt3_pred  = 0
for i in range(y_predict.shape[0]):
    if y_predict[i] == 0:
        tgt0_pred  += 1
    elif y_predict[i] == 1:
        tgt1_pred  += 1
    elif y_predict[i] == 2:
        tgt2_pred  += 1
    elif y_predict[i] == 3:
        tgt3_pred  += 1

tgt0_real = 0
tgt1_real  = 0
tgt2_real  = 0
tgt3_real  = 0
y_test = np.array(y_test)
for i in range(y_test.shape[0]):
    if y_test[i] == 0:
        tgt0_real  += 1
    elif y_test[i] == 1:
        tgt1_real  += 1
    elif y_test[i] == 2:
        tgt2_real  += 1
    elif y_test[i] == 3:
        tgt3_real  += 1

print(tgt0_pred,tgt1_pred,tgt2_pred)
print(tgt0_real,tgt1_real,tgt2_real,tgt3_real)
print("精确率和召回率为：", classification_report(y_test, estimator.predict(x_test), labels=[0,1,2], target_names=['正常','爆破音','哮鸣音']))


# 0.5~1之间，越接近于1约好

print("AUC指标：", roc_auc_score(np.where(y_test > 2.5, 1, 0), estimator.predict(x_test)))


