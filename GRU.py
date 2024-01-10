import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import cycle

import joblib
import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 18})


# 可视化数据
def visualize_data(data, row, col):
    cycol = cycle('bgrcmk')
    cols = list(data.columns)
    fig, axes = plt.subplots(row, col, figsize=(16, 4))
    fig.tight_layout()
    if row == 1 and col == 1:  # 处理只有1行1列的情况
        axes = [axes]  # 转换为列表，方便统一处理
    for i, ax in enumerate(axes.flat):
        if i < len(cols):
            ax.plot(data.iloc[:, i], c=next(cycol))
            ax.set_title(cols[i])
        else:
            ax.axis('off')  # 如果数据列数小于子图数量，关闭多余的子图
    plt.subplots_adjust(hspace=0.5)
    plt.show()

data = pd.read_csv('changzhou.csv')

# 特征删除和缺失值填充
data.drop(['时间','场站名称'], axis=1, inplace=True)
data = data.fillna(method='ffill')
# 调整列位置
data = data[[' 辐照强度(Wh/㎡)', ' 环境温度(℃)', ' 全场功率(kW)']]

# visualize_data(data, 1, 3)
dataf = data.values
#构造数据集
def create_dataset(datasetx,datasety,timesteps=36,predict_size=6):
    datax=[]#构造x
    datay=[]#构造y
    for each in range(len(datasetx)-timesteps - predict_steps):
        x = datasetx[each:each+timesteps,0:6]
        y = datasety[each+timesteps:each+timesteps+predict_steps,0]
        datax.append(x)
        datay.append(y)
    return datax, datay#np.array(datax),np.array(datay)

timesteps = 96*5  #构造x，为96*5个数据,表示每次用前5/4天的数据作为一段
predict_steps = 96  #构造y，为96个数据，表示用后1/4的数据作为一段
length = 96  #预测多步，预测96个数据据

# 特征和标签分开划分
datafx = dataf[:, :-1]
datafy = dataf[:, -1].reshape(dataf.shape[0], 1)

# 分开进行归一化处理
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
datafx = scaler1.fit_transform(datafx)
datafy = scaler2.fit_transform(datafy)

trainx, trainy = create_dataset(datafx[:-predict_steps*6,:],datafy[:-predict_steps*6],timesteps, predict_steps)
trainx = np.array(trainx)
trainy = np.array(trainy)

# Define the GPU device
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# GRU training
start_time = datetime.datetime.now()
model = Sequential()
model.add(GRU(128, input_shape=(timesteps, trainx.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(128, return_sequences=True))
model.add(GRU(64, return_sequences=False))
model.add(Dense(predict_steps))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(trainx, trainy, epochs=10, batch_size=128)
end_time = datetime.datetime.now()
running_time = end_time - start_time

# 保存模型
model.save('gru_model.h5')

y_true = dataf[-96:,-1]
predictx = datafx[-96*6:-96]

# 加载模型
from tensorflow.keras.models import load_model
model = load_model('gru_model.h5')


def predict_and_plot(x, y_true, model, scaler, timesteps):
    # 变换输入x格式，适应LSTM模型
    predict_x = np.reshape(x, (1, timesteps, 2))
    # 预测
    predict_y = model.predict(predict_x)
    predict_y = scaler.inverse_transform(predict_y)
    y_predict = []
    y_predict.extend(predict_y[0])
    # 计算误差
    train_score = np.sqrt(mean_squared_error(y_true, y_predict))
    print("train score RMSE: %.2f" % train_score)

    # 预测结果可视化
    cycol = cycle('bgrcmk')
    plt.figure(dpi=100, figsize=(14, 5))
    plt.plot(y_true, c=next(cycol), markevery=5)
    plt.plot(y_predict, c=next(cycol), markevery=5)
    plt.legend(['y_true', 'y_predict'])
    plt.xlabel('时间')
    plt.ylabel('功率(kW)')
    plt.show()

    return y_predict

y_predict = predict_and_plot(predictx, y_true, model, scaler2, timesteps)