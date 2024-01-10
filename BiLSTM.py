import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sampen import sampen2  # sampen库用于计算样本熵
from vmdpy import VMD  # VMD分解库

import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# 忽略警告信息
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams.update({'font.size':18})  #统一字体字号

# 导入数据
data_raw = pd.read_excel('2019.xlsx')

from itertools import cycle


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
    plt.subplots_adjust(hspace=0.6)
    plt.show()

# visualize_data(data_raw.iloc[:, 1:], 2, 4)
data = data_raw.iloc[:, 1:].values

timesteps = 96*5 #构造x，为96*5个数据,表示每次用前96*5个数据作为一段
predict_steps = 96 #构造y，为96个数据，表示用后96个数据作为一段
length = 96 #预测多步，预测96个数据
feature_num = 7 #特征的数量

# 构造数据集，用于真正预测未来数据
# 整体的思路也就是，前面通过前timesteps个数据训练后面的predict_steps个未来数据
# 预测时取出前timesteps个数据预测未来的predict_steps个未来数据。
def create_dataset(datasetx,datasety,timesteps=36,predict_size=6):
    datax=[]#构造x
    datay=[]#构造y
    for each in range(len(datasetx)-timesteps - predict_steps):
        x = datasetx[each:each+timesteps]
        y = datasety[each+timesteps:each+timesteps+predict_steps]
        datax.append(x)
        datay.append(y)
    return datax, datay


# 数据归一化操作
def data_scaler(datax, datay):
    # 数据归一化操作
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    datax = scaler1.fit_transform(datax)
    datay = scaler2.fit_transform(datay)
    # 用前面的数据进行训练，留最后的数据进行预测
    trainx, trainy = create_dataset(datax[:-timesteps - predict_steps, :], datay[:-timesteps - predict_steps, 0],
                                    timesteps, predict_steps)
    trainx = np.array(trainx)
    trainy = np.array(trainy)

    return trainx, trainy, scaler1, scaler2

datax = data[:,:-1]
datay = data[:, -1].reshape(data_raw.shape[0], 1)
trainx, trainy, scaler1, scaler2 = data_scaler(datax, datay)


# # 创建BiLSTM模型
def BiLSTM_model_train(trainx, trainy):
    # 调用GPU加速
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # BiLSTM网络构建
    start_time = datetime.datetime.now()
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(timesteps, feature_num)))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=(timesteps, feature_num)))
    model.add(Bidirectional(LSTM(units=150)))
    model.add(Dropout(0.1))
    model.add(Dense(predict_steps))
    model.compile(loss='mse', optimizer='adam')
    # 模型训练
    model.fit(trainx, trainy, epochs=10, batch_size=64)
    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    # 保存模型
    model.save('BiLSTM_model.h5')

    # 返回构建好的模型
    return model

model = BiLSTM_model_train(trainx, trainy)
model.save('lstm_model.h5')

# 加载模型
from tensorflow.keras.models import load_model
model = load_model('BiLSTM_model.h5')

y_true = datay[-timesteps-predict_steps:-timesteps]
x_pred = datax[-timesteps:]


# 预测并计算误差和可视化
def predict_and_plot(x, y_true, model, scaler, timesteps):
    # 变换输入x格式，适应LSTM模型
    predict_x = np.reshape(x, (1, timesteps, feature_num))
    # 预测
    predict_y = model.predict(predict_x)
    predict_y = scaler.inverse_transform(predict_y)
    y_predict = []
    y_predict.extend(predict_y[0])

    # 计算误差
    r2 = r2_score(y_true, y_predict)
    rmse = mean_squared_error(y_true, y_predict, squared=False)
    mae = mean_absolute_error(y_true, y_predict)
    mape = mean_absolute_percentage_error(y_true, y_predict)
    print("r2: %.2f\nrmse: %.2f\nmae: %.2f\nmape: %.2f" % (r2, rmse, mae, mape))

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

y_predict_nowork = predict_and_plot(x_pred, y_true, model, scaler2, timesteps)

