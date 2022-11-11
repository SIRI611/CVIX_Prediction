#导入扩展库
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import preprocessing
from sklearn import decomposition

#设置字体和显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.weight"] = "medium"
plt.rcParams["axes.labelweight"] = "bold"

df = pd.read_excel("C:\\Users\\siri\\OneDrive\\大创VIX指数\\data_af_std.xlsx",
                   index_col=0,
                   date_parser=["交易日期"])

# #绘制特征相关系数矩阵热力图
# corr = np.around(np.corrcoef(df.T),decimals=2)
# plt.subplots(figsiarray(df['上证50ETF波动率指数'])
target = target.reshape(-1,1)
std = StandardScaler()
std_target = std.fit_transform(target)
std_target_ser = pd.Series(std_target.flatten())
std_target_ser.index = df.index
df["上证50ETF波动率指数"]=std_target_ser
main_component["上证50ETF波动率指数"] = std_target_ser #可以后续改用主成分进行预测

#数据切片
test_split = round(len(df) * 0.03)

#主成分数据
mc_training = main_component[:-test_split]
mc_testing = main_component[-test_split:]
mc_training_arr = np.array(mc_training)
mc_testing_arr = np.array(mc_testing)

#测试集和训练集构建函数
def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)

n_past = 30
#主成分数据
trainX_mc,trainY_mc = createXY(mc_training_arr, n_past)
testX_mc,testY_mc = createXY(mc_testing_arr, n_past)


#模型构建函数(主成分数据)
def build_model_mc(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50, return_sequences=True, input_shape=(n_past, 4)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss='mse', optimizer=optimizer)
    return grid_model


#使用GrdiSearchCV自动调参(主成分数据)
grid_model = KerasRegressor(build_fn=build_model_mc,
                            verbose=1,
                            validation_data=(testX_mc, testY_mc))

parameters = {
    'batch_size': [3,5,7],
    'epochs': [7,9,11],
    'optimizer': ['adam', 'Adadelta']
}

grid_search = GridSearchCV(estimator=grid_model, param_grid=parameters, cv=2)

grid_search = grid_search.fit(trainX_mc, trainY_mc)

#输出初步调参最佳参数
print(grid_search.best_params_)
#存储模型
my_model_mc = grid_search.best_estimator_.model

#根据最优调参结果进行训练（主成分数据）
lstm_best_mc = KerasRegressor(build_fn=build_model_mc,
                            verbose=1,
                            validation_data=(testX_mc, testY_mc),
                          batch_size = 3,
                          epochs = 11,
                          optimizer = 'adam')
lstm_best_mc = lstm_best_mc.fit(trainX_mc,trainY_mc)
lstm_best_mc = lstm_best_mc.model

#模型在训练集预测（主成分数据）
pred_train_mc = lstm_best_mc.predict(trainX_mc)
#将训练集预测结果反标准化
pred_train_origin_mc = std.inverse_transform(pred_train_mc.reshape(-1,1))
#将训练集真实结果反标准化
trainY_origin_mc = std.inverse_transform(trainY_mc.reshape(-1,1))
#数据格式整理
train_result_mc = pd.DataFrame()
train_result_mc.index = main_component[n_past:-test_split].index #need to make sure that this is right
train_result_mc['训练集实际结果'] = trainY_origin_mc
train_result_mc['训练集预测结果'] = pred_train_origin_mc

print(pred_train_mc)