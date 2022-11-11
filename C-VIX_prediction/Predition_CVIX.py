import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

df=pd.read_excel("data2.xlsx",parse_dates=["date"],index_col=[0])

#数据切片
test_split=round(len(df)*0.20)

df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]

#标准化
scaler = MinMaxScaler(feature_range=(0,1))
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX) , np.array(dataY)

trainX,trainY=createXY(df_for_training_scaled,30)
testX,testY=createXY(df_for_testing_scaled,30)

#构建模型并基础优化
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

history=[]
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,20)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    history = LossHistory
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))

my_model = grid_model.fit(trainX,trainY,callbacks = [history])


parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)

grid_search = grid_search.fit(trainX,trainY)

#print(grid_search.best_params_)
#存储模型
my_model=grid_search.best_estimator_.model


#开始预测
prediction=my_model.predict(testX)

#处理预测的结果
prediction_copies_array = np.repeat(prediction,20, axis=-1)
pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),20)))[:,0]#inverse_tranform

#处理TextY
original_copies_array = np.repeat(testY,20, axis=-1)
original=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),20)))[:,0]

import matplotlib.pyplot as plt

plt.plot(testY, color = 'red', label = 'Real  Stock Price')
plt.plot(prediction, color = 'blue', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show();



#上面只是初步建立，训练和评估模型

#数据切片，只需要前30天的数据
df_30_days_past=df.iloc[-30:,:]

timestep = 30

df_30_days_past_array=scaler.transform(df_30_days_past)

print(df_30_days_past_array)

data_x = []
data_x.append(df_30_days_past_array[0:30,0:df_30_days_past_array.shape[1]])

data_x=np.array(data_x)
prediction=my_model.predict(data_x)

prediction_copies_array = np.repeat(prediction,20, axis=-1)

y_pred_future_30_days = scaler.inverse_transform(prediction_copies_array)[0,0]


print("预测结果是 ",y_pred_future_30_days)
