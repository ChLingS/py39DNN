import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import shap


def merge_tables(tables):
    # 读取所有表格
    dfs = [pd.read_csv(table, header=0) for table in tables]
    # 将所有表格按照 system:index 列合并
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df.iloc[:, :2], on='system:index', how='outer')
    # 将缺失值替换为 0
    merged_df = merged_df.dropna()
    return merged_df


tables = ['resi_1.csv', 'np0.csv', 'pr.csv', 'slope.csv', 'vap.csv', 'vs.csv']
merged_l = merge_tables(tables)
print(merged_l.describe())
# 归一化
scaler = MinMaxScaler()
res = merged_l[['Npp', 'vs', 'elevation', 'pr', 'vap']]
test = scaler.fit_transform(res)
merged_l[['Npp', 'vs', 'elevation', 'pr', 'vap']] = test

merged_l = merged_l[['rsei', 'Npp', 'vs', 'elevation', 'pr', 'vap']]

# 拆分测试集与训练集
train_dataset = merged_l.sample(frac=0.8, random_state=0)
test_dataset = merged_l.drop(train_dataset.index)
print('训练数据集\n', train_dataset)
print('测试数据集数据量\n', test_dataset.count())
print('\n训练数据集数据量\n', train_dataset.count())

sns.pairplot(train_dataset[['rsei', 'Npp', 'vs', 'elevation', 'pr', 'vap']], diag_kind="kde")
print('\n总体统计信息', train_dataset.describe().transpose())

train_stats = train_dataset.describe()
train_stats.pop("rsei")
train_stats = train_stats.transpose()

# 从标签中分离特征
train_labels = train_dataset.pop('rsei')
test_labels = test_dataset.pop('rsei')

print('train_labels\n', train_labels)
print('len(train_dataset.keys())\n', len(train_dataset.keys()))
# DNN多输入回归
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae'])
    return model


dnn_model = build_model()
print(dnn_model.summary())


# 测试
# example_batch = train_dataset[:10]
# print('example_batch\n', example_batch)
# example_result = dnn_model.predict(example_batch)
# print(example_result)

# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 500

history = dnn_model.fit(
    train_dataset, train_labels,
    epochs=EPOCHS, validation_data=(test_dataset, test_labels), verbose=1,)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [rsei]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 0.4])
    plt.legend()

    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel('Mean Square Error [$rsei^2$]')
    # plt.plot(hist['epoch'],
    #          label='Train Error')
    # plt.plot(hist['epoch'],
    #          label='Val Error')
    # plt.ylim([0, 0.15])
    # plt.legend()
    plt.show()


plot_history(history)

loss, mae = dnn_model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} rsei".format(mae))

print(dnn_model)

yes = input()
if yes == '1':
    tf.saved_model.save(dnn_model, "save/mymodel")

explainer = shap.KernelExplainer(dnn_model, train_dataset, session='tensorflow')
shap_values = explainer.shap_values(train_dataset)

# shap.summary_plot(shap_values, train_dataset[:10], feature_names=feature_names, plot_type='bar')
shap.summary_plot(shap_values, train_dataset,
                  plot_type="bar")

