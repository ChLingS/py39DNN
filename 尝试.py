import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import keras


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
# print(merged_l)
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
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('rsei')
test_labels = test_features.pop('rsei')

print(train_dataset[:10])