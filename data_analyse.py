import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from global_value import *
import re
import json

#os.chdir("data")

print("?????")
df = pd.read_excel('./data/training_dataset.xls')
df = day_rainy(df)          #处理降水项
df_data = df.drop(columns=columns_to_drop)
columns = df_data.columns.tolist()
columns.remove("RRR")
with open(Chinese_num_map_path, "r") as f:
    Chinese_to_num_map = json.load(f)
    """
    中文到数字的映射表
    """
for index, row in df_data.iterrows():       #将中文转化为数字
    for key_co in Chinese_to_num_map:
        submap = Chinese_to_num_map[key_co]
        if(df_data.loc[index,key_co] in submap):
            df_data.loc[index,key_co] = submap[df_data.loc[index,key_co]]
        
# test = df_data.iloc[7000:8000]
df = df_data.iloc[:7000]

print(df.head())
print(df.corr())

# 指定横坐标和纵坐标的列
x_column = 'U'  # 用作横坐标的列
y_column = 'Td'  # 用作纵坐标的列
color_column = 'RRR'  # 用作颜色的列

# 根据条件筛选数据，分别取出满足条件和不满足条件的数据
df_condition_1 = df[df[color_column] == 1]
df_condition_0 = df[df[color_column] == 0]

# 画散点图
plt.scatter(df_condition_1[x_column], df_condition_1[y_column], color='red', label='c=1',s=5)
plt.scatter(df_condition_0[x_column], df_condition_0[y_column], color='green', label='c=0',s=5)

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Scatter Plot with Color Condition')
plt.xlabel('Column A')
plt.ylabel('Column B')

# 显示图形
plt.show()
#plt.rc('font',family='SimHei',size=4)
#plt.subplots(figsize= (10,10),dpi=500)
#sns.heatmap(df.corr(),annot=True)
#plt.show()