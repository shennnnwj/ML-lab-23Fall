import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re

#os.chdir("data")

print("?????")
df = pd.read_excel('./data/training_dataset.xls')

columns_to_drop = ['Time Stamp', 'DD','ff10','N', 'WW','W2','Cl','Nh','H','Cm','Ch','E','Tg',"E'",'sss']

df = df.drop(columns=columns_to_drop)
for i in range(0,8000):
    if(df.loc[i,"RRR"] == "无降水"):
        df.loc[i,"RRR"] = 0
    else:
        df.loc[i,"RRR"] = 1

test = df.iloc[7000:8000]
df = df.iloc[:7000]

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