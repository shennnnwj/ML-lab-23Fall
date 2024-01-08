"""
分析每一个中文项与降不降水的关系，然后输出一个中文到数字的映射表
"""

import pandas as pd
from global_value import *
import json

co_to_map = {}          #记录每一列的中文映射词典，将中文映射到降水概率，保存为词典的词典
Max = 40000              #只解析前Max行

if __name__ == "__main__":
    df_data = pd.read_excel(train_data_path)
    df_data = df_data.drop(columns=columns_to_drop)
    column = df_data.columns
    #print(column)
    for index, row in df_data.iterrows():
        """
        对降水列进行处理
        """
        if(df_data.loc[index,"RRR"]!=df_data.loc[index,"RRR"]):
            df_data.loc[index,"RRR"] = -1
            #break                  #如果break，则将后面记录不全的数据舍弃掉
        elif(df_data.loc[index,"RRR"] == "无降水"):
            df_data.loc[index,"RRR"] = 0
        else:
            df_data.loc[index,"RRR"] = 1
    for co in column:
        """
        检测每一列是否有中文，有的话用字典处理
        """
        Chinese_to_num_map = {}         #记录中文词项出现次数
        map_rainy = {}            #记录词项对应降水次数
        rain_percent = {}
        for index, row in df_data.iterrows():
            if(index>Max):
                break
            ##if(df_data.loc[index,"RRR"] == -1):         #无降水记录，检测没意义，跳过
            ##    continue
            temp = df_data.loc[index,co]
            if(isinstance(temp,str)):      #出现非数字项
                if not (temp in Chinese_to_num_map):
                    Chinese_to_num_map[temp] = 0
                    map_rainy[temp] = 0
                Chinese_to_num_map[temp] += 1       #记出现+1次
                if(df_data.loc[index,"RRR"] == 1):
                    map_rainy[temp] += 1
                elif(df_data.loc[index,"RRR"] == -1):       #无记录，各占一半
                    map_rainy[temp] += 0.5
        for key in Chinese_to_num_map:              #计算降水百分比
            rain_percent[key] = map_rainy[key] / Chinese_to_num_map[key]
        co_to_map[co] = rain_percent

    for key in co_to_map:
        print(co_to_map[key])
    with open(Chinese_num_map_path, "w") as file:
        json_str = json.dumps(co_to_map)
        file.write(json_str)
    
