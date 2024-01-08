"""
根据输入的表格训练2-means模型
"""

from global_value import *
import pandas as pd
import json

model = {}
"""
用dict来存储训练后的模型参数，分别有该列最大值max，最小值min，
在2-means模型中正项的中心T_center，负项的中心F_center，两者的中点center(center为规范化后的0~1)
在预测时，如果遇到空数据，则将该项置为center，不影响预测，
如果遇到非空数据，就将该数据减去min然后除以max-min，映射到0~1区域进行预测，以确保各列数据影响一致
"""

if __name__ == "__main__":
    df = pd.read_excel(train_data_path)
    df_data = df.drop(columns=columns_to_drop)
    columns = df_data.columns.tolist()
    columns.remove("RRR")
    """
    数据的列表
    """
    for index, row in df_data.iterrows():
        """
        对降水列进行处理
        """
        if(df_data.loc[index,"RRR"]!=df_data.loc[index,"RRR"]):         #数据为nan
            df_data.loc[index,"RRR"] = -1
            #break                  #如果break，则将后面记录不全的数据舍弃掉
        elif(df_data.loc[index,"RRR"] == "无降水"):
            df_data.loc[index,"RRR"] = 0
        else:
            df_data.loc[index,"RRR"] = 1

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

    df_data = df_data.apply(pd.to_numeric, errors='coerce')
    mean = df_data.mean()
    df_max = df_data.max()
    df_min = df_data.min()

    for co in columns:
        model[co] = {}
        model[co]['max'] = df_max[co]
        model[co]['min'] = df_min[co]
        #print(train_data.loc[0,co])
        for index, row in df_data.iterrows():       # 将空值赋值为平均值
            if pd.isna(df_data.loc[index,co]):
                df_data.loc[index,co] = mean[co]
        for index, row in df_data.iterrows():       # 将值按比例规范化到[0,1]区间
            df_data.loc[index,co] = (df_data.loc[index,co] - df_min[co]) / (df_max[co] - df_min[co])
    print(df_data)

    true_center = {key: 0 for key in columns}           #正样例的中心
    false_center = {key: 0 for key in columns}          #负样例的中心
    true_next = {key: 0 for key in columns}           #正样例的中心迭代
    false_next = {key: 0 for key in columns}          #负样例的中心迭代
    true_cnt = 0
    false_cnt = 0
    for index, row in df_data.iterrows():
        """
        从有标注的样本中初始化正负样本的中心
        """
        if df_data.loc[index,"RRR"] == 0:
            false_cnt += 1
            for co in columns:
                false_center[co] += df_data.loc[index,co]
        if df_data.loc[index,"RRR"] == 1:
            true_cnt += 1
            for co in columns:
                true_center[co] += df_data.loc[index,co]
    true_center = {key: value / true_cnt for key, value in true_center.items()}
    false_center = {key: value / false_cnt for key, value in false_center.items()}

    for _ in range(0,epoches):        #进行训练
        print("当前训练轮数:" + str(_))
        true_cnt = 0
        false_cnt = 0
        for co in columns:
            true_next[co] = 0
            false_next[co] = 0
        for index, rows in df_data.iterrows():
            dis_true = cal_distance(index,df_data,true_center,columns)           #到正样例中心的距离
            dis_false = cal_distance(index,df_data,false_center,columns)         #到负样例中心的距离
            if dis_true < dis_false:      #视为正样例
                for co in columns:
                    true_next[co] += df_data.loc[index,co]
                true_cnt += 1
            else:
                for co in columns:
                    false_next[co] += df_data.loc[index,co]
                false_cnt += 1
        for co in columns:
            true_center[co] = true_next[co] / true_cnt
            false_center[co] = false_next[co] / false_cnt
        print(true_cnt,false_cnt)
        print(true_center)
        print(false_center)

    print("训练完毕！")

    for co in columns:
        model[co]['T_center'] = true_center[co]
        model[co]['F_center'] = false_center[co]
        model[co]['center'] = (true_center[co]+false_center[co])/2
    
    with open(Model_para_path, "w") as file:
        json_str = json.dumps(model)
        file.write(json_str)