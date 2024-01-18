"""
测试模型的正确率
"""

from global_value import *
import pandas as pd
import json

fine_tuning = 0.55 #由于降水的样例偏少，因此对模型做一个修正使之更倾向于给出降水的预测

if __name__ == "__main__":
    df = pd.read_excel(test_data_path)
    df = day_rainy(df)
    df_data = df.drop(columns=columns_to_drop)
    columns = df_data.columns.tolist()
    columns.remove("RRR")
    """
    数据的列表
    """
    # for index, row in df_data.iterrows():
    #     """
    #     对降水列进行处理
    #     """
    #     if(df_data.loc[index,"RRR"]!=df_data.loc[index,"RRR"]):
    #         df_data.loc[index,"RRR"] = -1
    #         #break                  #如果break，则将后面记录不全的数据舍弃掉
    #     elif(df_data.loc[index,"RRR"] == "无降水"):
    #         df_data.loc[index,"RRR"] = 0
    #     else:
    #         df_data.loc[index,"RRR"] = 1

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

    with open(Model_para_path, "r") as f:
        """
        读取模型参数
        """
        model_para = json.load(f)
        #print(model_para)

    for index, row in df_data.iterrows():
        for co in columns:
            """
            数据规范化，同时将空数据置为center的值，不影响预测
            """
            if(df_data.loc[index,co]!=df_data.loc[index,co]):       #数据为nan
                df_data.loc[index,co] = model_para[co]['center']
            else:
                df_data.loc[index,co] = (df_data.loc[index,co] - model_para[co]['min']) / (model_para[co]['max'] - model_para[co]['min'])

    true_center = {}
    false_center = {}
    for co in columns:
        true_center[co] = model_para[co]['T_center']
        false_center[co] = model_para[co]['F_center']
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    Unknown = 0
    for index, rows in df_data.iterrows():          #测试准确率
        dis_true = cal_distance(index,df_data,true_center,columns)           #到正样例中心的距离
        dis_false = cal_distance(index,df_data,false_center,columns) + fine_tuning        #到负样例中心的距离
        if df_data.loc[index,"RRR"] == 1:
            if dis_true < dis_false:
                TP += 1
            else:
                FN += 1
        elif df_data.loc[index,"RRR"] == 0:
            if dis_true < dis_false:
                FP += 1
            else:
                TN += 1
        else:
            Unknown += 1

    print("TP:"+str(TP))
    print("TN:"+str(TN))
    print("FP:"+str(FP))
    print("FN:"+str(FN))
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    print("Precision:"+str(Precision))
    print("Recall:"+str(Recall))
    F1_score = 2*Precision*Recall/(Precision+Recall)
    print("F1_score = " + str(F1_score))