import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import argparse
import warnings

## your model path should takes the form as
Model_Path = '/home/hyliu/ML_Project/your_model_path'

pd.options.mode.chained_assignment = None
#训练模型，采用2-Means方法，首先先计算0~6000的数据中心作为起始中心，
#接着不断迭代更新数据中心的位置（训练集为挖掉6000~8000的剩余数据）
#最后用6000~8000的数据进行测试

columns_to_drop = ['Time Stamp', 'DD','ff10','N', 'WW','W2','Cl','Nh','H','Cm','Ch','E','Tg',"E'",'sss']
"""
难以处理的列，先扔了
"""

rows_of_test = list(range(6000, 8000))     #用于测试

times = 100         #迭代次数


def cal_distance(row, table, mean, col_remain):
    ret = 0
    for co in col_remain:
        ret += (mean[co] - table.loc[row, co])**2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model train')
    parser.add_argument('--model_path', type=str,
                        default='/home/hyliu/ML_Project/my_model.txt')
    args = parser.parse_args()
    df_data = pd.read_excel('./data/training_dataset.xls')
    df_data = df_data.drop(columns=columns_to_drop)
    column_remain = df_data.columns.tolist()
    column_remain.remove("RRR")
    print(column_remain)
    for i in range(0,8000):
        if(df_data.loc[i,"RRR"] == "无降水"):
            df_data.loc[i,"RRR"] = 0
        else:
            df_data.loc[i,"RRR"] = 1
    print(df_data.head())
    df_data = df_data.apply(pd.to_numeric, errors='coerce')
    test_data = df_data.iloc[6000:8000]
    test_data.reset_index(drop=True, inplace=True)
    #print(test_data)
    train_data = df_data.drop(rows_of_test,inplace=True)
    #train_data = train_data.apply(pd.to_numeric, errors='coerce')
    print(train_data)
    print(train_data.loc[6000,"T"])
    mean = train_data.mean()
    print(mean)
    for co in column_remain:
        #print(train_data.loc[0,co])
        for i in range(0,train_data.shape[0]-1):       # 将空值赋值为平均值
            print(train_data.loc[i,co])
            print(i,co)
            if pd.isna(train_data.loc[i,co]):
                train_data.loc = mean[co]
        for i in test_data.shape[0]:
            if pd.isna(test_data.loc[i,co]):
                test_data.loc = mean[co]
    
    mean_true = {key: 0 for key in column_remain}           #正样例的中心
    mean_false = {key: 0 for key in column_remain}          #负样例的中心
    next_true = {key: 0 for key in column_remain}           #正样例的中心迭代
    next_false = {key: 0 for key in column_remain}          #负样例的中心迭代
    cnt_true = 0
    cnt_false = 0
    for i in range(0,6000):
        if train_data.loc[i,"RRR"] == 0:
            cnt_false += 1
        for co in column_remain:
            mean_false[co] += train_data.loc[i,co]
        if train_data.loc[i,"RRR"] == 1:
            cnt_true += 1
        for co in column_remain:
            mean_true[co] += train_data.loc[i,co]
    
    for _ in range(0,times):        #进行训练
        cnt_true = 0
        cnt_false = 0
        for i in range(0,train_data.shape[0]):
            dis_true = cal_distance(i,train_data,mean_true,column_remain)           #到正样例中心的距离
            dis_false = cal_distance(i,train_data,mean_false,column_remain)         #到负样例中心的距离
            if dis_true > dis_false:      #视为正样例
                for co in column_remain:
                    next_true[co] += train_data.loc[i,co]
                    cnt_true += 1
            else:
                for co in column_remain:
                    next_false[co] += train_data.loc[i,co]
                    cnt_false += 1
        for co in column_remain:
            mean_true[co] = next_true[co] / cnt_true
            mean_false[co] = next_false[co] / cnt_false

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, test_data.shape[0]):          #测试准确率
        dis_true = cal_distance(i,test_data,mean_true,column_remain)           #到正样例中心的距离
        dis_false = cal_distance(i,test_data,mean_false,column_remain)         #到负样例中心的距离
        if dis_true > dis_false and test_data.loc[i,"RRR"] == 1:
            TP += 1
        if dis_true < dis_false and test_data.loc[i,"RRR"] == 0:
            TN += 1
        if dis_true > dis_false and test_data.loc[i,"RRR"] == 0:
            FP += 1
        if dis_true < dis_false and test_data.loc[i,"RRR"] == 1:
            FN += 1

    print(TP,TN,FP,FN)
