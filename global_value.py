"""
全局变量库
"""

train_data_path = "./data/train.xls"
"""
训练集位置
"""

test_data_path = "./data/test.xls"
"""
测试集位置
"""

columns_to_drop = ['Time Stamp','E','Tg',"E'",'sss','tR','ff10']
"""
训练过程中丢弃的数据列（没什么用的数据列）
"""

Chinese_num_map_path = "./Model/Chinese_num_map.json"
"""
记录每一列中字符串到数字的转换，转换值为降水率
"""

Model_para_path = "./Model/parameter.json"
"""
记录模型的各种参数
用dict来存储训练后的模型参数，分别有该列最大值max，最小值min，
在2-means模型中正项的中心T_center，负项的中心F_center，两者的中点center(center为规范化后的0~1)
在预测时，如果遇到空数据，则将该项置为center，不影响预测，
如果遇到非空数据，就将该数据减去min然后除以max-min，映射到0~1区域进行预测，以确保各列数据影响一致
"""

epoches = 20
"""
训练轮数
"""

def cal_distance(row_idx, table, center, columns):
    """
    计算table的第row_idx行到某个中心点center的距离
    """
    ret = 0
    for co in columns:
        ret += (center[co] - table.loc[row_idx, co])**2

    ##权重调整，使相关度高的项目权重更高
    ret += 2*(center["U"] - table.loc[row_idx,"U"])**2
    ret += 2*(center["WW"] - table.loc[row_idx,"WW"])**2
    ret += (center["P"] - table.loc[row_idx,"P"])**2
    ret += (center["Td"] - table.loc[row_idx,"Td"])**2
    #print(ret)
    return ret

def to_day(dataset):
    """
    转化为对天的预测，供day_rainy调用
    """
    yesterday = 0
    begin = 0       #某天开始的位置
    flag = 0        #记录当天是否有雨
    dataframe = dataset.copy()
    for index, row in dataframe.iterrows():
        today = dataframe.loc[index,"Time Stamp"][0:10]
        if today != yesterday:          #进入下一天
            if flag == 1:               #当天下雨了，将那天的RRR全部改为0
                for i in range(begin,index):
                    dataframe.loc[i,"RRR"] = 0
            flag = 0
            begin = index
            yesterday = today
        if dataframe.loc[index,"RRR"] == 0:       #当天是下雨的
            flag = 1
    index += 1
    if flag == 1:               #当天下雨了，将那天的RRR全部改为0
        for i in range(begin,index):
            dataframe.loc[i,"RRR"] = 0
    return dataframe

def day_rainy(dataset):
    """
    处理RRR列表并将对时刻的预测转化为对天的预测
    """

    for index, row in dataset.iterrows():
        """
        对降水列进行处理
        """
        if(dataset.loc[index,"RRR"]!=dataset.loc[index,"RRR"]):         #数据为nan
            dataset.loc[index,"RRR"] = -1
            #break                  #如果break，则将后面记录不全的数据舍弃掉
        elif(dataset.loc[index,"RRR"] == "无降水"):                 #晴天为1，雨天为0
            dataset.loc[index,"RRR"] = 1
        else:
            dataset.loc[index,"RRR"] = 0

    to_day(dataset)
    return dataset
        