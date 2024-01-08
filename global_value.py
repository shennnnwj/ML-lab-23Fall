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
    #print(ret)
    return ret