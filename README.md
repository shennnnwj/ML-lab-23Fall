# ML-lab-23Fall
## Team Members: Runfeng Lin, Wenjie Shen, Jinghan Liu


2024.1.8修改：
Chinese_analyse.py: 分析各中文串对应降水概率并将中文映射到数字（降水率）

data_analyse.py: 之前分析数据的程序

global_value.py: 相当于头文件，里面存了文件路径等

main.py: 助教提供的格式代码，暂未使用，提交时要把代码改为对应格式

test_model.py: 测试模型，输出TP FP TN FN F1_score

train_model.py: 训练模型，将各种参数存在json文件的词典中

跑的顺序：Chinese_analyse.py → train_model.py → test_model.py

主要修改：
1、没有预测每天的降水，而是预测每一个时刻的降水。
2、已将各中文项纳入处理
3、将各列数据归一化到0~1范围
4、如果将降水视为正样例，F1分数为0.4068，如果将不降水视为正样例，F1分数为0.8586

可优化的点：
1、目前没有考虑各列的权重，可以根据相关系数给相关度大的列赋予更大的权重。
2、预测时不降水的情况基本能预测得到，但降水的情况容易被视为不降水，即预测偏向于预测不降水，可以加一个微调，使模型倾向于预测降水，或者改为对当天是否降水的预测（因为那样降水的比例会比较高，模型训练后也会偏向于给出降水的预测）。