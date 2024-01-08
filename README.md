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