# %%
import pandas as pd
import json
import numpy as np

# %%
train_data_path = "./data/train.xls"

test_data_path = "./data/test.xls"

columns_to_drop = ['Time Stamp','E','Tg',"E'",'sss','tR','ff10']

Chinese_num_map_path = "./Model/Chinese_num_map.json"

def day_rainy(dataset):
    """
    处理RRR列表并将对时刻的预测转化为对天的预测
    """
    yesterday = 0
    begin = 0       #某天开始的位置
    flag = 0        #记录当天是否有雨
    for index, row in dataset.iterrows():
        """
        对降水列进行处理
        """
        if(dataset.loc[index,"RRR"]!=dataset.loc[index,"RRR"]):         #数据为nan
            dataset.loc[index,"RRR"] = -1
            #break                  #如果break，则将后面记录不全的数据舍弃掉
        elif(dataset.loc[index,"RRR"] == "无降水"):
            dataset.loc[index,"RRR"] = 0
        else:
            dataset.loc[index,"RRR"] = 1

    for index, row in dataset.iterrows():
        today = dataset.loc[index,"Time Stamp"][0:10]
        if today != yesterday:          #进入下一天
            if flag == 1:               #当天下雨了，将那天的RRR全部改为1
                for i in range(begin,index):
                    dataset.loc[i,"RRR"] = 1
            flag = 0
            begin = index
            yesterday = today
        if dataset.loc[index,"RRR"] == 1:       #当天是下雨的
            flag = 1
    index += 1
    if flag == 1:               #当天下雨了，将那天的RRR全部改为1
        for i in range(begin,index):
            dataset.loc[i,"RRR"] = 1
    return dataset

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
    
def initialize_parameters(layers_dims):
    """
    Arguments：
        layers_dims -- list, including the numbers of nodes in every single layer

    Returns:
        Weight matrix and bias vector of every single layer
    """
 
    np.random.seed(42)
    parameters = {}
    L = len(layers_dims)
 
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
 
        assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))
 
    return parameters

def compute_loss(a3, Y):
    m = Y.shape[1]
    epsilon = 0.1
    logprobs = np.multiply(-np.log(a3 + epsilon),Y) + np.multiply(-np.log(1 - a3 + epsilon), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    
    return loss

def forward_propagation(X, parameters):
    """
    Arguments:
    X -- input dataset, of shape
    Y -- "RRR" vector
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape ()
                    b1 -- bias vector of shape ()
                    W2 -- weight matrix of shape ()
                    b2 -- bias vector of shape ()
                    W3 -- weight matrix of shape ()
                    b3 -- bias vector of shape ()
    
    Returns:
    loss -- the loss function (vanilla logistic loss)
    """
        
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    z1 = np.dot(W1, X) + b1
    a1 = leaky_relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = leaky_relu(z2) # because all the attributes are greater than 0, leaky_relu is equal to relu -> make bp easier
    z3 = np.dot(W3, a2) + b3 
    a3 = sigmoid(z3) # a3 is the output of the model
    
    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)

    print(W1)
    print(b1)
    print(W2)
    print(b2)
    print(W3)
    print(b3)
    print(z1)
    print(a1)
    print(z2)
    print(a2)
    print(z3)
    print(a3)
    input()
    
    return a3, cache
    
def backward_propagation(X, Y, cache):
    """
    Arguments:
    X -- input dataset, of shape
    Y -- true "label" vector
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1./m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims = True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims = True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims = True)
    
    gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                 "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                 "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
    
    return gradients

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients, output of n_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters['W' + str(i)] = ... 
                  parameters['b' + str(i)] = ...
    """
    
    L = len(parameters) // 2 # the number of layers in the neural networks
    for k in range(L):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
        
    return parameters

def train_neural_network(X, Y, hidden_size, output_size, epochs, learning_rate):
    input_size = X.shape[0]
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    print(W1.dtype)
    for epoch in range(epochs):
        # Forward Propagation
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)

        # Compute Loss
        loss = compute_loss(Y, A2)

        # Backward Propagation
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        print(dW1.dtype)
        # Update Parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        # Print Loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return W1, b1, W2, b2

# %%
df_test = pd.read_excel(test_data_path)
df_test = day_rainy(df_test)
df_data_test = df_test.drop(columns=columns_to_drop)
columns_test = df_data_test.columns.tolist()
columns_test.remove("RRR")
columns_test

# %%
with open(Chinese_num_map_path, "r") as f:
    Chinese_to_num_map = json.load(f)
    for index, row in df_data_test.iterrows():       
        for key_co in Chinese_to_num_map:
            submap = Chinese_to_num_map[key_co]
            if(df_data_test.loc[index,key_co] in submap):
                df_data_test.loc[index,key_co] = submap[df_data_test.loc[index,key_co]]

df_data_test = df_data_test.apply(pd.to_numeric, errors='coerce')
df_data_test

# %%
df_train = pd.read_excel(train_data_path)
df_train = day_rainy(df_train)          #处理降水项
df_data_train = df_train.drop(columns=columns_to_drop)
columns_train = df_data_train.columns.tolist()
columns_train.remove("RRR")

with open(Chinese_num_map_path, "r") as f:
    Chinese_to_num_map = json.load(f)
    
    for index, row in df_data_train.iterrows():       #将中文转化为数字
        for key_co in Chinese_to_num_map:
            submap = Chinese_to_num_map[key_co]
            if(df_data_train.loc[index,key_co] in submap):
                df_data_train.loc[index,key_co] = submap[df_data_train.loc[index,key_co]]
df_data_train = df_data_train.fillna(value=df_data_train.mean())
df_data_train

# %%
X_train = df_data_train.to_numpy()
Y_train = df_data_train['RRR'].to_numpy()
Y_train = Y_train.reshape(1, -1).T

grads = {}
costs = []
m = X_train.shape[1]
layers_dims = [X_train.shape[0], 10, 5, 38774]
learning_rate = 0.01
num_iterations = 20000

parameters = initialize_parameters(layers_dims)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

for i in range(0, num_iterations):
    # 前向传播
    a3, cache = forward_propagation(X_train, parameters)
    
    # 计算成本
    cost = compute_loss(a3, Y_train)
    
    # 反向传播
    grads = backward_propagation(X_train, Y_train, cache)
    
    # 更新参数
    parameters = update_parameters(parameters, grads, learning_rate)

    # 记录成本
    if i % 1000 == 0:
        costs.append(cost)
        # 打印成本
        print("Epoch:" + str(i) + ", Loss:" + str(cost))

# %%


# %%



