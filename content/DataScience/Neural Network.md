---
title: 
draft: false
tags: 
date: 2024-07-21
---
## 神经网络基本概念

### 前向传播
前向传播是指输入数据通过网络各层进行计算，直到输出层获得预测结果。其数学表示为：
$$
z = Wx + b
$$
其中，$W$ 是权重矩阵，$x$ 是输入向量，$b$ 是偏置向量。

### 激活函数
激活函数引入非线性，使神经网络能够拟合复杂的函数。常见的激活函数有 ReLU、Sigmoid 和 Softmax。在本例中，我们使用 Softmax 作为输出层的激活函数：
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

### 损失函数
损失函数用于衡量模型预测结果与真实结果的差异。在分类问题中，常用的损失函数是交叉熵损失：
$$
\text{loss} = -\sum_{i} y_i \log(p_i)
$$
其中，$y_i$ 是真实标签，$p_i$ 是模型预测的概率。

### 梯度下降
梯度下降是一种优化算法，通过最小化损失函数来更新模型参数。其更新公式为：
$$
W \leftarrow W - \eta \frac{\partial \text{loss}}{\partial W}
$$
$$
b \leftarrow b - \eta \frac{\partial \text{loss}}{\partial b}
$$
其中，$\eta$ 是学习率。

### 反向传播
反向传播用于计算梯度，是神经网络训练的核心。通过链式法则，将损失函数对输出的梯度反向传递，计算各层参数的梯度。

#### 数学公式

1. **输出层**：
    计算损失对输出的梯度：
    $$
    \delta^{(2)} = a^{(2)} - y
    $$
    计算损失对参数的梯度：
    $$
    \frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} \cdot (a^{(1)})^T
    $$
    $$
    \frac{\partial L}{\partial b^{(2)}} = \delta^{(2)}
    $$

2. **隐藏层**：
    计算损失对激活值的梯度（使用链式法则）：
    $$
    \delta^{(1)} = (W^{(2)})^T \delta^{(2)} \cdot \text{ReLU}'(z^{(1)})
    $$
    其中，$\text{ReLU}'(z^{(1)})$ 是 ReLU 函数的导数：
    $$
    \text{ReLU}'(z^{(1)}) = \begin{cases} 
    1 & \text{if } z^{(1)} > 0 \\ 
    0 & \text{if } z^{(1)} \leq 0 
    \end{cases}
    $$
    计算损失对参数的梯度：
    $$
    \frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} \cdot (a^{(0)})^T
    $$
    $$
    \frac{\partial L}{\partial b^{(1)}} = \delta^{(1)}
    $$

### 代码实现

#### 数据加载
首先，定义加载数据的函数：
```python
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

def get_data(inputs_file_path, labels_file_path, num_examples):
    with gzip.open(inputs_file_path, 'rb') as inputfile:
        input_data = np.frombuffer(inputfile.read(), dtype=np.uint8, offset=16)
        input_data = input_data.reshape(num_examples, 28*28)
        input_data = input_data / 255.0
        print(f'input_data shape: {input_data.shape}')
        
    with gzip.open(labels_file_path, 'rb') as labelfile:
        label_data = np.frombuffer(labelfile.read(), dtype=np.uint8, offset=8)
        print(f'label_data shape: {label_data.shape}')

    return input_data, label_data

mnist_data_folder = './MNIST_data'
train_inputs, train_labels = get_data(
    os.path.join(mnist_data_folder, 'train-images-idx3-ubyte.gz'),
    os.path.join(mnist_data_folder, 'train-labels-idx1-ubyte.gz'),
    60000
)
test_inputs, test_labels = get_data(
    os.path.join(mnist_data_folder, 't10k-images-idx3-ubyte.gz'),
    os.path.join(mnist_data_folder, 't10k-labels-idx1-ubyte.gz'),
    10000
)
```

#### 参数初始化
初始化权重和偏置：
```python
input_size = 784
num_classes = 10
learning_rate = 0.05

W = np.random.rand(num_classes, input_size) * 0.01
b = np.zeros((num_classes, 1))
```

#### 前向传播
定义前向传播函数：
```python
def forward_pass(inputs, W, b):
    return np.dot(inputs, W.T) + b.T
```

#### 损失函数和梯度计算
定义计算损失和梯度的函数：
```python
def compute_loss_and_gradients(inputs, outputs, labels, W, b):
    num_samples = inputs.shape[0]
    y_true = np.eye(num_classes)[labels]

    exp_z = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    probs = np.clip(probs, 1e-10, 1.0)

    loss = -np.sum(y_true * np.log(probs)) / num_samples

    dL_dz = probs - y_true
    gradW = np.dot(dL_dz.T, inputs) / num_samples
    gradB = np.sum(dL_dz.T, axis=1, keepdims=True) / num_samples

    return loss, gradW, gradB
```

#### 参数更新
定义参数更新函数：
```python
def update_parameters(W, b, gradW, gradB, learning_rate):
    W -= learning_rate * gradW
    b -= learning_rate * gradB
    return W, b
```

#### 训练模型
定义训练模型的函数：
```python
def train_model(train_inputs, train_labels, W, b, learning_rate, batch_size, num_epochs):
    num_samples = train_inputs.shape[0]
    for epoch in range(num_epochs):
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            inputs = train_inputs[start:end]
            labels = train_labels[start:end]

            outputs = forward_pass(inputs, W, b)

            loss, gradW, gradB = compute_loss_and_gradients(inputs, outputs, labels, W, b)

            W, b = update_parameters(W, b, gradW, gradB, learning_rate)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')

    return W, b

num_epochs = 10
batch_size = 100
W, b = train_model(train_inputs, train_labels, W, b, learning_rate, batch_size, num_epochs)
```

#### 测试模型
定义测试模型的函数：
```python
def test_model(test_inputs, test_labels, W, b):
    outputs = forward_pass(test_inputs, W, b)
    predictions = np.argmax(outputs, axis=1)
    accuracy = np.mean(predictions == test_labels)
    return accuracy

accuracy = test_model(test_inputs, test_labels, W, b)
print(f'Test accuracy: {accuracy}')
```

#### 可视化结果
定义可视化预测结果的函数：
```python
def visualize_predictions(test_inputs, test_labels, W, b, num_samples=10):
    indices = np.random.choice(test_inputs.shape[0], num_samples, replace=False)
    sample_inputs = test_inputs[indices]
    sample_labels = test_labels[indices]

    outputs = forward_pass(sample_inputs, W, b)
    predictions = np.argmax(outputs, axis=1)

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(sample_inputs[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {sample_labels[i]}\nPred: {predictions[i]}')
        axes[i].axis('off')
    plt.show()

visualize_predictions(test_inputs, test_labels, W, b)
```
