---
title: 
draft: false
tags: 
date: 2024-10-29
---
明白了！以下是一个完整、逻辑清晰、适合学习的 Obsidian 笔记格式的 Self-Attention 介绍。

---

# Self-Attention 机制

Self-Attention 是一种在自然语言处理和计算机视觉中常见的机制，用于捕捉输入序列中不同位置之间的相互关系。这种机制能够使模型在处理每个位置时关注序列中其他相关的位置，从而更好地理解上下文信息。

## 为什么使用 Self-Attention？

传统的序列模型（如 RNN）在处理长序列时难以捕捉远距离的依赖关系。Self-Attention 通过允许每个位置“关注”序列中的其他位置，提供了一种在全局范围内捕捉依赖关系的方式。对于自然语言处理等任务，这种全局关系非常重要。

## Self-Attention 的核心：Q, K, V

Self-Attention 的核心思想是通过 `Query` (Q)、`Key` (K)、和 `Value` (V) 三个矩阵来计算序列中各位置之间的关系。这个过程主要包括以下几步：

### 1. 输入表示

假设我们有一个序列输入 $X=[x_1, x_2, ..., x_n]$，其中 $x_i$ 是第 $i$ 个位置的词向量表示。对于每个位置，我们通过线性变换生成 `Query`、`Key` 和 `Value` 三个向量。

### 2. 生成 Q, K, V 矩阵

为了生成 Q, K, V，我们定义三个权重矩阵 $W_Q$、$W_K$ 和 $W_V$，然后将输入 $X$ 分别与这三个矩阵相乘，得到 Q, K, V 矩阵：

$$Q = X \cdot W_Q$$
$$K = X \cdot W_K$$
$$V = X \cdot W_V$$

- **Query (Q)**：表示我们当前在查找的信息。
- **Key (K)**：表示输入序列中可能被匹配的特征。
- **Value (V)**：表示实际被提取的内容。

### 3. 计算注意力权重 (Attention Weights)

对于每个位置的 Query，我们与其他位置的 Key 进行相似度计算，从而判断这个位置需要关注的其他位置。相似性通常通过点积来计算。

对于第 $i$ 个 Query 向量 $q_i$ 和第 $j$ 个 Key 向量 $k_j$，它们的相似性计算公式如下：

$$\text{attention}_{i,j} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

这里 $d_k$ 是 Key 的维度，用 $\sqrt{d_k}$ 进行缩放可以防止点积值过大导致数值不稳定。

### 4. 归一化注意力权重 (Softmax)

为了得到权重的概率分布，我们对所有相似性分数应用 Softmax：

$$\text{attention\_weights}_{i,j} = \text{softmax}(\text{attention}_{i,j})$$

通过 Softmax 操作，我们可以确保所有权重之和为 1，从而构成一个概率分布，表示每个位置对其他位置的关注程度。

### 5. 加权求和得到输出

有了注意力权重后，我们对每个位置的 Value 向量进行加权求和，从而得到输出向量：

$$\text{output}_i = \sum_{j=1}^{n} \text{attention\_weights}_{i,j} \cdot v_j$$

这样，Self-Attention 的最终输出是一个新的向量序列，每个向量包含了该位置与其他位置的上下文信息。

### Self-Attention 的总公式

Self-Attention 的整个过程可以使用以下公式概括：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

- $Q \cdot K^T$：计算 Query 和 Key 之间的相似性。
- $\sqrt{d_k}$：缩放因子。
- `softmax`：转换为概率分布。
- 最后乘上 $V$，得到最终的加权结果。

---

## 实现 Self-Attention

为了更好地理解 Self-Attention，我们可以使用 `TensorFlow` 实现一个基本的 Self-Attention 类。这个类包含 `__init__` 和 `call` 方法。

```python
# Author: Aemon Wang
# Email: aemooooon@gmail.com
# This code implements a basic self-attention layer in TensorFlow.

import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v):
        """
        Initialize the SelfAttention layer.
        d_k: Dimension of the key (and query) vectors.
        d_v: Dimension of the value vectors.
        """
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        
        # Initialize weight matrices for Q, K, V
        self.W_q = tf.keras.layers.Dense(d_k)
        self.W_k = tf.keras.layers.Dense(d_k)
        self.W_v = tf.keras.layers.Dense(d_v)

    def call(self, inputs):
        """
        Forward pass of the SelfAttention layer.
        inputs: A tensor of shape (batch_size, sequence_length, feature_dim)
        Returns: A tensor of shape (batch_size, sequence_length, d_v)
        """
        # Unpack inputs
        Q = self.W_q(inputs)  # (batch_size, seq_len, d_k)
        K = self.W_k(inputs)  # (batch_size, seq_len, d_k)
        V = self.W_v(inputs)  # (batch_size, seq_len, d_v)

        # Step 1: Compute the attention scores
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch_size, seq_len, seq_len)
        scores /= tf.sqrt(tf.cast(self.d_k, tf.float32))  # Scale by sqrt(d_k)

        # Step 2: Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (batch_size, seq_len, seq_len)

        # Step 3: Multiply the weights with the values
        output = tf.matmul(attention_weights, V)  # (batch_size, seq_len, d_v)

        return output, attention_weights
```

### 代码解释

1. **初始化**：
   - `d_k` 和 `d_v` 分别是 `Key/Query` 和 `Value` 的维度。
   - `W_q`、`W_k`、`W_v` 是 `Query`、`Key` 和 `Value` 的线性变换矩阵，通过 `tf.keras.layers.Dense` 定义。
   
2. **`call` 方法**：
   - **Step 1**：通过 `W_q`、`W_k`、`W_v` 计算 `Q`、`K`、`V`。
   - **Step 2**：通过点积计算 `Q` 和 `K` 的相似性并缩放。
   - **Step 3**：对相似性得分应用 Softmax，得到注意力权重。
   - **Step 4**：将注意力权重与 `V` 相乘，得到输出。

Self-Attention 机制的基本思想是通过 Q、K、V 的计算，使模型能够关注序列中不同位置的相关性，从而在全局范围内捕捉上下文依赖。这一机制极大地提升了序列建模的能力，特别是用于捕捉长距离依赖和灵活的上下文信息。在 Transformer 中，Self-Attention 是一个核心模块，也奠定了 BERT 和 GPT 等现代深度学习模型的基础。

---

## Multi-Head Attention

在实际应用中，我们通常会使用多头注意力（Multi-Head Attention）。多头注意力通过多个 Q、K、V 头部来捕捉不同层次的特征信息。每个头独立地计算 Self-Attention，最终将所有头的输出连接起来。这种方法能够使模型关注序列中多个不同位置的信息，进一步增强模型的表达能力。
### Implementation

下面是 `MultiHeadAttention` 类的代码，它会使用多个 `SelfAttention` 层并将输出拼接。

```python
# Author: Aemon Wang
# Email: aemooooon@gmail.com
# This code implements a basic multi-head attention layer in TensorFlow using the SelfAttention layer.

import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        """
        Initialize the MultiHeadAttention layer.
        num_heads: Number of attention heads.
        d_model: Total dimension of the model (embedding dimension).
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Dimension of each head
        self.depth = d_model // num_heads
        
        # Layers to project inputs to multi-head attention
        self.W_q = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        self.W_k = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        self.W_v = [tf.keras.layers.Dense(self.depth) for _ in range(num_heads)]
        
        # Final linear layer to combine the heads' outputs
        self.linear = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        """
        Forward pass of the MultiHeadAttention layer.
        inputs: A tensor of shape (batch_size, sequence_length, feature_dim)
        Returns: A tensor of shape (batch_size, sequence_length, d_model)
        """
        # Calculate attention heads
        heads_output = []
        for i in range(self.num_heads):
            # Calculate Q, K, V for this head
            Q = self.W_q[i](inputs)
            K = self.W_k[i](inputs)
            V = self.W_v[i](inputs)
            
            # Create a SelfAttention instance for each head
            attention_layer = SelfAttention(self.depth, self.depth)
            head_output, _ = attention_layer.call(inputs)  # Ignore attention weights for now
            heads_output.append(head_output)
        
        # Concatenate all heads' outputs along the last dimension
        concatenated_heads = tf.concat(heads_output, axis=-1)  # (batch_size, sequence_length, d_model)
        
        # Final linear layer to combine the multi-head outputs
        output = self.linear(concatenated_heads)
        
        return output
```

### 代码解释

1. **初始化参数**：
   - `num_heads` 是注意力头的数量。
   - `d_model` 是模型的总维度（也可以理解为嵌入维度）。
   - 我们将 `d_model` 均分给每个头，因此每个头的维度为 `depth = d_model // num_heads`。

2. **多头设置**：
   - 使用 `num_heads` 个 `Dense` 层分别生成每个头的 `Q`、`K` 和 `V`，确保每个头都是独立计算的。
   
3. **前向传播**：
   - 通过一个循环调用 `SelfAttention` 类计算每个头的输出，将 `Q`、`K` 和 `V` 传入 `SelfAttention`。
   - 所有头的输出通过 `tf.concat` 在最后一个维度上拼接起来。

4. **线性变换**：
   - 最终的输出会通过一个 `Dense` 层来映射到 `d_model` 维度，确保输出形状与输入保持一致。

### 示例使用

假设我们有一个批量大小为 `2`、序列长度为 `5`、嵌入维度为 `16` 的输入，并设置 `num_heads=4` 和 `d_model=16`：

```python
# Example usage
batch_size = 2
sequence_length = 5
d_model = 16
num_heads = 4

# Create random input
inputs = tf.random.normal((batch_size, sequence_length, d_model))

# Initialize multi-head attention layer
multi_head_attention = MultiHeadAttention(num_heads, d_model)

# Get output
output = multi_head_attention(inputs)

print("Output shape:", output.shape)  # Expected shape: (batch_size, sequence_length, d_model)
```

### 总结

`MultiHeadAttention` 的实现依赖于 `SelfAttention` 类，通过 `num_heads` 个独立的 `SelfAttention` 实例对输入进行多头处理，然后拼接所有头的输出。多头注意力使模型能够从不同的角度分析输入序列的相互关系，提高对复杂模式的捕捉能力。
