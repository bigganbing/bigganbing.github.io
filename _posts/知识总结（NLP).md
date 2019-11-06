### 知识总结（NLP)

#### 1.NLP四大类任务

<img src="../img/v2-ec6f99a704502efe7cb38febb4f8b5d9_hd.jpg" alt="v2-ec6f99a704502efe7cb38febb4f8b5d9_hd" style="zoom:80%;" />

引用自： https://zhuanlan.zhihu.com/p/54743941 

#### 2. NLP特征抽取器

- $CNN$
  - CNN捕获的是n-gram片段信息（n为卷积核尺寸，即覆盖的单词的数量），单卷积层无法捕获远距离特征。
  - CNN没有序列依赖关系，适合并行。
  - CNN的卷积层会保留序列相对位置信息，但是pooling层一般会破坏这一点，因此有时也需要在CNN网络中添加位置编码。
  - CNN捕获远距离特征的方式：
    - 增大卷积核大小
    - Dilated CNN （膨胀卷积）
    - 加深CNN网络深度



- $RNN \to LSTM、GRU \to RNNs+Attention$

  - RNNs天然适合不定长的线性序列结构，不需要位置向量加持。
  - 传统RNN存在**梯度消失**、**对长期依赖不敏感**的问题，使用LSTM、GRU能得到一定的缓解。

  - RNNs  + Attention机制，能更进一步学习长距离的信息。




-  $Transformer （self-attetion）$

  - 优势： ①**并行能力** 、②捕捉长距离信息和特征抽取能力（self-attention）
  - 劣势：不能很好的体现序列的先后关系，添加位置向量（ position embedding ）进行弥补



#### 2.深度学习一些trick

- 参数初始化（正态分布>平均？）
- 标准化（batchnormal、layernormal)


$$
y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$


- 残差连接
- Highway
- warm up
- learning_rate、batch_size影响



#### 问题荟萃

##### 1. 为什么LSTM能缓解梯度消失？

每一个时刻的输出为输出门系数\*（遗忘门系数\*当前学习到的信息+更新门系数\*上一时刻的记忆），其中的系数计算，加法运算等，增加了梯度回传的路径。即链式求导路径增加，对同一参数，不同路径所求偏导相加，使得梯度较一条路径大，不易接近于0。