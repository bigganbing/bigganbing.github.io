---
typora-root-url: ..
---

### NLP预训练语言模型

##### 1. 统计语言模型——文本单词序列的联合概率

- 对语句$s = w_1 w_2 ... w_m$，其先验概率为：

$$
p(s)=p(w_1)×p(w_2|w_1)×p(w_3|w_1w_2)×...×p(w_m|w_1w_2...w_{m-1})
= \prod_{i=1}^mp(w_i|w_1^{i-1})
$$

- **语言模型应用**：

  （1 中文拼音输入法，选择最可能的与拼音对应的汉字短语。

  （2 拼写检查， 对用户的输出句子进行实时检查 

  （3 机器翻译或语音识别中，在多个候选答案中选择最可能的翻译。

  

##### 2. n-gram语言模型

- **思想**：每一个词出现的概率，只依赖前n-1个词，即基于 **n阶马尔科夫假设** ，能够**降低概率估算难度，并减少参数量**。即认为：$p(w_i|w_1^{i-1})=p(w_i|w_{i-n+1}^{i-1})$，则

$$
p(s)=\prod_{i=1}^{m} p\left(w_{i} | w_{i-n+1}^{i-1}\right)
$$

- **示例**：

  1-gram，$p(s)=p(w_1)×p(w_2)×…×p(w_m)=\prod_{i=1}^mp(w_i)$

  2-gram，$p(s)=p(w_1)×p(w_2|w_1)×…×p(w_m|w_{m-1})=\prod_{i=1}^mp(w_i|w_{i-1})$

-  **问题**：

  数据稀疏引起零概率问题——解决：数据平滑技术，如 Add-one (Laplace) Smoothing 
  
  

##### 3. 神经网络语言模型（NNLM）

- **特点**： 引入了神经网络的n-gram语言模型，以**给定前n-1个词，预测下一个词**为训练任务，同时学习**词的分布式表示**（词向量）和**词序列的概率**。（词向量是模型的副产品）

- **代表作**：

  Begio等人在2003 年发表的 [ A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

  模型结构如下：

  <img src="/img/1-1572954838111.png" alt="1" style="zoom:50%;" />

  数值计算如下：
  $$
  x=\left(C\left(w_{t-1}\right), C\left(w_{t-2}\right), \cdots, C\left(w_{t-n+1}\right)\right) \\
  y=b+W x+U \tanh (d+H x) \\
  p=Softmax(y)
  $$
  **Note**：改论文是使用FFNN实现的语言模型，往后也有人使用RNN等结构进行改进。

  

##### 4. Word2Vec、FastText、Glove等 Word Embedding 技术

**Note**：这些方法生成的词向量是静态的， 未考虑一词多义、无法理解复杂语境 。

- **Word2Vec**（**用某词的上下文替换NNLM中的前n-1个词**）

   Mikolov等人在2013年发表 [Efficient Estimation of Word Representations in Vector Space]( https://arxiv.org/pdf/1301.3781.pdf )

  （1）两种模型（预训练任务不同）：

  - cbow：由上下文预测中心词

  - skip-gram：由中心词预测上下文

    <img src="/img/20181201152905657.png" alt="20181201152905657" style="zoom:60%;" />

  （2）两个训练方法（替换softmax层）：

  -  Hierarchical Softmax
  -  Negative Sampling

- **FastText**

  FastText相对于Word2Vec考虑了subword 信息，这对未登录词更友好。

  FastText在github上开源工具：[fastText](https://github.com/facebookresearch/fastText)

- **Glove**

  基于共现矩阵。

  

##### 5. ELMO、GPT、BERT等预训练语言模型 

###### 5.1 分类：

- **单向**特征表示的**自回归**预训练语言模型，统称为**单向模型**：
  - ELMO、ULMFiT、SiATL、GPT1.0、GPT2.0
- **双向**特征表示的**自编码**预训练语言模型，统称为**BERT系列模型：**
  - BERT、MASS、UNILM、ERNIE1.0、ERNIE(THU)、MTDNN、ERNIE2.0、SpanBERT、RoBERTa
- **双向**特征表示的**自回归**预训练语言模型：
  - XLNet

<div align="center"><img src="/img/v2-adf45870fa647599bd4332efd2b44964_hd.jpg" alt="v2-adf45870fa647599bd4332efd2b44964_hd" style="zoom:80%;" /></div>

###### 5.2 比较

（1） 特征抽取机制不同

- RNNs：ELMO/ULMFiT/SiATL；

- Transformer：BERT系列模型采用 Transformer  encoder部分， GPT 采用 Transformer  decoder部分。

- Transformer-XL：XLNet；

   **长距离依赖建模能力**： Transformer-XL > Transformer > RNNs > CNNs 

  **并行能力**：Transformer-XL 和 Transformer > RNNs



（2）预训练语言目标不同

- 自编码（AutoEncode）：BERT系列模型；

- 自回归（AutoRegression）：单向模型（ELMO/ULMFiT/SiATL/GPT1.0/GPT2.0）和XLNet；

  自回归适合处理自然语言**生成任务** 



训练任务不同



（3）单/双向语言模型

- 单向模型： ELMO/ULMFiT/SiATL/GPT1.0/GPT2.0 。

  其中ELMO/ULMFiT/SiATL都是不完全双向模型，bi-LSTM使其仅在最后将两个方向的特征拼接起来，中间过程中，每一个词位上的特征还是仅仅依赖于一个方向。

- 双向模型：BERT系列、XLNet

   使用基于微调的方法处理token级别的任务（如QA、NER），融合两个方向上的信息是至关重要的。









参考： https://zhuanlan.zhihu.com/p/76912493 

