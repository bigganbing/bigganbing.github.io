---
layout:     post
title:      Text Classification（2）
subtitle:   文本分类二（RNN+Attention)
date:       2018-12-05
author:     biggan
header-img: img/post-bg-swift.jpg
catalog: true
tags:
    - Text Classification
typora-root-url: ..
---

## 文本分类二

### 1.深度学习方法做文本分类

- 使用深度学习方法进行文本分类时，可以使用word2vec，glove，fasttext等工具预训练词向量，或者直接使用开源的训练好的词向量。
- 深度学习方法一般都是基于CNN、RNN以及Attention机制对文本进行特征提取，表示成文本向量，最后接一个softmax层进行分类。

### 2.利用RNN、Attention机制进行文本分类

#### （1）TextRNN

&emsp;本质上是对seq2seq Translation模型Encoder部分的应用。

- 表示：

  1. 直接取encoder最后一个时刻的输出$h_n$，作为文本的向量表示。
  2. 将encoder各个时刻的输出加和取平均，作为文本的向量表示。

  **note**:

  一般选用LSTM、GRU较裸的RNN要好一点，还可以指定单向/双向，单层/多层等。

- 分类

  根据多分类还是二分类问题，输出层选用softmax或者sigmoid。

#### （2）TextRNN+Attention机制

&emsp;本质是对上文TextRNN的第二种文本表示方式的该进。

- 表示

  - 思路

    文本的向量表示不再取encoder各个时刻的输出的平均，而是认为不同单词对文本主题的反应程度不同，给每个时刻的输出乘以一个权重，故文本表示为encoder各个时刻的输出的加权平均。

  - 模型结构

    ![textRNN_attn](/img/textRNN_attn-1556979692010.jpg)

    - word
      embedding

      - 词向量表$W_e$可以使用Word2vec等工具预训练的词向量，也可以先随机初始化一个词向量表，然后词向量随模型一起训练。 
      - 对于单词$w_t$，其词嵌入$x_{it}$如下

      $$
      x_{t}=W_ew_{t},t\in[1,T]
      $$

    - word
      encoder

      - 双向gru编码

      句子中的每个单词就是gru中的一个时刻，先查询词向量表, 得到单词对应的词向量，然后将词向量喂 给这个gru，然后可以得到一个正向的输出和一个反向的输出，接着把正向和反向的输出向量串接起来，作为当前时刻的输出。
      $$
      \vec{h_{t}}=\vec{gru}(x_t),t\in[1,T]	\\
      \overleftarrow {h_t}=\overleftarrow{gru}(x_t),t\in[1,T]\\
      h_t=[\vec{h_{t}},\overleftarrow {h_t}]
      $$

    - Word Attention

      因为文本中每个单词对文本主题的重要性不相同，因此使用Attention机制描述每个单词的重要性。对word encoder中每个单词的输出$h_t$乘以一个权重$\alpha_t$ ，再求和作为文本的向量表示$s$。
      $$
      u_t=tanh(W_w h_t+b_w)\\
      \alpha_t=\frac{exp(u_{t}^Tu_w)}{\sum_t exp(u_{k}^Tu_w)}\\
      s=\sum_t \alpha_t h_t
      $$

      - $tanh(W_w h_t+b_w)$相当于一个线性层
      - $\alpha_t=\frac{exp(u_{t}^Tu_w)}{\sum_t exp(u_{k}^Tu_w)}$其实是一个点乘+一个softmax层
      - $u_w$是一个和模型一起训练的参数。

- 分类

  根据多分类还是二分类问题，输出层选用softmax或者sigmoid。

- 网络搭建

  ```python
  class Word_attn(nn.Module):
      def __init__(self, lang_size, hidden_size, pretrained_weight, class_num):
          super(Word_attn, self).__init__()
          self.hidden_size = hidden_size
          self.class_num = class_num
          self.embedding = nn.Embedding(lang_size, hidden_size)
          # print(pretrained_weight.size())
          self.embedding.weight.data.copy_(pretrained_weight)
          self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)
          self.h_linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
          self.a_linear = nn.Linear(2*self.hidden_size, 1, bias=False)
          self.output = nn.Linear(2*hidden_size, self.class_num)
          self.soft_max = nn.Softmax()
  
      def forward(self, input, hidden0):
          embedded = self.embedding(input)    
          # [batch, seq] -->  [batch, seq, feature]
          input = embedded.permute(1, 0, 2)  #[seq, batch, feature]
          outputs, hidden = self.gru(input, hidden0)  
  		#[seq, batch, 2*hidden_size], [2, batch, hidden_size]
          x = outputs.permute(1, 0, 2)
          x = self.h_linear(x)    #[batch, seq, 1, 2*hidden_size]
          x = self.a_linear(x)    #[batch, seq, 1]
          x = self.soft_max(x)
          attn_weight = x.permute(0, 2, 1)    #[batch,1,seq]
          sent_wv = torch.bmm(attn_weight, outputs.permute(1, 0, 2))  
  		#[batch,1,seq] * [batch, seq, 2*hidden_size]  == [batch, 1, 2*hidden_size]
          # output_class = self.output(sent_wv)         #[batch, 1, class_num]
          return sent_wv
  
      def initHidden(self, batch_size):
          result = Variable(torch.zeros(2, batch_size, self.hidden_size))  
  									#训练集并不正好按batch_size划分
          if use_cuda:
              return result.cuda()
          else:
              return result
  
  ```

  

