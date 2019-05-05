---
layout:     post
title:      Text Classification（3）
subtitle:   文本分类二（TextCNN、TextRCNN、DPCNN)
date:       2019-01-05
author:     biggan
header-img: img/post-bg-swift.jpg
catalog: true
tags:
    - Text Classification
typora-root-url: ..
---

## 文本分类三

### 1.TextCNN

#### （1）概述

- 利用CNN进行文本分类，和TextRNN仅仅在文本表示上有区别，对文本向量分类的操作是相同的。
- TextCNN比TextRNN的主要优势在于训练速度快
- TextCNN重点
  - 将文本表示成词向量矩阵，称为文本词向量矩阵。
  - 利用若干数目不同尺寸的卷积核，对文本词向量矩阵进行信息提取，然后经池化层得到有效信息。
  - 将每一个卷积核提取到的有效信息拼接在一起得到文本的向量表示。

#### （2）网络结构

![textCnn](/img/textCnn.png)

#### （3）结构解析

&emsp;一共包含5层

- 词向量层

  &emsp;将文本表示成文本词向量矩阵。文本中的每个词，都将其映射到词向量空间，假设词向量为d维，则n 个词映射后，相当于生成一张n*d维的向量矩阵。

  **note**:

  实际项目中，每一个文本包含的单词数量不一定相同，若要批量处理，需要将所有文本填充至相同长度。

- 卷积层

  &emsp;使用多种尺寸的多个卷积核作用于词向量图，每个卷积核以一定步长在 词向量图上滑动，生成不同的feature map（一维向量），feature map向量的长度=$\frac{text\_len-kernal\_size}{stride}-1$

  其中，text_len为文本的长度（单词数），kernal_size为卷积核的长度，stride为卷积核移动的步长。

  **note**:

  - 卷积核是一个二维的参数矩阵，矩阵的宽度要求和词向量的维度d相同，长度可以任意指定。一般会取多种尺寸的卷积核，如[2,d],[3,d],[4,d]，这实际上提取的是文本的**n\_gram信息**。

  - 对于每一种尺寸的卷积核的数量都是可以任意指定的。

- pooling层

  &emsp;每一个卷积核作用在文本词向量矩阵上都会生成一个feature map向量，pooling层就是对这个feature map向量的信息进一步的筛选。常用有k-maxpooling，mean-pooling两种池化方式。

  - k-maxpooling

    k-maxpooling是取feature map向量中最大的k个点，构成一个长度为k的向量。

    **note**：k的值可以指定，一般取1~3。

  - mean-pooling

    k-maxpooling是对feature map向量上所有的点的值求平均，得到一个值。

- 连接层

  一个卷积核作用在文本词向量矩阵上都会生成一个feature map向量，一个feature map向量经过池化层都会得到一个点或者一个较短的向量。最后直接将这些点或者向量串接起来就得到了文本的向量表示。

- 输出层（分类）

  先接一个线性层将文本表示v映射到类别数目，再根据问题用softmax或者sigmoid输出各个类别的概率。

#### （4）网络搭建

```python
class textCNN(nn.Module):
    def __init__(self, args):
        super(textCNN, self).__init__()
        vocb_size = args['vocb_size']
        dim = args['dim']
        n_class = args['n_class']
        max_len = args['max_len']
        embedding_matrix=args['embedding_matrix']
        ker_size = [3, 4, 5, 6]
        in_channels = 256
        out_channels = 80
        self.topks = 1
        self.embeding = nn.Embedding(vocb_size, dim, _weight=embedding_matrix)    
        self.dropout = nn.Dropout()                                            
        self.conv1 = nn.Sequential(
                     nn.Conv1d(in_channels=in_channels, 			out_channels=out_channels, kernel_size=ker_size[0], stride=1),                                     
                     nn.BatchNorm1d(out_channels),                                
                     nn.ReLU(),
                     nn.MaxPool1d(kernel_size=max_len-ker_size[0]+1)
                     )
        self.conv2 = nn.Sequential(
                     nn.Conv1d(in_channels=in_channels, 		out_channels=out_channels, kernel_size=ker_size[1], stride=1),
                     nn.BatchNorm1d(out_channels),
                     nn.ReLU(),
                     nn.MaxPool1d(kernel_size=max_len-ker_size[1]+1)
                     )
        self.conv3 = nn.Sequential(
                     nn.Conv1d(in_channels=in_channels, 				 		out_channels=out_channels, kernel_size=ker_size[2], stride=1),
                     nn.BatchNorm1d(out_channels),
                     nn.ReLU(),
                     nn.MaxPool1d(kernel_size=max_len-ker_size[2]+1)
        )
        self.conv4 = nn.Sequential(
             nn.Conv1d(in_channels=in_channels,
	out_channels=out_channels, 	kernel_size=ker_size[3], stride=1),
             nn.BatchNorm1d(out_channels),
             nn.ReLU(),
             nn.MaxPool1d(kernel_size=max_len-ker_size[3]+1),
		)
		self.out =  nn.Sequential(
            nn.Linear(out_channels*4*self.topks, n_class),
            nn.Softmax()
		)

    def forward(self, x):
        x = self.embeding(x)    #[100, 64, 256]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  #[100, 256, 64]
        x1 = self.conv1(x)      #[100, 80, maxlen-ker_size] --> [100, 80, 1]
        x1 = x1.view(x1.size(0), -1)
        x2 = self.conv2(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.conv3(x)
        x3 = x3.view(x3.size(0), -1)
        x4 = self.conv4(x)
        x4 = x4.view(x4.size(0), -1)
        x = torch.cat((x1, x2, x3, x4), 1)
        output = self.out(x)
        return output
```



### 2.TextRCNN

- 先后利用RNN、CNN对文本进行信息提取。

- TextCNN和TextCNN的区别仅仅在于上文提到的词向量层。

  TextCNN在词向量层，是把文本表示成**词向量矩阵**，而TextCNN是把文本表示成**词隐状态向量矩阵**。即先将文本先输入RNN循环神经网络，得到每一个时刻（单词）对应的隐状态（输出)，然后用单词的隐状态替代词向量，将文本表示成词隐状态向量矩阵。



### 3.DPCNN

Deep Pyramid Convolutional Neural Networks

本质是一种深层的cnn，能有效表达文本长距离关系的复杂度词粒度的CNN。

#### （1）网络结构