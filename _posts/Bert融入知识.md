---
typora-root-url: ..
---

###  Bert融入知识

本文主要介绍两篇都叫ERNIE的论文，两者都通过融入知识对Bert模型进行改进。本文主要关注两论文对应模型是**如何融入知识**的、以及**模型的预训练任务**。

#### 1. Enhanced Language Representation with Informative Entities 

##### **1.1 论文原文**

 https://www.aclweb.org/anthology/P19-1139.pdf 

##### 1.2 论文摘要

 在大规模语料库上预先训练的BERT等神经语言表示模型可以很好地从纯文本中捕获丰富的语义模式，并通过微调的方式一致地提高各种NLP任务的性能。然而，**现有的预训练语言模型很少考虑融入知识图谱(KGs)**，知识图谱可以为语言理解提供丰富的结构化知识。我们认为知识图谱中的信息实体可以通过外部知识增强语言表示。**在这篇论文中，我们利用大规模的文本语料库和知识图谱来训练一个增强的语言表示模型(ERNIE)，它可以同时充分利用词汇、句法和知识信息。**实验结果表明，ERNIE在各种知识驱动任务上都取得了显著的进步，同时在其他常见的NLP任务上，ERNIE也能与现有的BERT模型相媲美。本文的源代码和实验细节可以从 https://github.com/thunlp/ERNIE 获得。 

##### 1.3 两大挑战及解决方法： 

（1） 将外部知识融入语言表示模型将面临两大挑战。（这是论文里说的）

-  Structured Knowledge Encoding （ 结构化知识编码 ）

   对于给定的文本，如何高效地抽取并编码对应的知识图谱中的信息； 

-  Heterogeneous Information Fusion（ 异构信息融合 ）

  语言表示的预训练过程和知识表示过程有很大的不同，它们会产生两个独立的向量空间。因此，如何设计一个特殊的预训练目标，以融合词汇、句法和知识信息又是另外一个难题。 

 （2）为此，本文提出了ERNIE模型，同时在大规模语料库和知识图谱上预训练语言模型：

-  抽取、编码知识信息

  首先识别文本中的实体，并将这些实体与知识图谱中相应的实体对齐。具体做法：采用知识嵌入算法（如**TransE**） 对KGs的图结构进行编码 ，并将得到的**信息实体embedding作为ERNIE模型的输入**。基于文本和知识图谱的对齐，ERNIE 将知识模块的实体表征整合到语义模块的底层。 

-  语言模型的预训练任务

    在训练语言模型时， 除了采用Bert中的**遮蔽语言模型（MLM）**和**下一个句子预测（NSP)**作为预训练的目标，为了更好地融合文本和知识特征，还设计了一个**新的预训练目标**——**dEA**：在输入文本中随机屏蔽一些实体，要求模型从KGs中选出适当的实体来完成对齐。

##### 1.4  模型结构

![TIM截图20191109115958](/img/TIM截图20191109115958.jpg)

**模型结构主要包含两个层叠的模块，T-Encoder 和 K-Encoder。**

- **T-Encoder**（ textual encoder ）

  结构同Bert，包含Nx层，每一层包含两个子层：一个Muti-Head Attention子层、一个Feed Forward子层。

  T-Encoder的输入为文本对应的 tokens序列，输出为与文本中各个token对应的隐状态序列，用于捕获输入文本中的词法和语法信息。

- **K-Encoder**（ knowledgeable encoder ）

  K-Encoder是模型体现**融入知识**的主要结构，包含Mx层，每一层包含三个子层：

  - 一个以T-Encoder的输出（Token Input）为输入的Muti-Head Attention子层，

  - 一个以**实体Embedding序列**（Entity input：通过TranE等方法得到）为输入的Muti-Head Attention子层，

  - 一个融合文本信息和实体信息的**Information Fusion**层，输出为Token Output，Entity Output分别与Token Input、Entity input维度相同。

**模型对应计算过程：**

设文本对应token序列为 {w1, . . . , wn} ， entity 序列为 {e1, . . . , em} 。

- **{w1, . . . , wn}** = T-Encoder({w1, . . . , wn})                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

  token序列{w1, . . . , wn}先经过T-Encoder层，同Bert（一个多层双向 Transformer encoder 结构），在输入时，对每个token的 token embedding, segment embedding, positional embedding 相加作为input embedding，输出的**{w1, . . . , wn}** 为与文本每个token对应的隐状态序列。

-  **{e1, . . . , em}** =  TransE ({e1, . . . , em})

  entity 序列 {e1, . . . , em} 先通过一个知识嵌入模型（如 TransE），变成一个entity embedding 序列 **{e1, . . . , em}** 。

-  **{w1, . . . , wn}**, **{e1, . . . , en}** = K-Encoder( **{w1, . . . , wn},** **{e1, . . . , em})**.  

  K-Encoder以T-Encoder的输出 **{w1, . . . , wn}**、通过 TransE得到的 **{e1, . . . , em}** 为输入，输出仍为和输入同维度的 **{w1, . . . , wn}**和**{e1, . . . , en}**。 

  K-Encoder的具体计算如下：

  - 
