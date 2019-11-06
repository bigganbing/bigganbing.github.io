## Learning Bert

BERT—— Bidirectional Encoder Representations from Transformers 

#### 1.Bert论文

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding





#### 2.Bert预训练模型

 谷歌开放了预训练的 **BERT-Base** 和 **BERT-Large** 模型，且每一种模型都有 **Uncased** 和 **Cased** 两种版本。 其中， Uncased版 在分词之前都转换为小写格式，并剔除所有 Accent Marker，而 Cased 会保留它们。因此，使用谷歌提供的Uncased版的预训练模型时，需要在预处理时对任务的语料做小写转换。作者表示一般使用 Uncased 模型就可以了，除非大小写对于任务很重要才会使用 Cased 版本。  

每一个预训练文件 包含了三部分，即**保存预训练模型与权重**的 ckpt 文件、将 WordPiece 映射到单词 id 的 **vocab** 文件，以及指定**模型超参数**的 json 文件。 

















