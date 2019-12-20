---
layout:     post
title:      Common code segment
subtitle:   常用代码段
date:       2019-02-15
author:     biggan
header-img: img/post-bg-swift.jpg
catalog: true
tags:
    - Trick
typora-root-url: ..
---

### 常用的代码段



##### 文件读写

- txt文件

  ```python
  '''文件读'''
  with open('data_path', 'r', encoding='utf-8') as f:
      f.read()		#读取整个文件，返回结果为str类型
      f.readline()	#读取一行，返回结果为str类型
      f.readlines()	#读取整个文件，返回结果为list类型，list的元素为str类型的一行文本
      
  '''文件写'''
  with open('data_path', 'w', encoding='utf-8') as f:
      f.write(string)	#string为要写入的str类型文本，如果用此方法按行写，子文本末尾要加'\n'
      f.writelines(list_str)	#list_str为要写入的list类型文本列表，list元素为str类型文本
      
  ```

- csv文件

  ```python
  import pandas as pd
  
  '''文件读'''
  data = pd.read_csv('data_path', header=None, names=['ID', 'subject'], sep='\t', encoding='utf-8')
  # 文件第一行为列标题时，header不用指定。若文件没有列标题,header要指定为None。
  # names重新命名类标题
  # sep为一行记录中各元素的分割符，默认是','。
  
  '''文件写'''
  writer = pd.DataFrame({"ID": test_id, "Pred": test_id})	#test_id、test_id为待写入文件#的numpy类型数据
  writer.to_csv('filename.csv', index=False, header=False)#index表示是否写入行标题，#header表示是否写入列标题
  ```

  

##### 统计词频、构建词汇表

```python
'''data为语料列表，列表的一个元素表示一个str类型的文本'''

def build_vocabulary(data):
	
	# 词汇表中的词要求在语料中出现次数 >= min_count
	min_count = 2

	# 统计词频
    #vocab为词频表，key为单词，value为词频
	vocab = {}
	for text in data:
		for word in nltk.word_tokenize(text):
			vocab[word] = vocab.get(word, 0) + 1
	vocab = {i: j for i, j in vocab.items() if j >= min_count}  # 去掉低频词

	# 单词以及单词的标签映射表
	# id2word
	id2word = {}
	id2word['PAD'] = 0
	id2word['UNK'] = 1  # pad: 0, unk: 1 (填充词，未登录词)
	id2word = {i + 2: j for i, j in enumerate(vocab)}
	# word2id
	word2id = {j: i for i, j in id2word.items()}
    
    return id2word, word2id
```



##### 批量文本填充

``` python
'''X是一个语料list，其元素是一个文本list，文本list的元素是单词对应的id'''
def seq_padding(X):
    L = [len(x) for x in X]	
    ML = max(L)		# 最大句长
    return [x + [0] * (ML - len(x)) for x in X]
```



##### 持久化某一变量或对象

```python
import pickle as pk

'''data_info为要持久化的变量或对象'''
data_info = {
        'train':train_data,
        'test':test_data,
        'train_labels':train_labels,
        'test_id':test_id,
        'id2word':id2word,
        'word2id':word2id,
        'vocab':vocab
    }

'''保存到相应文件'''
pk.dump(data_info, open('../data/data_info.sav', 'wb'))

'''加载持久化的对象'''
data_info = pk.load(open('../data/data_info.sav', 'rb'))

```



##### 设置随即种子，使结果可复现

```python
def seed_set(seed=2019):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```



##### 时间相关

``` python
'''1.获取当前时间''' 
import time
time1 = time.localtime()    # 获取当前的时间
print(time1)		# time.struct_time(tm_year=2019, tm_mon=12, tm_mday=14, tm_hour=21, tm_min=10, tm_sec=0, tm_wday=5, tm_yday=348, tm_isdst=0)

str_time = str(time1.tm_mon) + '.' + str(time1.tm_mday) + '_' + str(time1.tm_hour) + ':' + str(time1.tm_min) # 获取时间字符串形式
print(str_time)		# 12.14_21:10

localtime = time.asctime(time1) # 获取可读时间（一种格式化方式）
print(localtime)	# Sat Dec 14 21:10:00 2019

strftime = time.strftime("%Y-%m-%d %H:%M:%S", time1)    # 自定义时间的格式化形式
print(strftime)		# 2019-12-14 21:10:00


'''2.时间间隔'''
import time
import datetime
time1 = time.time()	# 每个时间戳都以自从1970年1月1日午夜（历元）经过了多长时间来表示。
time.sleep(3)
time2 = time.time()
delta_time = time2 -time1	# 时间间隔是以秒为单位的浮点小数。
print(delta_time)	# 3.000441789627075
delta_time = datetime.timedelta(seconds=delta_time)
print(delta_time)	# 0:00:03.000442
```

