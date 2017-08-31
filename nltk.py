#coding:utf-8
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter


lemmatizer = WordNetLemmatizer() #WordNetLemmatizer词型转换,将do、did、done都能统一的返回do
hm_lines = 10000000

#提取文本中关键的词---》》[like, handsome, beautiful,......]
def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines() #按行读取文件
            for l in contents[: hm_lines]:
                all_words = word_tokenize(l.lower()) #将一行语句拆分为单词
                lexicon += list(all_words) #lexicon存储词list
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon] #词型转换,例如将do、did、done都能统一的返回do
    w_counts = Counter(lexicon) #对不同的词数量统计,返回的是字典{'the':1000,'love':20......}
    l2 = [] #存储关键的词
    for key in w_counts:
        if 1000 > w_counts[key] > 50: #出现频率较高的词如：the is,忽略,出现频率较低的词忽略
            l2.append(key)
    #print(len(l2)) #总共有多少关键词
    return l2

#将文本转为特征数据集
def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower()) #将一行语句拆分为单词
            current_words = [lemmatizer.lemmatize(i) for i in current_words] #词型转换,例如将do、did、done都能统一的返回do
            features = np.zeros(len(lexicon)) #定义一个全0的关键词向量
            for word in current_words:
                if word.lower() in lexicon: #如果某一行中的某个单词在关键词中出现
                    index_value = lexicon.index(word.lower()) #找出这个词的索引
                    features[index_value] += 1 #将关键词向量在这个索引处加1
            features = list(features)
            featureset.append([features, classification])
            """
            [like, handsome, beautiful,......]关键词向量
            [like, handsome, beautiful,handsome,handsome,handsome]第一行所提取的词
            [1,4,1,0,0,0,......]第一行对应的词向量
            featureset:
            [
            data                 label
            [1,4,1,0,0,0,......,[1,0]],
            [2,4,1,0,0,2,......,[0,1]],
            ]
            """
    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.3):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features) #打乱顺序
    features = np.array(features) #将list转为array
    """
    featureset:
    np.array
    [
    data                 label
    [1,4,1,0,0,0,......,[1,0]],
    [2,4,1,0,0,2,......,[0,1]],
    ]
    """
    test_size = int(test_size * len(features))
    train_x = list(features[:, 0][:test_size])
    train_y = list(features[:, 1][:test_size])
    test_x = list(features[:, 0][test_size:])
    test_y = list(features[:, 1][test_size:])
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f) #存入pickle文件