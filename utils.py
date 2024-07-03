#!/user/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
from tqdm import tqdm


def sample(datapath, DATASETS, resample = False, trainNumPerClass=1000):
    if resample:
        X = []
        Y = []
        from sklearn.model_selection import train_test_split
        with open(datapath+'{}.txt'.format(DATASETS), 'r') as f:
            for line in f:
                ind, cat, title = line.strip('\n').split('\t')
                t = cat.lower()
                X.append([ind])
                Y.append(t)

        cateset = list(set(Y))
        catemap = dict()
        for i in range(len(cateset)):
            catemap[cateset[i]] = i
        Y = [catemap[i] for i in Y]
        X = np.array(X)
        Y = np.array(Y)

        trainNum = trainNumPerClass*len(catemap)
        ind_train, ind_test = train_test_split(X,
                                        train_size=trainNum, random_state=1, )
        # 这里的train_size 是一个比例值，它确保了从 ind_test 中提取与 ind_train 相同数量样本的比例
        ind_vali, ind_test = train_test_split(ind_test,
                                        train_size=trainNum/(len(X)-trainNum), random_state=1, )
        train = sum(ind_train.tolist(), [])
        vali = sum(ind_vali.tolist(), [])
        test = sum(ind_test.tolist(), [])

        print( len(train), len(vali), len(test) )
        alltext = set(train + vali + test)
        print( "train: {}\nvali: {}\ntest: {}\nAllTexts: {}".format( len(train), len(vali), len(test), len(alltext)) )

        with open(datapath+'train.list', 'w') as f:
            f.write( '\n'.join(map(str, train)) )
        with open(datapath+'vali.list', 'w') as f:
            f.write( '\n'.join(map(str, vali)) )
        with open(datapath+'test.list', 'w') as f:
            f.write( '\n'.join(map(str, test)) )
    else:
        train = []
        vali = []
        test = []
        with open(datapath + 'train.list', 'r') as f:
            for line in f:
                train.append(line.strip())
        with open(datapath + 'vali.list', 'r') as f:
            for line in f:
                vali.append(line.strip())
        with open(datapath + 'test.list', 'r') as f:
            for line in f:
                test.append(line.strip())
        alltext = set(train + vali + test)

    return train, vali, test, alltext

def tokenize(sen):
    return WordPunctTokenizer().tokenize(sen)


def preprocess_corpus_notDropEntity(corpus, stopwords, involved_entity):
    # 文本预处理，停用词，非英文字母
    corpus1 = [[word.lower() for word in tokenize(sentence)] for sentence in tqdm(corpus)]
    corpus2 = [[word for word in sentence if word.isalpha() if word not in stopwords] for sentence in tqdm(corpus1)]
    """defaultdict(int) 创建了一个默认字典，其中的值的默认类型是整数（int）。这意味着当我们访问字典中不存在的键时，会自动创建该键，并且将其对应的值设置为整数类型的默认值，也就是 0
    """
    all_words = defaultdict(int)
    for c in tqdm(corpus2):
        for w in c:
            all_words[w] += 1
    # 文本预处理，出现频次小于5的 & 不是实体
    low_freq = set(word for word in set(all_words) if all_words[word] < 5 and word not in involved_entity)
    text = [[word for word in sentence if word not in low_freq] for sentence in tqdm(corpus2)]
    ans = [' '.join(i) for i in text]
    return ans

def load_stopwords(filepath='./data/stopwords_en.txt'):
    stopwords = set()
    with open(filepath, 'r') as f:
        for line in f:
            stopwords.add(line.strip())
    print(len(stopwords))
    return list(stopwords)