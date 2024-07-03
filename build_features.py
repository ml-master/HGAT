#!/user/bin/env python
# -*- coding: utf-8 -*-
import networkx
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import pickle as pkl
from nltk.tokenize import WordPunctTokenizer
import os
from utils import sample, preprocess_corpus_notDropEntity, load_stopwords

import jieba
DATASETS = 'gossipcop'



rootpath = './'
datapath = rootpath + 'data/{}/'.format(DATASETS)

def tokenize(sen):
    return WordPunctTokenizer().tokenize(sen)
    # return jieba.cut(sen)


def build_entity_feature_with_description(datapath, stopwords=list()):
    with open(datapath + 'model_network_sampled.pkl', 'rb') as f:
        g = pkl.load(f)
    nodesset = set(g.nodes())
    entityIndex = []
    corpus = []
    cnt = 0
    t = 0
    for i in tqdm(range(40), desc="Read desc: "):
        # if i==30:
        #     continue
        filename = str(i).zfill(4)
        with open("./data/data/wikiAbstract/"+filename, 'r') as f:
            for line in f:
                t+=1
                if len(line.strip('\n').split('\t'))!=2:
                    print(line.strip('\n').split('\t'))
                ent, desc = line.strip('\n').split('\t')
                entity = ent.replace(" ", "_")
                if entity in nodesset:
                    if entity not in entityIndex:
                        entityIndex.append(entity)
                        cnt += 1
                    else:
                        print('error')
                    # 分词
                    content = tokenize(desc)
                    """数据预处理，word.isalpha() 判断字符串 word 是否只包含字母字符,我们不考虑非英文字符
                    """
                    content = ' '.join([ word.lower() for word in content if word.isalpha() ])
                    corpus.append(content)

    print(len(corpus), len(entityIndex))
    """
    entity description的预处理：
    使用 CountVectorizer 将文本语料库（corpus）转换为词频矩阵。忽略在整个语料库中出现频率低于 min_df 参数值的词，并且排除指定的停用词
    返回值 X 是一个稀疏矩阵，其中行表示文档，列表示词汇，矩阵中的值表示词汇在每个文档中的出现次数
    """
    vectorizer = CountVectorizer(min_df = 10, stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)
    print("Entity feature shape: ", X.shape)
    # 用文本特征表示方法TF-IDF得到embedding
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    print("Caculated! Saving...")
    with open(datapath+"vectorizer_model.pkl", 'wb') as f:
        pkl.dump(vectorizer, f)
    with open(datapath+"transformer_model.pkl", 'wb') as f:
        pkl.dump(transformer, f)
    with open(datapath+"features_entity_descBOW.pkl", 'wb') as f:
        pkl.dump(X, f)
    with open(datapath+"features_entity_descTFIDF.pkl", 'wb') as f:
        pkl.dump(tfidf, f)
    with open(datapath+"features_entity_index_desc.pkl", 'wb') as f:
        pkl.dump(entityIndex, f)
    print("done!")

def build_text_feature(datapath, DATASETS, rho=0.3, lp=0.5, stopwords=list()):
    train, vali, test, alltext = sample(datapath=datapath, DATASETS=DATASETS, resample=False)
    # 这里先把未替换的ind-content对存在字典中
    pre_replace = dict()
    index2ind = {}
    cnt = 0
    corpus = []
    involved_entity = set()
    
    with open("{}{}.txt".format(datapath, DATASETS), 'r', encoding='utf8') as f:
        for line in f:
            ind, cate, content = line.strip('\n').split('\t')
            if ind not in alltext:
                continue
            pre_replace[ind] = content.lower()
            content = pre_replace[ind]
            corpus.append(content)
            index2ind[cnt] = ind
            cnt += 1
    print(len(pre_replace))

    print("loading entities...")
    with open('{}{}2entity.txt'.format(datapath, DATASETS), 'r', encoding='utf8') as f:
        for line in tqdm(f):
            if 'null' in line:
                continue
            ind, entityList = line.strip('\n').split('\t')
            # ind = int(ind)
            if ind not in pre_replace or entityList == '':
                continue
            
            entityList = json.loads(entityList)
            for d in entityList:
                if d['rho'] < rho:
                    continue
                if d['link_probability'] < lp:
                    continue
                if 'title' not in d:
                    print("An entity with no title, whose spot is: {}".format(d['spot']))
                    continue
                ent = d['title'].replace(" ", '')
                involved_entity.add(ent)
                ori = d['spot'].lower()
                content.replace(ori, ent)

            
    len(corpus)
    print("text preprocessing...")
    
    corpus = preprocess_corpus_notDropEntity(corpus,
                        stopwords=stopwords, involved_entity=involved_entity)
    print("text feature transforming...")

    vectorizer = CountVectorizer(min_df=10 if DATASETS != "example" else 0.0, stop_words=stopwords)
    X = vectorizer.fit_transform(corpus)

    with open(datapath + 'TextBoW_model.pkl', 'wb') as f:
        pkl.dump(vectorizer, f)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    print("text feature transformed.")
    
    with open(datapath + "features_BOW.pkl", 'wb') as f:
        pkl.dump(X, f)
    with open(datapath + "features_TFIDF.pkl", 'wb') as f:
        pkl.dump(tfidf, f)
    with open(datapath + "features_index.pkl", 'wb') as f:
        pkl.dump(index2ind, f)
    print(X.shape)

    alllength = sum([len(sentence.split(' ')) for sentence in corpus])
    avg_length = alllength / len(corpus)
    print('train: {}\tvali: {}\ttest: {}'.format(len(train), len(vali), len(test)))
    print('num of all corpus: {}'.format(len(train + vali + test)))
    print('avg of tokens: {:.1f}'.format(avg_length))
    vocab = set()
    for s in corpus:
        vocab.update(s.split(' '))
    print('involved entities: {}'.format(len(involved_entity)))
    print('vocabulary size: {}'.format(len(vocab)))


def build_topic_feature_sklearn(datapath, DATASETS, TopicNum=20, stopwords=list(), train=False):
    # sklearn-lda

    idxlist = []
    corpus = []
    catelist = []
    with open('{}{}.txt'.format(datapath, DATASETS), 'r', encoding='utf8') as f:
        for line in f:
            ind, cate, content = line.strip().split('\t')
            idxlist.append(ind)
            corpus.append(content)
            catelist.append(cate)

    with open(datapath + 'doc_index_LDA.pkl', 'wb') as f:
        pkl.dump(idxlist, f)

    print("text feature transforming...")
    corpus = preprocess_corpus_notDropEntity(corpus,stopwords=stopwords, involved_entity=set())

    with open(datapath + "features_BOW.pkl", 'rb') as f:
        X = pkl.load(f)
    # vocabulary_ 的对照关系，读上面那个bow的模型就可以了
    if train:
        alpha, beta = 0.1, 0.1
        """
        潜在狄利克雷分配（Latent Dirichlet Allocation, LDA）是一种生成模型，广泛应用于主题建模和文档聚类。LDA 模型假设每个文档都是由多个主题混合生成的，而每个主题则是由各种单词按一定概率生成的。
        LDA 模型对text数据进行主题建模，得到文档的主题分布
        """
        lda = LatentDirichletAllocation(n_components=TopicNum, max_iter=1200,
                                        learning_method='batch', n_jobs=-1,
                                        doc_topic_prior=alpha, topic_word_prior=beta,
                                        verbose=1,
                                        )
        lda_feature = lda.fit_transform(X)
        with open(datapath + 'lda_model.pkl', 'wb') as f:
            pkl.dump(lda, f)
        with open(datapath + 'topic_word_distribution.pkl', 'wb') as f:
            pkl.dump(lda.components_, f)
    else:
        with open(datapath + 'lda_model.pkl', 'rb') as f:
            lda = pkl.load(f)
        lda_feature = lda.transform(X)

    with open(datapath + 'doc_topic_distribution.pkl', 'wb') as f:
        pkl.dump(lda_feature, f)



if __name__ == '__main__':
    stopwords = load_stopwords()

    build_entity_feature_with_description(datapath, stopwords=stopwords)
    build_text_feature(datapath, DATASETS, stopwords=stopwords)
    build_topic_feature_sklearn(datapath, DATASETS, stopwords=stopwords, train=True)
