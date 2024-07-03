#!/user/bin/env python
# -*- coding: utf-8 -*-

import networkx
import json
import pickle
import gensim
from tqdm import tqdm

from utils import sample

DATASETS = 'gossipcop'

NumOfTrainTextPerClass = 40
TOPK = 10
SIM_MIN = 0.5

rootpath = './'
datapath = rootpath + 'data/{}/'.format(DATASETS)
g = networkx.Graph()
train, vali, test, alltext = sample(datapath=datapath, DATASETS=DATASETS, resample=True,
                                    trainNumPerClass=NumOfTrainTextPerClass)

# load text-entity
entitySet = set()
rho = 0.2
link_probability = 0.2
noEntity = set()
with open(datapath+'{}2entity.txt'.format(DATASETS), 'r') as f:
    for line in tqdm(f, desc="text-ent: "):
        if 'null' in line:
            continue
        # if len(line.strip('\n').split('\t'))!=2:
        #     tmp = line.strip('\n').split('\t')
        #     print(line.strip('\n').split('\t'))
        ind, entityList = line.strip('\n').split('\t')
        # ind = int(ind)
        if ind not in alltext or entityList=='':
            continue
        entityList = json.loads(entityList)
        entities = [(d['title'].replace(" ", '_'), d['rho'], d['link_probability'])
                        for d in entityList if 'title' in d and float(d['rho']) > rho and float(d['link_probability']) > link_probability]
        entitySet.update([d['title'].replace(" ", '_')
                        for d in entityList if 'title' in d and float(d['rho']) > rho and float(d['link_probability']) > link_probability])
        """
        ind是对应的文本的idx
        (ind, e[0], {'rho': e[1], 'link_probability': e[2]}): 这是一个三元组，表示一条边的信息，包括起始节点 ind、终止节点 e[0] 和边的属性字典 {'rho': e[1], 'link_probability': e[2]}
        """
        g.add_edges_from([(ind, e[0], {'rho': e[1], 'link_probability': e[2]})
                             for e in entities])
        if len(entities) == 0:
            noEntity.add(ind)
            g.add_node(ind)
print("text-entity done.")

# load labels
with open(datapath+'{}.txt'.format(DATASETS), 'r', encoding='utf8') as f:
    for line in tqdm(f,desc="text label: "):
        ind, cate, title = line.strip('\n').split('\t')
        # ind = int(ind)
        if ind not in alltext:
            continue
        if ind not in g.nodes():
            g.add_node(ind)
        g.nodes[ind]['type'] = cate


# load similarities between entities
print("loading Gensim.word2vec. ")
model = gensim.models.Word2Vec.load(rootpath+'data/data/word2vec/word2vec_gensim_5')
print("word2vec model done.")

# topK + 阈值
sim_min = SIM_MIN
topK = TOPK
el = list(entitySet)
entity_edge = []
cnt_no = 0
cnt_yes = 0
cnt = 0
for i in tqdm(range(len(el)), desc="ent-ent: "):
    simList = []
    topKleft = topK
    for j in range(len(el)):
        if i == j:
            continue
        cnt += 1
        try:
            sim = model.wv.similarity(el[i].lower().strip(')'), el[j].lower().strip(')'))
            cnt_yes += 1
            if sim >= sim_min:
                entity_edge.append( (el[i], el[j], {'sim': sim}) )
                topKleft -= 1
            else:
                simList.append( (sim, el[j]) )
        except Exception as e:
            cnt_no += 1
    simList = sorted(simList, key=(lambda x: x[0]), reverse=True)
    # sim >= sim_min的不够topk个，就选相似度相对高的
    for i in range(min(max(topKleft, 0), len(simList))):
        entity_edge.append( (el[i], simList[i][1], {'sim': simList[i][0]}) )
print(cnt_yes, cnt_no)

g.add_edges_from(entity_edge)

# save the network
with open(datapath+'model_network_sampled.pkl', 'wb') as f:
    pickle.dump(g, f)