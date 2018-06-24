#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:29:29 2018

@author: slave1
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
import progressbar
flatten = lambda l: [item for sublist in l for item in sublist]
import jieba_fast as jieba
jieba.set_dictionary('dict.txt.big.txt')


print(torch.__version__)
print(nltk.__version__)
USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None else LongTensor([word2index["<UNK>"]]))


# Weighting Function
#borrowed image from https://nlp.stanford.edu/pubs/glove.pdf
#def weighting(w_i, w_j):
#    try:
#        x_ij = X_ik[(w_i, w_j)]
#    except:
#        x_ij = 1
#        
#    x_max = 100 #100 # fixed in paper
#    alpha = 0.75
#    
#    if x_ij < x_max:
#        result = (x_ij/x_max)**alpha
#    else:
#        result = 1
#    
#    return result

def SkipGram(data = None, columns = 'comment', cut = True, WINDOW_SIZE = 2, dataname = 'newdata.csv'): # data= datause
    
    #take a look at segmented chinese texts
    try:
        
        dataTrans = data[columns].values.tolist()
    except:
        import pandas as pd
        data = pd.read_csv(dataname)
        dataTrans = data[columns].values.tolist()
    
    # isolate X and y from dataTrans
    X =list(dataTrans)
    
# linear regression and xgboost DS
    with open('stopwords.txt') as f:
        stopwords = f.readlines()
    
    
    if cut:
        combine = []
        count = 0
        print('中文切詞')
        for i in progressbar.progressbar(X):
            cuttext = " ".join(jieba.cut( ''.join(i)  ))
            filtered_words = [word for word in  list(cuttext.split(' ')) if word not in stopwords]
            combine.append(filtered_words)
#            combine.append(cuttext)
            
            count += 1
#            print(round(count /len(data[columns].values),2))
    else:
        combine = data[columns].tolist()
    
    # Exclude sparse words    
    word_count = Counter(flatten(combine))
    MIN_COUNT = 3
    exclude = []
    for w, c in word_count.items():
        if c < MIN_COUNT:
            exclude.append(w)
    
    vocab = list(set(flatten(combine)) - set(exclude))
    
#    vocab = list(set(flatten(combine)))
#    vocab = list(set(flatten(corpus)))
    
    #corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
    #corpus = [[word.lower() for word in sent] for sent in corpus]


    word2index = {}
    for vo in vocab:
        if word2index.get(vo) is None:
#            print(vo)
            word2index[vo] = len(word2index)
            
    index2word={v:k for k, v in word2index.items()}
    
    WINDOW_SIZE = WINDOW_SIZE
    windows =  flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in combine])
    
    train_data = []
    
    g= 0
    print('設定每個字詞的移動窗格與上下文字詞')
    for window in progressbar.progressbar(windows):
        g+=1
#        print( round( g/ len(windows),2))
        for i in range(WINDOW_SIZE * 2 + 1):
            if window[i] in exclude or window[WINDOW_SIZE] in exclude: 
                continue # min_count
            if i == WINDOW_SIZE or window[i] == '<DUMMY>': 
                continue
            train_data.append((window[WINDOW_SIZE], window[i]))
    
    X_p = []
    y_p = []
    
    for tr in train_data:
        X_p.append(prepare_word(tr[0], word2index).view(1, -1))
        y_p.append(prepare_word(tr[1], word2index).view(1, -1))
        
        if USE_CUDA:
            torch.cuda.empty_cache()
            
    train_data = list(zip(X_p, y_p))
    
    
    # Build Unigram Distribution**0.75
    Z = 0.001
    word_count = Counter(flatten(combine))
    num_total_words = sum([c for w, c in word_count.items() if w not in exclude])
    unigram_table = []
    
    for vo in vocab:
        unigram_table.extend([vo] * int(((word_count[vo]/num_total_words)**0.75)/Z))
    print(len(vocab), len(unigram_table))
    
    return train_data,word2index, unigram_table,vocab

def negative_sampling(targets, unigram_table, k):
    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []
        target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]
        while len(nsample) < k: # num of sampling
            neg = random.choice(unigram_table)
            if word2index[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(prepare_sequence(nsample, word2index).view(1, -1))
    
    return torch.cat(neg_samples)


class SkipgramNegSampling(nn.Module):
    
    def __init__(self, vocab_size, projection_dim):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim) # center embedding
        self.embedding_u = nn.Embedding(vocab_size, projection_dim) # out embedding
        self.logsigmoid = nn.LogSigmoid()
                
        initrange = (2.0 / (vocab_size + projection_dim))**0.5 # Xavier init
        self.embedding_v.weight.data.uniform_(-initrange, initrange) # init
        self.embedding_u.weight.data.uniform_(-0.0, 0.0) # init
        
    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words) # B x 1 x D
        target_embeds = self.embedding_u(target_words) # B x 1 x D
        
        neg_embeds = -self.embedding_u(negative_words) # B x K x D
        
        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2) # Bx1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(negs.size(0), -1) # BxK -> Bx1
        
        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        
        return -torch.mean(loss)
    
    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)
        
        return embeds

#
#train_data,word2index, unigram_table,vocab = SkipGram(data = None, columns = 'comment', cut = True, WINDOW_SIZE = 2, dataname = 'newdata_pytorch.csv')
#
#word2index,unigram_table,vocab = load_Glove_element(path = '/home/slave1/git/MandyThesis_main_wd/glove/')
#train_data=torch.load('glove/train_data.pth')
##np.save('word2index.npy', word2index) 
##np.save('unigram_table.npy', unigram_table) 
##np.save('vocab.npy', vocab) 
###word2index = np.load(word2index_name).item()
##torch.save(train_data, 'train_data.pth') 
#
#EMBEDDING_SIZE = 300
#BATCH_SIZE = 256
#EPOCH = 100
#NEG = 10 # Num of Negative Sampling
#losses = []
#model = SkipgramNegSampling(len(word2index), EMBEDDING_SIZE)
#if USE_CUDA:
#    model = model.cuda()
#    
#optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#for epoch in range(EPOCH):
#    for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
#        
#        inputs, targets = zip(*batch)
#        
#        inputs = torch.cat(inputs) # B x 1
#        targets = torch.cat(targets) # B x 1
#        negs = negative_sampling(targets, unigram_table, NEG)
#        model.zero_grad()
#
#        loss = model(inputs, targets, negs)
#        
#        loss.backward()
#        optimizer.step()
#    
#        losses.append(loss.data.tolist())
#        
#    if epoch % 5 == 0:
#        print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
#        losses = []
##        torch.save(model, 'skipgram.pt')
#        torch.cuda.empty_cache()
#
#model = torch.load('glove/skipgram.pt')

# test

def word_similarity(target, vocab):
    if USE_CUDA:
        target_V = model.prediction(prepare_word(target, word2index))
    else:
        target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target: 
            continue
        
        if USE_CUDA:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        else:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]


#Test
def word_similarity(target, vocab,USE_CUDA, word2index, model, how_many ):
    if USE_CUDA:
        target_V = model.prediction(prepare_word(target, word2index))
    else:
        target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target: 
            continue
        
        if USE_CUDA:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        else:
            vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0] 
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:how_many]

def load_Glove_element(path, trainload = False):
    word2index = np.load(path+'word2index.npy').item()
    unigram_table = np.load(path+'unigram_table.npy') 
    vocab = np.load(path+'vocab.npy') 
    if trainload:
        train_data = torch.load('train_data.pth')
        return train_data, word2index,unigram_table,vocab
    else:
        return word2index,unigram_table,vocab
